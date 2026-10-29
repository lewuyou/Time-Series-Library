# -*- coding: utf-8 -*-
"""批量导出股票池数据，供共享模型联合训练（完整流程第 2 步）。

完整流程（第 1~4 步）：
    第 1 步 粗筛股票池
        python select_universe.py --db ../AutoTS/download/autots.duckdb \
            --out dataset/global_001/universe.csv \
            --top-mv 3000 --min-years 3 --baidu-fresh-days 30

    第 2 步 批量导出（本脚本）
        python export_global.py --db ../AutoTS/download/autots.duckdb \
            --universe-csv dataset/global_001/universe.csv \
            --data-root dataset/global_001 \
            --workers 4 --seq-len 60 --test-days 30 --val-days 60
        输出 dataset/global_001/export_results.csv，只有 status == ok
        的股票进入共享训练。

    第 3 步 联合训练
        python train_global_timexer.py --mode train \
            --data-root dataset/global_001 \
            --out-dir checkpoints/global_timexer_001 \
            --epochs 30 --patience 5 --batch-size 256 --workers 0 \
            --d-model 64 --e-layers 2 --patch-len 10 --device auto
        输出 checkpoints/global_timexer_001/ 下的 checkpoint.pth、
        history.json、schema.json、test_predictions.csv、
        test_metrics.json、test_metrics_by_stock.csv。

    第 4 步 推理预测（复用训练期预处理生成最新窗口）
        python dataset.py --mode infer --db ../AutoTS/download/autots.duckdb \
            --preprocessor dataset/global_001/601398/preprocessing.json \
            --out-dir dataset/global_infer_001 --fail-on-history-drift
        python train_global_timexer.py --mode predict \
            --checkpoint checkpoints/global_timexer_001/checkpoint.pth \
            --data-dir dataset/global_infer_001/601398 --device auto

抽查导出质量（第 2 步之后、训练之前建议抽查几只）：
    python dataset.py --mode validate --data-dir dataset/global_001/601398
    覆盖窗口索引、标签一致性、切分边界、预处理与可用时间审计。

设计要点：
    - 每只股票独立导出、统一日期边界；特征并集 + 缺失标记由
      train_global_timexer.py 统一处理，本脚本只负责单股导出与复检。
    - 切分口径（两种模式）：
      1) 自动倒推（默认，与单次评估一致）：最后 test_days（默认 30）个
         交易日为目标日的 test 集；val_end = 倒数第 test_days+1 个交易日，
         train_end = 再往前 val_days 个交易日，保证 train/val 的输入窗口
         与标签完全不触碰最后 30 天数据；
      2) 显式边界（walk-forward 每折）：--as-of、--train-end、--val-end
         必须同时提供，均为交易日历内日期且满足 train-end < val-end <
         as-of <= 库内最新行情日；此时 --test-days/--val-days 只用于
         登记与样本量下限校验，实际边界以显式日期为准。可选
         --train-start 限制最早训练目标日期（rolling 训练窗口），
         不提供时使用全部可用历史（expanding）。
    - dataset.py 会按训练期覆盖率过滤特征列；本脚本导出后检查
      feature_columns 仍保留 baidu_* 与 rzrq_*，缺失者标记为
      filtered_no_baidu / filtered_no_rzrq，不进入共享训练。
    - 断点续跑：data-root/export_config.json 登记导出配置（含
      train_start）；配置一致的重复运行复用已有股票目录（含
      sample_selection 比对，避免相同截止日、不同训练起点的目录被
      误复用），配置变化必须换新的 data-root。
    - 建议固定项目 git commit，避免官方 TimeXer.py / embedding 实现变化
      后与 checkpoint 结构不兼容。

walk-forward 滚动评估（第 5 步，可选）：
    python walk_forward.py --db ... --universe-csv ... --run-root runs/wf_001 \
        --first-test-date 2024-01-02 --last-test-date 2024-12-31 \
        --val-days 60 --test-days 20 ...
    调度器按交易日历生成互不重叠的测试折，每折用显式边界调用本脚本
    独立导出（独立拟合预处理），再调用 train_global_timexer.py 从头
    训练，最后拼接所有折的样本外预测。

注意：
    select_universe.py 的粗筛（行情天数等）不保证百度/两融各自覆盖
    训练期，最终以本脚本导出后的复检结果为准。
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

import dataset as ds

# 进程级共享状态，由 init_worker 填充。
_G = {}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default=ds.DEFAULT_DB)
    p.add_argument("--universe-csv", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--val-days", type=int, default=60)
    p.add_argument("--min-train-samples", type=int, default=250)
    p.add_argument("--min-eval-samples", type=int, default=20)
    p.add_argument(
        "--as-of",
        help="显式切分：本折数据截止日（测试结束日）；与 --train-end、"
             "--val-end 一起提供，用于 walk-forward 历史折",
    )
    p.add_argument(
        "--train-end",
        help="显式切分：训练目标日上限（交易日历内日期）",
    )
    p.add_argument(
        "--val-end",
        help="显式切分：验证目标日上限（交易日历内日期）",
    )
    p.add_argument(
        "--train-start",
        help="最早训练目标日期（rolling 训练窗口下限）；不提供时使用"
             "全部可用历史（expanding）",
    )
    return p.parse_args()


def resolve_split(db, test_days, val_days, as_of=None, train_end=None,
                  val_end=None, train_start=None):
    """返回 (as_of, train_end, val_end, test_start)。

    与 dataset.py 同源的交易日历。两种模式：
      - 未提供显式边界：按库内最新行情日自动倒推（与历史单次评估一致）；
      - 显式边界：as_of/train_end/val_end 必须同时提供，均为交易日历内
        日期且满足 train_end < val_end < as_of <= 库内最新行情日；
        train_start 可选，限制最早训练目标日期（rolling），必须不晚于
        train_end。
    """
    explicit = [as_of, train_end, val_end]
    if any(x is not None for x in explicit):
        if not all(x is not None for x in explicit):
            raise ValueError(
                "显式切分必须同时提供 --as-of、--train-end、--val-end"
            )
        as_of = pd.Timestamp(as_of).normalize()
        train_end = pd.Timestamp(train_end).normalize()
        val_end = pd.Timestamp(val_end).normalize()

    con = duckdb.connect(db, read_only=True)
    try:
        calendar, _ = ds.load_calendar(
            con, None, ds.DEFAULT_CALENDAR_START, 50
        )
        value = con.execute(
            "SELECT max(date) FROM akshare_tx WHERE adjust='hfq'"
        ).fetchone()[0]
        if value is None:
            raise ValueError("数据库没有后复权行情")
        latest = pd.Timestamp(value).normalize()

        if as_of is None:
            as_of = latest
            if as_of not in calendar:
                raise ValueError(f"as-of {as_of.date()} 不在交易日历内")
            upto = calendar[calendar <= as_of]
            # 全局日期的粗略下限检查；单股样本是否达标由 prepare_one() 判断。
            need = test_days + val_days + 300
            if len(upto) < need:
                raise ValueError(f"交易日不足：{len(upto)} < {need}")
            val_end = upto[-(test_days + 1)]
            train_end = upto[-(test_days + val_days + 1)]
        elif as_of > latest:
            raise ValueError(
                f"as-of {as_of.date()} 晚于库内最新行情日 {latest.date()}；"
                "walk-forward 历史折的 as-of 不能超过库内数据"
            )

        for name, boundary in (
            ("as-of", as_of),
            ("train-end", train_end),
            ("val-end", val_end),
        ):
            if boundary not in calendar:
                raise ValueError(f"{name}={boundary.date()} 不在交易日历")

        if not train_end < val_end < as_of:
            raise ValueError("必须满足 train-end < val-end < as-of")

        if calendar.get_loc(as_of) + 1 >= len(calendar):
            raise ValueError("交易日历缺少 as-of 之后的下一交易日")

        if train_start is not None:
            start = pd.Timestamp(train_start).normalize()
            if start not in calendar:
                raise ValueError(
                    f"train-start={start.date()} 不在交易日历"
                )
            if start > train_end:
                raise ValueError("train-start 不能晚于 train-end")

        test_start = calendar[calendar.get_loc(val_end) + 1]
        return as_of, train_end, val_end, test_start
    finally:
        con.close()


def init_worker(db, data_root, as_of, train_end, val_end,
                seq_len, min_train_samples, min_eval_samples,
                train_start=None):
    """进程级初始化：只读 duckdb 连接 + 交易日历 + 预处理配置。

    train_start 为 "YYYY-MM-DD" 或 None：rolling 训练窗口的最早目标日，
    传入 dataset.py 的 --start（保留原始历史用于窗口与指标预热）。
    """
    con = duckdb.connect(db, read_only=True)
    calendar, calendar_info = ds.load_calendar(
        con, None, ds.DEFAULT_CALENDAR_START, 50
    )
    policies = ds.load_policies(None)
    macro_fields = ["close", "volume"]
    config = {
        "seq_len": seq_len,
        "return_type": "simple",
        "macro_fields": macro_fields,
        "macro_symbols": None,
        "policies": policies,
        "decision_hour": 8,
        "decision_minute": 30,
        "baidu_keyword": None,
    }
    max_lookback = max(
        x["max_age_days"] + x["lag_days"] for x in policies.values()
    )
    aux_start = calendar.min() - pd.Timedelta(days=max_lookback + 10)
    next_session = calendar[calendar.get_loc(as_of) + 1]
    macro_specs = ds.load_macro(con, macro_fields, None, aux_start, next_session)
    config["macro_symbols"] = sorted({x["symbol"] for x in macro_specs})

    args_ns = argparse.Namespace(
        mode="prepare", db=db, out_dir=str(data_root),
        calendar=None, calendar_start=ds.DEFAULT_CALENDAR_START,
        min_market_symbols=50,
        symbol=None, industry=None, top=None, baidu_keyword=None,
        as_of=str(as_of.date()), start=train_start, tail=None,
        train_end=str(train_end.date()), val_end=str(val_end.date()),
        seq_len=seq_len, return_type="simple",
        min_train_samples=min_train_samples,
        min_eval_samples=min_eval_samples,
        min_feature_coverage=0.60, min_observations=100,
        macro_fields=macro_fields, macro_symbols=None, source_policy=None,
        decision_hour=8, decision_minute=30, preprocessor=None,
        fail_on_history_drift=False, allow_calendar_drift=False,
        continue_on_error=True,
    )
    _G.update(con=con, calendar=calendar, calendar_info=calendar_info,
              config=config, macro_specs=macro_specs, args=args_ns,
              as_of=as_of,
              # 日历指纹每进程只算一次，供 export_one 复用校验。
              calendar_fingerprint=ds.calendar_fingerprint(
                  calendar, as_of, calendar_info
              ))


def export_one(row):
    """单只股票：断点导出 → 完整性检查 → 配置一致 → 辅助特征复检。"""
    symbol = ds.norm_symbol(row["symbol"])
    row = dict(row, symbol=symbol)
    root = Path(_G["args"].out_dir) / symbol

    try:
        if not root.exists():
            ds.prepare_one(
                _G["con"],
                row,
                _G["args"],
                _G["calendar"],
                _G["as_of"],
                _G["macro_specs"],
                _G["config"],
                _G["calendar_info"],
            )

        required = [
            "feature_matrix.npy",
            "samples.csv",
            "preprocessing.json",
            "dates.npy",
            "manifest.json",
        ]
        for name in required:
            if not (root / name).exists():
                raise ValueError(f"不完整导出目录：缺少 {name}")

        prep = json.loads(
            (root / "preprocessing.json").read_text(encoding="utf-8")
        )

        # 复用已有目录前做完整口径校验：预处理配置（含百度关键词来源）、
        # 日历指纹、数据截止日必须与本次任务一致，避免混用不同口径的导出。
        args = _G["args"]
        expected_config = dict(_G["config"])
        expected_config["baidu_keyword"] = (
            expected_config.get("baidu_keyword") or str(row["name"])
        )
        expected_calendar = _G["calendar_fingerprint"]
        if (
            prep.get("symbol") != symbol
            or prep.get("version") != ds.VERSION
            or prep.get("config") != expected_config
            or prep.get("calendar") != expected_calendar
            or prep.get("min_feature_coverage") != args.min_feature_coverage
            or prep.get("min_observations") != args.min_observations
            or prep.get("train_end") != args.train_end
            or prep.get("val_end") != args.val_end
            or prep.get("sample_selection") != {
                "start": args.start, "tail": args.tail,
            }
        ):
            raise ValueError("已有导出的预处理配置或日历与本次任务不一致")

        manifest = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("as_of") != str(_G["as_of"].date()):
            raise ValueError("已有导出的 as-of 与本次任务不一致")

        bases = [
            c for c in prep["feature_columns"]
            if not c.endswith(("__missing", "__age_days"))
        ]

        if not any(c.startswith("baidu_") for c in bases):
            return {"symbol": symbol, "status": "filtered_no_baidu"}
        if not any(c.startswith("rzrq_") for c in bases):
            return {"symbol": symbol, "status": "filtered_no_rzrq"}

        return {"symbol": symbol, "status": "ok"}

    except Exception as exc:
        return {
            "symbol": symbol,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    a = parse_args()

    if min(
        a.workers, a.seq_len, a.test_days, a.val_days,
        a.min_train_samples, a.min_eval_samples,
    ) < 1:
        raise ValueError("数量参数必须为正数")
    if a.seq_len < 2:
        raise ValueError("--seq-len 必须 >= 2")
    if min(a.test_days, a.val_days) < a.min_eval_samples:
        # 只是必要条件：停牌等情况仍可能导致单股有效样本不足，
        # 最终由 prepare_one() 判断。
        raise ValueError("test-days 和 val-days 不能小于 min-eval-samples")

    rows = pd.read_csv(
        a.universe_csv, dtype={"symbol": str}
    ).to_dict("records")
    if not rows:
        raise ValueError("股票池为空")

    symbols = [ds.norm_symbol(r["symbol"]) for r in rows]
    if len(symbols) != len(set(symbols)):
        raise ValueError("股票池存在重复股票代码")

    as_of, train_end, val_end, test_start = resolve_split(
        a.db, a.test_days, a.val_days,
        as_of=a.as_of, train_end=a.train_end, val_end=a.val_end,
        train_start=a.train_start,
    )
    # 统一为 "YYYY-MM-DD"，与 preprocessing.json 的 sample_selection 一致。
    train_start = (
        str(pd.Timestamp(a.train_start).normalize().date())
        if a.train_start else None
    )

    root = Path(a.data_root)
    root.mkdir(parents=True, exist_ok=True)

    spec = {
        "export_version": "global_export_wf_v1",
        "db": str(Path(a.db).resolve()),
        "as_of": str(as_of.date()),
        "train_end": str(train_end.date()),
        "val_end": str(val_end.date()),
        "test_start": str(test_start.date()),
        "train_start": train_start,
        "seq_len": a.seq_len,
        "min_train_samples": a.min_train_samples,
        "min_eval_samples": a.min_eval_samples,
        "symbols": sorted(symbols),
        "dataset_version": ds.VERSION,
        "calendar_start": ds.DEFAULT_CALENDAR_START,
        "min_market_symbols": 50,
        "policies": ds.load_policies(None),
        "macro_fields": ["close", "volume"],
        "min_feature_coverage": 0.60,
        "min_observations": 100,
        "return_type": "simple",
        "decision_hour": 8,
        "decision_minute": 30,
        "universe_sha256": ds.sha256_file(a.universe_csv),
        "dataset_script_sha256": ds.sha256_file(ds.__file__),
        "export_script_sha256": ds.sha256_file(__file__),
    }

    config_path = root / "export_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != spec:
            raise ValueError("导出配置已变化，请使用新的 data-root")
    elif any(root.glob("*/preprocessing.json")):
        raise ValueError("目录中存在未登记的旧导出，请使用新的 data-root")

    ds.json_write(config_path, spec)
    print(json.dumps(spec, ensure_ascii=False, indent=2))

    results = []
    with ProcessPoolExecutor(
        max_workers=a.workers,
        initializer=init_worker,
        initargs=(
            a.db,
            str(root),
            as_of,
            train_end,
            val_end,
            a.seq_len,
            a.min_train_samples,
            a.min_eval_samples,
            train_start,
        ),
    ) as pool:
        futures = {
            pool.submit(export_one, row): ds.norm_symbol(row["symbol"])
            for row in rows
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                # worker 初始化失败、进程崩溃等异常会绕过 export_one
                # 内部的 try，这里兜底记录，避免整个进程池静默失败。
                result = {
                    "symbol": symbol,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            print(result, flush=True)

    report = pd.DataFrame(results).sort_values("symbol")
    temporary = root / "export_results.tmp.csv"
    report.to_csv(temporary, index=False)
    temporary.replace(root / "export_results.csv")
    print(report["status"].value_counts().to_dict())

    if (report["status"] == "error").any():
        # filtered_* 是按设计排除，error 是处理失败，不能当作"不合格"。
        raise SystemExit(
            "存在导出错误；请查看 export_results.csv，修复后重跑"
        )


if __name__ == "__main__":
    main()

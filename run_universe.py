#!/usr/bin/env python
"""批量导出 + 训练符合条件股票池（不改动现有 dataset.py / train_timexer.py）。

条件（由 select_universe.py 预筛 + 本脚本导出后复检）：
  1) 市值排名前 3000；
  2) 行情跨度与交易日数不少于 3 年；
  3) 入库有 rzrq 记录、有新鲜的 baidu 关键词数据；
  4) 导出后的特征列中仍保留 baidu_* 与 rzrq_*（dataset.py 会按训练期
     覆盖率过滤列，覆盖率不足的辅助源会被丢弃，故必须复检）。

切分口径：最后 test_days（默认 30）个交易日为目标日的 test 集；
val_end = 倒数第 test_days+1 个交易日，train_end = 再往前 val_days 个
交易日，保证 train/val 的输入窗口与标签完全不触碰最后 30 天数据。

流程：多进程各自持有 duckdb 只读连接，导出（dataset.prepare_one）→
复检特征列 → subprocess 调 train_timexer.py 训练；断点续跑以
metrics.json / run_results.jsonl 为准。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

import dataset as ds

PROJECT_ROOT = Path(__file__).resolve().parent

# 进程级共享状态，由 init_worker 填充。
_G = {}

# run_results.jsonl 中的终态：续跑时直接跳过。
TERMINAL_STATUS = {"ok", "skipped_done", "filtered_no_baidu", "filtered_no_rzrq"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(PROJECT_ROOT.parent / "AutoTS" / "download" / "autots.duckdb"))
    p.add_argument("--universe-csv", default=str(PROJECT_ROOT / "dataset" / "universe_run_001" / "universe.csv"))
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "dataset" / "universe_run_001"))
    p.add_argument("--train-root", default=str(PROJECT_ROOT / "checkpoints" / "universe_001"))
    p.add_argument("--as-of", help="数据截止交易日，默认取库内最新行情日")
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--val-days", type=int, default=60)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--min-train-samples", type=int, default=250)
    p.add_argument("--min-eval-samples", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--timeout", type=int, default=3600, help="单只股票训练子进程超时秒数")
    p.add_argument("--limit", type=int, help="仅处理前 N 只（调试用）")
    p.add_argument("train_extra", nargs=argparse.REMAINDER, help="透传给 train_timexer.py 的额外参数，如 -- --epochs 30")
    return p.parse_args()


def resolve_split(db, as_of_arg, test_days, val_days):
    """返回 (as_of, train_end, val_end)。与 dataset.py 同源的交易日历。"""
    con = duckdb.connect(db, read_only=True)
    try:
        calendar, _ = ds.load_calendar(con, None, None, 50)
        if as_of_arg:
            as_of = pd.Timestamp(as_of_arg).normalize()
        else:
            as_of = pd.Timestamp(
                con.execute("SELECT max(date) FROM akshare_tx WHERE adjust='hfq'").fetchone()[0]
            ).normalize()
        if as_of not in calendar:
            raise ValueError(f"as-of {as_of.date()} 不在交易日历内")
        upto = calendar[calendar <= as_of]
        need = test_days + val_days + 300
        if len(upto) < need:
            raise ValueError(f"交易日不足：{len(upto)} < {need}")
        val_end = upto[-(test_days + 1)]
        train_end = upto[-(test_days + val_days + 1)]
        test_start = upto[-test_days]
        return as_of, train_end, val_end, test_start
    finally:
        con.close()


def init_worker(db, data_root, train_root, as_of, train_end, val_end,
                seq_len, min_train_samples, min_eval_samples,
                torch_threads, train_extra, timeout):
    con = duckdb.connect(db, read_only=True)
    calendar, calendar_info = ds.load_calendar(con, None, None, 50)
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
    max_lookback = max(x["max_age_days"] + x["lag_days"] for x in policies.values())
    aux_start = calendar.min() - pd.Timedelta(days=max_lookback + 10)
    next_session = calendar[calendar.get_loc(as_of) + 1]
    macro_specs = ds.load_macro(con, macro_fields, None, aux_start, next_session)
    config["macro_symbols"] = sorted({x["symbol"] for x in macro_specs})

    args_ns = argparse.Namespace(
        mode="prepare", db=db, out_dir=str(data_root),
        calendar=None, calendar_start=None, min_market_symbols=50,
        symbol=None, industry=None, top=None, baidu_keyword=None,
        as_of=str(as_of.date()), start=None, tail=None,
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
              as_of=as_of, train_root=Path(train_root),
              torch_threads=torch_threads,
              train_extra=list(train_extra or []), timeout=int(timeout))


def check_aux_features(data_dir):
    """复检导出特征列：返回 (has_baidu, has_rzrq, 特征数)。"""
    prep = json.loads(
        (Path(data_dir) / "preprocessing.json").read_text(encoding="utf-8"))
    cols = prep["feature_columns"]
    return (any(c.startswith("baidu_") for c in cols),
            any(c.startswith("rzrq_") for c in cols),
            len(cols))


def process_stock(row):
    """单只股票：断点检查 → 导出 → 特征复检 → subprocess 训练。"""
    symbol = str(row["symbol"])
    t0 = time.time()
    data_dir = Path(_G["args"].out_dir) / symbol
    train_dir = _G["train_root"] / symbol
    rec = {"symbol": symbol, "name": row.get("name")}
    try:
        if (train_dir / "metrics.json").exists():
            rec["status"] = "skipped_done"
            return rec

        if not data_dir.exists():
            try:
                ds.prepare_one(_G["con"], row, _G["args"], _G["calendar"],
                               _G["as_of"], _G["macro_specs"], _G["config"],
                               _G["calendar_info"])
            except FileExistsError:
                pass  # 并发或历史残留，直接复用

        has_baidu, has_rzrq, n_feat = check_aux_features(data_dir)
        if not has_baidu:
            rec["status"] = "filtered_no_baidu"
            rec["features"] = n_feat
            return rec
        if not has_rzrq:
            rec["status"] = "filtered_no_rzrq"
            rec["features"] = n_feat
            return rec
        rec["features"] = n_feat

        train_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(PROJECT_ROOT / "train_timexer.py"),
               "--data-dir", str(data_dir), "--out-dir", str(train_dir)]
        cmd += list(_G.get("train_extra") or [])
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(_G["torch_threads"])
        env["MKL_NUM_THREADS"] = str(_G["torch_threads"])
        try:
            proc = subprocess.run(
                cmd, cwd=str(PROJECT_ROOT), env=env,
                capture_output=True, text=True, timeout=_G["timeout"])
            if proc.returncode != 0 or not (train_dir / "metrics.json").exists():
                tail = (proc.stderr or proc.stdout or "")[-1500:]
                rec["status"] = "error_train"
                rec["error"] = tail.replace("\n", " | ")[:1500]
                return rec
        except subprocess.TimeoutExpired:
            rec["status"] = "error_train"
            rec["error"] = f"timeout after {_G['timeout']}s"
            return rec

        metrics = json.loads(
            (train_dir / "metrics.json").read_text(encoding="utf-8"))
        rec["status"] = "ok"
        rec["best_epoch"] = metrics["train_config"].get("best_epoch")
        rec["test_metrics"] = metrics.get("test_metrics_simple_return", {}).get("model")
        return rec
    except Exception as exc:  # 导出失败（样本不足、停牌缺口等）
        rec["status"] = "error_export"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:1500]
        return rec
    finally:
        rec["seconds"] = round(time.time() - t0, 1)


def main():
    args = parse_args()
    train_extra = [a for a in args.train_extra if a != "--"]
    df = pd.read_csv(args.universe_csv, dtype={"symbol": str})
    rows = df.to_dict("records")
    if args.limit:
        rows = rows[:args.limit]

    as_of, train_end, val_end, test_start = resolve_split(
        args.db, args.as_of, args.test_days, args.val_days)
    print(f"[SPLIT] as_of={as_of.date()} train_end={train_end.date()} "
          f"val_end={val_end.date()} test={test_start.date()}..{as_of.date()} "
          f"({args.test_days} 个目标交易日)")

    data_root = Path(args.data_root)
    train_root = Path(args.train_root)
    data_root.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)

    out_path = train_root / "run_results.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                prev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if prev.get("status") in TERMINAL_STATUS:
                done.add(str(prev["symbol"]))
    todo = [r for r in rows if str(r["symbol"]) not in done]
    print(f"[PLAN] universe={len(rows)} done={len(done)} todo={len(todo)} "
          f"workers={args.workers}")

    counters = {}
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fout, \
            ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=init_worker,
                initargs=(args.db, str(data_root), str(train_root),
                          as_of, train_end, val_end, args.seq_len,
                          args.min_train_samples, args.min_eval_samples,
                          args.torch_threads, train_extra,
                          args.timeout)) as ex:
        futures = {ex.submit(process_stock, r): str(r["symbol"]) for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            sym = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {"symbol": sym, "status": "error",
                       "error": repr(exc)[:1500]}
            counters[rec["status"]] = counters.get(rec["status"], 0) + 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            status = rec["status"]
            if status.startswith("error") or i <= 5 or i % 20 == 0:
                err = f" {rec.get('error', '')[:120]}" if status.startswith("error") else ""
                print(f"[{i}/{len(todo)}] {sym} {status} "
                      f"{rec.get('seconds', 0)}s{err}", flush=True)
    summary = {
        "split": {"as_of": str(as_of.date()), "train_end": str(train_end.date()),
                  "val_end": str(val_end.date()),
                  "test_start": str(test_start.date()),
                  "test_days": args.test_days, "val_days": args.val_days},
        "universe": len(rows), "todo": len(todo),
        "counters": counters, "seconds": round(time.time() - t0, 1),
        "finished_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    }
    (train_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] {json.dumps(counters, ensure_ascii=False)} "
          f"in {summary['seconds']}s -> {out_path}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

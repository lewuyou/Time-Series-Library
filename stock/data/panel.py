"""训练侧面板数据：装载导出目录、构造全局 schema 与窗口 Dataset。

自 train_global_timexer.py 原样迁出（load_catalog / PanelWindows），
逻辑与产物契约不变；列布局解析统一引用 stock.data.schema。
"""

import bisect
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from stock.data.schema import (
    DROP_BASES,
    TARGET,
    base_name,
    compute_ex_mask,
    exogenous_mask_layout,
    is_calendar_col,
    is_mark_col,
    target_calendar_columns,
)
from stock.utils import read_json


def load_catalog(data_root, expected_layout=None):
    root = Path(data_root)
    report = pd.read_csv(
        root / "export_results.csv", dtype={"symbol": str}
    )
    required = {"symbol", "status"}
    if not required.issubset(report.columns):
        raise ValueError("export_results.csv 缺少必要字段（symbol/status）")
    if report["symbol"].duplicated().any():
        raise ValueError("export_results.csv 存在重复股票")
    if (report["status"] == "error").any():
        raise ValueError(
            "导出报告仍有 error，请先修复导出，避免静默训练部分股票池"
        )

    # 与导出任务登记对比，确保时间边界与窗口长度和本次 data-root
    # 一致，而不只是股票之间相互一致。
    spec = read_json(root / "export_config.json")
    for key in ("train_end", "val_end", "seq_len"):
        if key not in spec:
            raise ValueError(f"export_config.json 缺少字段 {key}")
    layout = spec.get("split_layout", "train_val_test")
    if layout not in ("train_val_test", "train_test"):
        raise ValueError(f"不支持的 split_layout：{layout}")
    if expected_layout is not None and layout != expected_layout:
        raise ValueError(
            f"导出布局 split_layout={layout} 与训练模式要求"
            f" {expected_layout} 不一致；请检查 --data-root"
        )
    # walk-forward / rolling 防错配：单股导出的样本起点必须与导出任务
    # 登记一致，避免相同截止日、不同训练起点的导出混入同一训练。
    expected_selection = {
        "start": spec.get("train_start"),
        "tail": None,
    }

    symbols = sorted(
        report.loc[report.status == "ok", "symbol"].unique().tolist()
    )
    if not symbols:
        raise ValueError("没有导出复检通过的股票")

    records = []
    reference = None

    for symbol in symbols:
        directory = root / symbol
        prep = read_json(directory / "preprocessing.json")
        columns = prep["feature_columns"]

        if prep["symbol"] != symbol:
            raise ValueError(f"{symbol}: 预处理股票代码不匹配")
        if columns[-1] != TARGET or len(columns) != len(set(columns)):
            raise ValueError(f"{symbol}: 特征列定义异常")
        if prep["config"]["return_type"] != "simple":
            raise ValueError("本脚本要求 simple 收益率，请重新导出")
        if prep.get("split_layout", "train_val_test") != layout:
            raise ValueError(f"{symbol}: split_layout 与导出任务登记不一致")
        if prep["train_end"] != spec["train_end"]:
            raise ValueError(f"{symbol}: train_end 与导出任务登记不一致")
        if prep["val_end"] != spec["val_end"]:
            raise ValueError(f"{symbol}: val_end 与导出任务登记不一致")
        if prep["config"]["seq_len"] != spec["seq_len"]:
            raise ValueError(f"{symbol}: seq_len 与导出任务登记不一致")
        if prep.get("sample_selection") != expected_selection:
            raise ValueError(
                f"{symbol}: 样本起点配置与 export_config.json 不一致"
            )

        # baidu_keyword 本来就应该随股票不同，不纳入一致性比较。
        config = {
            k: v for k, v in prep["config"].items()
            if k != "baidu_keyword"
        }
        signature = {
            "config": config,
            "train_end": prep["train_end"],
            "val_end": prep["val_end"],
            "calendar": prep.get("calendar"),
            "version": prep["version"],
        }
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"{symbol}: 时间切分/特征配置/日历不一致")

        bases = [c for c in columns if "__" not in c]
        for prefix in ("baidu_", "rzrq_"):
            if not any(c.startswith(prefix) for c in bases):
                raise ValueError(f"{symbol}: 缺少 {prefix} 数值特征")

        matrix = np.load(
            directory / "feature_matrix.npy", mmap_mode="r"
        )
        dates = np.load(directory / "dates.npy")
        if matrix.shape != (len(dates), len(columns)):
            raise ValueError(f"{symbol}: 矩阵维度不匹配")
        if not np.all(dates[1:] > dates[:-1]):
            raise ValueError(f"{symbol}: 日期未严格递增")
        del matrix

        sample = pd.read_csv(
            directory / "samples.csv",
            usecols=["split", "label_valid", "y_raw"],
        )
        # 训练统计只用可评分训练样本（label_valid）。
        train_y = sample.loc[
            (sample.split == "train") & sample.label_valid, "y_raw"
        ]
        if train_y.empty:
            raise ValueError(f"{symbol}: 没有训练样本")

        records.append({
            "symbol": symbol,
            "root": str(directory),
            "prep": prep,
            "columns": columns,
            "dates": dates,
            "train_mean": float(train_y.mean()),
        })

    # 日历特征列集合必须全股票一致（同一导出规格）。禁止不同日历规格
    # 的导出目录混用，也禁止用补零对齐日历列（会把 0 误当成语义值，
    # 例如把 dow=0 当成周一）；不一致即显式报错，要求换新 data-root 重导。
    calendar_sets = {
        tuple(c for c in r["columns"] if is_calendar_col(c))
        for r in records
    }
    if len(calendar_sets) > 1:
        raise ValueError(
            "各股票日历特征列集合不一致（禁止混用不同日历规格的导出目录，"
            "也不得补零对齐日历列）；请使用同一导出规格的新 data-root 重导"
        )

    # 仅依据导出时在训练期保留的列生成 schema。
    exogenous = sorted({
        c
        for r in records
        for c in r["columns"]
        if c != TARGET and base_name(c) not in DROP_BASES
    })
    if not exogenous:
        raise ValueError("没有可用外生变量")

    # 日历列固定排在外生列末尾（紧邻 TARGET），且按 target_calendar_columns()
    # 的规范顺序（时间标记列在前、旗标列在后）：calendar_mode 的 x_mark
    # 拆分与 ex_mask 重排都依赖这一顺序（见 stock/adapters/timexer.py
    # 与 CALENDAR_DESIGN.md）。
    value_cols = [c for c in exogenous if not is_calendar_col(c)]
    cal_set = {c for c in exogenous if is_calendar_col(c)}
    calendar_cols = [c for c in target_calendar_columns() if c in cal_set]
    calendar_cols += sorted(cal_set - set(calendar_cols))
    return records, value_cols + calendar_cols + [TARGET]


class PanelWindows(Dataset):
    """全局样本索引只定位到某股票内部，绝不跨股票切片。

    layout：
      train_val_test：target <= train_end 为 train，<= val_end 为 val，
        其余为 test（原协议）；
      train_test：target <= train_end（重拟合截止日 fit_end）为 train，
        其余为 test，没有 val。
    样本只装载 label_valid=True 的行（可评分样本；无标签行是覆盖登记）。
    __getitem__ 返回 (x, y, sid, target, y_raw, ex_mask, x_mark, head_cal)：
    y 为标准化 float32 标签（训练/验证损失用），y_raw 为 float64 原始
    标签（评估真值直接取用），x_mark 为时间标记通道（可能为 0 列），
    head_cal 为目标日星期 one-hot（float32 (5,)，供预测头日历拼接；
    由目标日期直接推导，与日历列开关无关）。
    allow_missing：允许个别股票在本 split 全程无有效样本（停牌等；
    覆盖率不决定训练资格），装载时跳过，该股票不参与该 split；
    其余情况样本缺失视为导出/装载错配，直接报错。

    calendar_mode（目标日日历列的喂入方式，训练/推理必须一致）：
      time_marks（默认，官方通道）：3 个时间标记列（target_mark_*）
        从 x 移入 x_mark，5 个旗标列仍留在 x；
      columns：全部日历列随其他特征留在 x 中，不使用 x_mark；
      none：日历列全部移除（关闭全部日历信息，A/B 基线）。
    ex_mask 的长度恒等于 x 中外生 token 数；时间标记 token 与日历
    旗标 token 永不屏蔽（日历列无 __missing 伴随列）。
    """

    def __init__(self, records, schema, split, layout="train_val_test",
                 allow_missing=False, calendar_mode="time_marks"):
        if layout not in ("train_val_test", "train_test"):
            raise ValueError(f"不支持的 split_layout：{layout}")
        if layout == "train_test" and split == "val":
            raise ValueError("train_test 布局没有 val 样本")
        if calendar_mode not in ("time_marks", "columns", "none"):
            raise ValueError(f"不支持的 calendar_mode：{calendar_mode}")

        self.records = records
        self.schema = schema
        self.split = split
        self.layout = layout
        self.calendar_mode = calendar_mode
        self.parts = []
        self.record_ids = []
        self.cache = OrderedDict()
        self.seq_len = records[0]["prep"]["config"]["seq_len"]

        # 日历通道拆分位置（依赖 load_catalog 的"日历列排在外生列末尾"
        # 顺序；此处仍按名字显式定位，避免隐式假设）。
        ex_columns = schema[:-1]
        mark_pos = [i for i, c in enumerate(ex_columns) if is_mark_col(c)]
        cal_pos = [i for i, c in enumerate(ex_columns) if is_calendar_col(c)]
        self.mark_pos = np.asarray(mark_pos, dtype=np.int64)
        self.cal_pos = np.asarray(cal_pos, dtype=np.int64)
        if calendar_mode == "time_marks":
            drop = set(mark_pos)
        elif calendar_mode == "none":
            drop = set(cal_pos)
        else:  # columns
            drop = set()
        self.keep_pos = np.asarray(
            [i for i in range(len(schema)) if i not in drop],
            dtype=np.int64,
        )
        # ex_mask 覆盖 x 中的外生 token（不含 TARGET），顺序与 x 一致：
        # 在全 schema 掩码上按 keep 的外生位置重排即可。
        self.ex_mask_order = np.asarray(
            [i for i in self.keep_pos if i < len(schema) - 1],
            dtype=np.int64,
        )
        # 目标日星期 one-hot（预测头日历向量来源）：交易日必为周一~周五。
        self.head_onehot = []
        for r in records:
            dow = pd.DatetimeIndex(r["dates"]).dayofweek
            if (dow > 4).any():
                raise ValueError(
                    f"{r['symbol']}: 交易日历包含周末日期，无法构造星期 one-hot"
                )
            onehot = np.zeros((len(dow), 5), dtype=np.float32)
            onehot[np.arange(len(dow)), dow] = 1.0
            self.head_onehot.append(onehot)

        counts = []
        for record_id, r in enumerate(records):
            frame = pd.read_csv(
                Path(r["root"]) / "samples.csv",
                usecols=[
                    "start_pos", "end_pos", "target_pos",
                    "split", "label_valid", "y_scaled", "y_raw",
                ],
            )
            # 只装载可评分样本（label_valid）：输入窗口有效但目标日无
            # 标签的行是覆盖登记，不能作为监督样本。
            frame = frame.loc[
                (frame.split == split) & frame.label_valid
            ].copy()
            if frame.empty:
                if allow_missing:
                    continue
                raise ValueError(f"{r['symbol']}: {split} 没有样本")

            positions = frame[
                ["start_pos", "end_pos", "target_pos"]
            ].to_numpy(dtype=np.int64)

            start, end, target = positions.T
            if not (
                (start >= 0).all()
                and (end - start + 1 == self.seq_len).all()
                and (target == end + 1).all()
                and (target < len(r["dates"])).all()
            ):
                raise ValueError(f"{r['symbol']}: 窗口位置非法")

            target_dates = r["dates"][target].astype("datetime64[D]")
            train_end = np.datetime64(r["prep"]["train_end"], "D")
            val_end = (
                np.datetime64(r["prep"]["val_end"], "D")
                if r["prep"]["val_end"] is not None else None
            )

            if split == "train":
                valid = target_dates <= train_end
                # rolling 导出（--train-start）时训练目标日不得早于
                # 登记的训练起点；make_samples 已限制，这里是装载期
                # 的纵深防御。
                start_value = (
                    r["prep"].get("sample_selection") or {}
                ).get("start")
                if start_value:
                    valid &= target_dates >= np.datetime64(start_value, "D")
            elif split == "val":
                valid = (
                    (target_dates > train_end) & (target_dates <= val_end)
                )
            else:
                if layout == "train_test":
                    valid = target_dates > train_end
                else:
                    valid = target_dates > val_end
            if not valid.all():
                raise ValueError(f"{r['symbol']}: {split} 日期越界")

            stats = r["prep"]["statistics"][TARGET]
            expected = (
                frame.y_raw.to_numpy() - stats["mean"]
            ) / stats["scale"]
            ys = frame.y_scaled.to_numpy(dtype=np.float32)
            if not np.allclose(ys, expected, rtol=1e-5, atol=1e-5):
                raise ValueError(f"{r['symbol']}: 标签标准化不一致")
            # float64 y_raw 随样本透传：y_true 评估直接用原始标签，
            # 不经 float32 标签反标准化重建（两阶段协议的 y_true 一致性
            # 检查因此可以精确到 0）。
            raws = frame.y_raw.to_numpy(dtype=np.float64)
            if not np.isfinite(raws).all():
                raise ValueError(
                    f"{r['symbol']}: {split} 可评分样本存在非有限 y_raw"
                )

            self.parts.append((positions, ys, raws))
            self.record_ids.append(record_id)
            counts.append(len(frame))

        self.counts = np.asarray(counts, dtype=np.int64)
        self.ends = (
            np.cumsum(self.counts)
            if len(self.counts) else np.zeros(0, dtype=np.int64)
        )

        # 每个 epoch 所有股票总权重相同（权重只覆盖实际进入本 split
        # 的股票；被跳过的股票权重为 0，也不会出现在任何 batch 里）。
        self.stock_weights = np.zeros(len(records), dtype=np.float32)
        if len(self.record_ids) and len(self):
            total = len(self)
            for local_id, record_id in enumerate(self.record_ids):
                self.stock_weights[record_id] = (
                    total
                    / (len(self.record_ids) * self.counts[local_id])
                )

        lookup = {c: i for i, c in enumerate(schema)}
        self.maps = []
        for r in records:
            pairs = [
                (lookup[c], i)
                for i, c in enumerate(r["columns"]) if c in lookup
            ]
            self.maps.append((
                np.array([p[0] for p in pairs], dtype=np.int64),
                np.array([p[1] for p in pairs], dtype=np.int64),
            ))
        self.missing = [
            i for i, c in enumerate(schema) if c.endswith("__missing")
        ]
        # 外生值列 -> 缺失标记列映射：__getitem__ 据此产出 ex_mask。
        # 是否把 ex_mask 传入模型由调用方（--ex-mask）决定，数据集恒产出。
        self.ex_mask_layout = exogenous_mask_layout(schema)

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["cache"] = OrderedDict()
        return state

    def matrix(self, sid):
        if sid not in self.cache:
            path = Path(self.records[sid]["root"]) / "feature_matrix.npy"
            self.cache[sid] = np.load(path, mmap_mode="r")
            # 防止几千个股票矩阵长期同时占用文件描述符。
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        self.cache.move_to_end(sid)
        return self.cache[sid]

    def __getitem__(self, index):
        local_id = bisect.bisect_right(self.ends, index)
        offset = 0 if local_id == 0 else int(self.ends[local_id - 1])
        local = index - offset

        positions, labels, raws = self.parts[local_id]
        sid = self.record_ids[local_id]
        start, end, target = positions[local]
        raw = self.matrix(sid)[start:end + 1]

        x_full = np.zeros(
            (self.seq_len, len(self.schema)), dtype=np.float32
        )
        x_full[:, self.missing] = 1.0
        dst, src = self.maps[sid]
        x_full[:, dst] = raw[:, src]

        # 外生 token 有效性掩码（见 compute_ex_mask）。
        # 只标记"窗口内整段无观测"的外生值列；缺失标记 / 年龄列不屏蔽；
        # 日历列无 __missing 伴随列，永不屏蔽。
        ex_mask_full = compute_ex_mask(x_full, self.ex_mask_layout)

        # 日历通道拆分（calendar_mode）：日历列集合全股票一致
        # （load_catalog 已校验），不存在日历列补零对齐。
        mode = self.calendar_mode
        empty_mark = np.zeros((self.seq_len, 0), dtype=np.float32)
        if mode == "columns":
            x, x_mark = x_full, empty_mark
        elif mode == "time_marks":
            x = x_full[:, self.keep_pos]
            x_mark = (
                x_full[:, self.mark_pos]
                if len(self.mark_pos) else empty_mark
            )
        else:  # none
            x = x_full[:, self.keep_pos]
            x_mark = empty_mark

        # ex_mask 重排到 x 的外生 token 顺序（columns 模式下为恒等）。
        ex_mask = ex_mask_full[self.ex_mask_order]

        y = np.array([[labels[local]]], dtype=np.float32)
        # 返回 8 元组：x, y, sid, target, y_raw, ex_mask, x_mark, head_cal。
        # y_raw 为 float64 原始标签（评估侧直接使用）；ex_mask 是否启用
        # 由训练/推理脚本的 --ex-mask 决定，数据集恒产出；x_mark 为
        # 时间标记通道（columns/none 模式下为 0 列）；head_cal 为目标日
        # 星期 one-hot（预测头日历拼接的输入，恒产出）。
        return (
            x, y, sid, int(target), np.float64(raws[local]),
            ex_mask, x_mark, self.head_onehot[sid][target],
        )

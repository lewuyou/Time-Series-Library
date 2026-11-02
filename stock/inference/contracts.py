"""预测记录契约：字段名、状态枚举与 DataFrame 规范化。

历史实现把"是否可打分"（no_infer_output / bad_shape / non_finite）、
"是否过期"（stale）和"是否发信号"（signal）混在同一个 note 列里。
本模块把三者拆开：

- status：数据/推理层面是否可用（PredictionStatus）；
- signal：选股规则输出（布尔列，由 selection 层写）；
- stale：仅为 status 的一个取值，不再与 signal 混用。

所有落库/导出/回测的预测记录必须经过 normalize_prediction_frame，
缺字段直接报错，避免下游各自解释。
"""

from enum import Enum

import pandas as pd

from stock.data.schema import TARGET


class PredictionStatus(str, Enum):
    """预测记录的数据可用性状态（不含选股结论）。"""

    OK = "ok"
    STALE = "stale"
    NO_INFER_OUTPUT = "no_infer_output"
    BAD_SHAPE = "bad_shape"
    NON_FINITE = "non_finite"
    ERROR = "error"


#: 统一预测记录字段（selection 层会再补 rank/signal 等列）。
REQUIRED_FIELDS = (
    "symbol",
    "as_of_date",
    "target_date",
    "model_id",
    "dataset_id",
    "y_pred",
    "status",
)

#: 状态列允许值（落库前校验；出现未知状态即报错，不静默转 ok）。
STATUS_VALUES = {s.value for s in PredictionStatus}


def normalize_prediction_frame(frame):
    """补齐/校验预测记录；y_pred 单位为小数收益率。

    - target_date/as_of_date 统一为 YYYY-MM-DD 字符串；
    - status 必须是 PredictionStatus 的取值；
    - pred_return_pct 若缺失则按 y_pred*100 补齐（便于兼容旧 CSV）；
    - input_date 保留为 as_of_date 的别名（旧调用方仍可读取）。
    """
    if frame is None or frame.empty:
        raise ValueError("预测记录为空，不能发布")

    out = frame.copy()
    for column in ("symbol", "target_date"):
        if column not in out:
            raise ValueError(f"预测记录缺少必需字段：{column}")
        out[column] = out[column].astype(str)

    if "as_of_date" not in out:
        if "input_date" not in out:
            raise ValueError("预测记录缺少必需字段：as_of_date/input_date")
        out["as_of_date"] = out["input_date"]
    out["as_of_date"] = out["as_of_date"].astype(str)
    if "input_date" not in out:
        out["input_date"] = out["as_of_date"]

    for column in ("model_id", "dataset_id"):
        if column not in out:
            raise ValueError(f"预测记录缺少必需字段：{column}")

    if "status" not in out:
        raise ValueError("预测记录缺少必需字段：status")
    bad = sorted(set(out["status"].astype(str)) - STATUS_VALUES)
    if bad:
        raise ValueError(f"未知预测状态：{bad}；允许值={sorted(STATUS_VALUES)}")

    if "y_pred" not in out:
        if "pred_return_pct" not in out:
            raise ValueError("预测记录缺少 y_pred/pred_return_pct")
        out["y_pred"] = out["pred_return_pct"].astype(float) / 100.0
    out["y_pred"] = out["y_pred"].astype(float)
    if "pred_return_pct" not in out:
        out["pred_return_pct"] = out["y_pred"] * 100.0

    for column in REQUIRED_FIELDS:
        if column not in out:
            raise ValueError(f"预测记录缺少必需字段：{column}")
    return out


def check_target_columns(schema):
    """特征 schema 必须包含目标列（供回测/评估读取）。"""
    if TARGET not in schema:
        raise ValueError(f"特征 schema 缺少目标列 {TARGET}")

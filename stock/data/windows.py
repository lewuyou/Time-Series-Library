"""样本窗口切分（samples.csv 的生成规则）。"""

import numpy as np
import pandas as pd

from stock.data.common import SPLIT_LAYOUTS
from stock.data.schema import TARGET
from stock.data.targets import label_valid as is_label_valid

def make_samples(features, core_valid, config, train_end, val_end,
                 start=None, tail=None, split_layout="train_val_test"):
    """返回 (样本表, 候选目标日数, 剔除统计)。

    split_layout 见模块常量 SPLIT_LAYOUTS：
      train_val_test：target <= train_end 为 train，<= val_end 为 val，
        其余为 test；
      train_test：target <= train_end（此时 train_end 即重拟合截止日
        fit_end，包含原验证区间）为 train，其余为 test，不产生 val 样本。

    样本口径区分"可预测 / 可评分"：
      输入窗口有效的行全部保留（该预测在目标日早 06:00 本可合法发出），
      其中目标日无有效行情或标签的行标记 label_valid=False、y_raw 为空，
      保留在 samples.csv 中作为覆盖登记；监督训练/评估装载时只取
      label_valid=True 的行（见 fit_preprocessing / PanelWindows）。
      评分时排除无标签样本不可避免，但不得把它伪装成"那天没有生成预测"。

    剔除统计区分三类：
      insufficient_history：历史不足，尚不能开窗（结构性预热）；
      invalid_input_window：输入窗口含无效核心行（多为停牌），无法在
        预测时点生成输入；
      invalid_target_label：输入窗口有效、预测本可发出，但目标日无有效
        行情或标签（= samples 中 label_valid=False 的行数，事后筛选
        偏差应被报告而非静默）。
    """
    if split_layout not in SPLIT_LAYOUTS:
        raise ValueError(f"不支持的 split_layout：{split_layout}")
    if split_layout == "train_test" and val_end is not None:
        raise ValueError("train_test 布局没有验证集，val_end 必须为 None")
    length = config["seq_len"]
    index = features.index
    y = features[TARGET].to_numpy(dtype=float)
    valid = core_valid.to_numpy(dtype=bool)

    # 目标日期限制，不先截断底层行情，保留滚动计算预热。
    candidate_targets = index[1:]
    if start is not None:
        candidate_targets = candidate_targets[candidate_targets >= start]
    if tail:
        candidate_targets = candidate_targets[-tail:]
    allowed = set(candidate_targets)

    records = []
    # 前缀和：O(1) 检查一个窗口是否存在无效核心行。
    bad_prefix = np.r_[0, np.cumsum(~valid)]
    exclusions = {
        "insufficient_history": 0,
        "invalid_input_window": 0,
        "invalid_target_label": 0,
    }
    positions = index.get_indexer(candidate_targets)
    exclusions["insufficient_history"] = int((positions < length).sum())

    for end in range(length - 1, len(index) - 1):
        first, target_pos = end - length + 1, end + 1
        target_date = index[target_pos]
        if target_date not in allowed:
            continue
        if bad_prefix[end + 1] - bad_prefix[first] != 0:
            exclusions["invalid_input_window"] += 1
            continue

        label_ok = is_label_valid(valid[target_pos], y[target_pos])
        if not label_ok:
            exclusions["invalid_target_label"] += 1

        if split_layout == "train_test":
            split = "train" if target_date <= train_end else "test"
        else:
            split = (
                "train" if target_date <= train_end
                else "val" if target_date <= val_end
                else "test"
            )
        records.append({
            "start_pos": first,
            "end_pos": end,
            "target_pos": target_pos,
            "input_date": index[end],
            "target_date": target_date,
            "split": split,
            "label_valid": label_ok,
            "y_raw": float(y[target_pos]) if label_ok else np.nan,
        })

    columns = [
        "start_pos", "end_pos", "target_pos", "input_date",
        "target_date", "split", "label_valid", "y_raw",
    ]
    result = pd.DataFrame(records, columns=columns)
    return result, len(candidate_targets), exclusions



"""数据层共享常量与基础工具（自根目录 dataset.py 原样迁出）。"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from stock.utils import write_json

VERSION = "stock_timexer_v8"
# 可读取的历史版本：旧版缺少日历指纹、sample_selection、split_layout、
# label_valid 等新增字段，infer 按兼容逻辑回退旧行为并提示。
# v7 起 samples.csv 区分"可预测"（input_eligible，行保留）与"可评分"
# （label_valid=True），且单股仅强制训练样本下限，验证/测试覆盖率只登记。
# v8 起特征加入目标日星期 one-hot（target_dow_mon..fri，保持 0/1
# 不标准化）；旧版 preprocessor infer 仍兼容（one-hot 列被计算但不使用）。
COMPATIBLE_VERSIONS = (
    "stock_timexer_v3", "stock_timexer_v4", "stock_timexer_v5",
    "stock_timexer_v6", "stock_timexer_v7", "stock_timexer_v8",
)

# 样本划分布局：
#   train_val_test：train <= train_end < val <= val_end < test（原协议）；
#   train_test：   train <= train_end（即 fit_end，含验证区间）< test。
# train_test 用于 walk-forward 两阶段协议的重拟合导出：验证集在测试开始前
# 已成为历史，可以进入最终训练；测试集不参与任何训练期决策。
SPLIT_LAYOUTS = ("train_val_test", "train_test")
# TARGET / TARGET_DOW_NAMES 改自 stock.data.schema 导入（单一事实源），
# 值与历史版本一致。

DEFAULT_DB = "../AutoTS/download/autots.duckdb"
DEFAULT_CALENDAR_START = "2016-01-01"

# volume 单位为股（采集端已从手换算），特征侧用 log1p 压缩量级。
PRICE_FIELDS = ["open", "close", "high", "low", "volume", "turnover"]
BAIDU_FIELDS = ["search_all", "feed"]
VAL_FIELDS = ["pe", "pb", "ps", "pcf"]
RZRQ_FIELDS = ["fin_tval_ratio", "loan_tval_ratio"]

# 可覆盖来源策略，也可精确覆盖到 "macro:品种:字段"。
DEFAULT_POLICIES = {
    "baidu": {"lag_days": 2, "release_hour": 0, "max_age_days": 7},
    "guzhi": {"lag_days": 2, "release_hour": 0, "max_age_days": 10},
    "rzrq": {"lag_days": 2, "release_hour": 0, "max_age_days": 5},
    "macro": {"lag_days": 2, "release_hour": 0, "max_age_days": 10},
}




def norm_symbol(value):
    text = re.sub(r"^(sh|sz)", "", str(value).strip().lower())
    if not re.fullmatch(r"\d{1,6}", text):
        raise ValueError(f"非法股票代码：{value!r}")
    return text.zfill(6)




def sql_symbol(column):
    return (
        "lpad(regexp_replace("
        f"lower(trim(cast({column} as varchar))), "
        "'^(sh|sz)', ''), 6, '0')"
    )


json_write = write_json




def parse_dates(df):
    df = df.copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    if df["date"].isna().any():
        raise ValueError("原始数据存在空日期")
    return df




def assert_unique(df, keys, name):
    if df.empty:
        return
    bad = df.duplicated(keys, keep=False)
    if bad.any():
        example = df.loc[bad, keys].head(8).to_dict("records")
        raise ValueError(f"{name} 存在重复业务键：{example}")




def as_numeric(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)




def load_policies(path):
    policies = json.loads(json.dumps(DEFAULT_POLICIES))
    if path:
        updates = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, update in updates.items():
            if key not in DEFAULT_POLICIES and not key.startswith("macro:"):
                raise ValueError(f"未知策略键：{key}")
            base = policies.get(key, policies["macro"]).copy()
            base.update(update)
            policies[key] = base

    for key, policy in policies.items():
        if set(policy) != {"lag_days", "release_hour", "max_age_days"}:
            raise ValueError(f"{key} 策略字段不正确")
        if policy["lag_days"] < 0 or policy["max_age_days"] < 0:
            raise ValueError(f"{key} 的滞后/年龄不能为负")
        if not 0 <= policy["release_hour"] <= 23:
            raise ValueError(f"{key} release_hour 非法")
        # 防泄露护栏：本库全部来源都是日频观测（行情/估值/两融收盘后才有，
        # 百度指数更晚），观测日 d 的值最早在 d 收盘（约 15:00）后才存在。
        # lag_days=0 且 release_hour<15 等于声称"当日盘中值已知"，会把
        # 尚未发生的值当作可用（例如读取到 next_session 的日线）。
        if policy["lag_days"] == 0 and policy["release_hour"] < 15:
            raise ValueError(
                f"{key}：lag_days=0 且 release_hour={policy['release_hour']} "
                "不合理——日频观测值最早在观测日收盘后才存在，该配置会把"
                "未发生的值当作可用（未来数据泄露）。请增大 lag_days 或把"
                "release_hour 设为不早于实际发布时刻的小时（>=15）。"
            )
    return policies



"""特征列布局的单一事实源（生成侧 + 解析侧共用）。

原先 dataset.py（生成侧：`__missing` / `__age_days` / 目标日日历列）
与 train_global_timexer.py（解析侧：base_name / align_features /
exogenous_mask_layout / compute_ex_mask）各写一份布局规则，改特征方案
需要同步两处，漏改一处即静默错位。现统一收敛到本模块：

- 生成侧（dataset.py 生成特征矩阵）：引用命名常量与
  missing_col / age_col / is_* 判定；
- 解析侧（stock.data.panel / stock.adapters / stock.inference）：
  引用 base_name / align_features / exogenous_mask_layout /
  compute_ex_mask / DROP_BASES。

改特征布局（增删伴随列、目标日日历列、外生来源前缀、价格基础列
剔除范围）只改本文件。本模块保持纯 Python（仅依赖 numpy 的
张量对齐函数除外），供 dataset.py 安全引用。
"""

import hashlib
import json

import numpy as np

TARGET = "stock_return"

#: 特征列布局版本。列名/顺序/分组语义变化时递增；写入导出目录与
#: checkpoint，推理契约不匹配即报错。
#: v3：删除 target_dow_* one-hot（星期信息改由 x_mark 时间标记通道
#: 与预测头 one-hot 承载，不做双份编码）；新增 3 个官方归一化时间
#: 特征列（target_mark_dow/dom/doy）+ 5 个交易日历旗标列，全部按
#: 目标日（下一交易日）取值。
FEATURE_SCHEMA_VERSION = "stock_features_v3"


# 伴随列后缀。
MISSING_SUFFIX = "__missing"
AGE_SUFFIX = "__age_days"

# 目标日时间标记列（官方 TimeXer 日频三特征，归一化写法与
# utils/timefeatures.py 的 DayOfWeek/DayOfMonth/DayOfYear 一致；
# 行 t 存"预测目标交易日 = t 的下一交易日"的取值）。默认经
# calendar_mode=time_marks 移入 x_mark 通道（官方通道），也可按
# calendar_mode=columns 留在输入列。
TARGET_MARK_DOW = "target_mark_dow"
TARGET_MARK_DOM = "target_mark_dom"
TARGET_MARK_DOY = "target_mark_doy"

#: 时间标记列（规范顺序 = 官方 time_features_from_frequency_str("d")）。
TARGET_MARK_NAMES = (TARGET_MARK_DOW, TARGET_MARK_DOM, TARGET_MARK_DOY)

# 目标日日历旗标列（语义见 stock.data.features.target_calendar_features：
# 行 t 存"预测目标交易日 = t 的下一交易日"的属性，纯交易日历推导）。
TARGET_GAP_DAYS_LOG = "target_gap_days_log"
TARGET_IS_GAP_GE4 = "target_is_gap_ge4"
TARGET_IS_PRE_LONG_HOLIDAY = "target_is_pre_long_holiday"
TARGET_IS_MONTH_START_3 = "target_is_month_start_3"
TARGET_IS_MONTH_END_3 = "target_is_month_end_3"

#: 日历旗标列（x_mark 装不下、有独立信息量，恒为普通外生列）。
TARGET_CALENDAR_EXTRA_NAMES = (
    TARGET_GAP_DAYS_LOG,
    TARGET_IS_GAP_GE4,
    TARGET_IS_PRE_LONG_HOLIDAY,
    TARGET_IS_MONTH_START_3,
    TARGET_IS_MONTH_END_3,
)

# 外生（辅助）来源前缀：dataset.py 按此前缀对齐可用时间并生成伴随列。
EXOGENEOUS_SOURCES = ("baidu", "val", "rzrq", "macro")
EXOGENEOUS_PREFIXES = tuple(f"{s}_" for s in EXOGENEOUS_SOURCES)

# 价格原值列不进入外生 schema（已转成收益率/比率等衍生特征）。
DROP_BASES = {"px_open", "px_close", "px_high", "px_low"}


# ---------------------------------------------------------------- 命名规则（生成侧）

def missing_col(base):
    """基础列的缺失标记伴随列名。"""
    return f"{base}{MISSING_SUFFIX}"


def age_col(base):
    """辅助来源基础列的数据年龄伴随列名。"""
    return f"{base}{AGE_SUFFIX}"


def target_calendar_columns():
    """全部目标日日历列（时间标记列 + 旗标列，规范顺序）。"""
    return list(TARGET_MARK_NAMES) + list(TARGET_CALENDAR_EXTRA_NAMES)


def parse_calendar_features(spec):
    """日历列集合配置 -> 启用列名列表（规范顺序）。

    spec: "all"（默认，3 个时间标记列 + 5 个旗标列）/ "none"（全部
    关闭）/ 逗号分隔列名列表（子集，用于 A/B）。
    """
    if spec is None:
        spec = "all"
    text = str(spec).strip()
    known = target_calendar_columns()
    if text == "all":
        return list(known)
    if text == "none":
        return []
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(
            f"未知日历特征列：{unknown}；可选：{known}"
        )
    return [c for c in known if c in set(names)]


def is_missing_col(name):
    return name.endswith(MISSING_SUFFIX)


def is_age_col(name):
    return name.endswith(AGE_SUFFIX)


def is_mark_col(name):
    """目标日时间标记列（官方日频三特征；time_marks 模式移入 x_mark）。"""
    return name in TARGET_MARK_NAMES


def is_calendar_col(name):
    """目标日日历列（时间标记列与旗标列共用 target_ 前缀）。

    这些列永远已知（无 __missing 伴随列）、不参与标准化与屏蔽；
    装载/训练侧按此判定拆分日历通道（见 stock.data.panel）。
    """
    return name.startswith("target_")


def is_companion_col(name):
    """缺失标记或数据年龄伴随列。"""
    return is_missing_col(name) or is_age_col(name)


def is_exogenous_col(name):
    """辅助（外生）来源的基础列。"""
    return name.startswith(EXOGENEOUS_PREFIXES)


def base_name(column):
    """列名 → 基础列名（去掉伴随后缀；'a__b__missing' → 'a'）。"""
    return column.split("__", 1)[0]


# ---------------------------------------------------------------- 张量对齐（解析侧）

def align_features(x, source_columns, schema):
    """重排列；被某股票过滤掉的特征作为缺失，而不是挪用其他列。"""
    result = np.zeros((*x.shape[:-1], len(schema)), dtype=np.float32)
    missing_positions = [
        i for i, c in enumerate(schema) if c.endswith(MISSING_SUFFIX)
    ]
    result[..., missing_positions] = 1.0

    source = {c: i for i, c in enumerate(source_columns)}
    dst = [i for i, c in enumerate(schema) if c in source]
    src = [source[schema[i]] for i in dst]
    result[..., dst] = x[..., src]
    return result


def exogenous_mask_layout(schema):
    """外生值列 -> 缺失标记列的位置映射（不含 target）。

    schema[:-1] 与 TimeXer 的外生 token 顺序（x_enc[:, :, :-1]）一致。
    缺失标记列与数据年龄列本身是信息来源，不参与屏蔽判定；
    返回 (n, 2) int64 数组：第 0 列为值列 token 位置，第 1 列为
    对应 `值列__missing` 标记列位置。
    """
    pos = {c: i for i, c in enumerate(schema)}
    pairs = []
    for j, c in enumerate(schema[:-1]):
        if is_missing_col(c) or is_age_col(c):
            continue
        flag = pos.get(c + MISSING_SUFFIX)
        if flag is not None:
            pairs.append((j, flag))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def compute_ex_mask(x, layout):
    """按窗口计算外生 token 有效性掩码。

    x: 已对齐到 schema 的特征矩阵，[..., seq_len, n_schema]。
    layout: exogenous_mask_layout(schema) 的输出。
    返回 [..., n_schema-1] bool：True = 该外生值列在本窗口内
    整段无有效观测（标记列全为 1；股票缺列时 align_features
    已把标记列置 1）。判定只用标记列，不看值列数值是否为 0。
    缺失标记 / 数据年龄列本身永不屏蔽。
    """
    ex_mask = np.zeros(x.shape[:-2] + (x.shape[-1] - 1,), dtype=bool)
    if len(layout):
        masked = (x[..., layout[:, 1]] == 1.0).all(axis=-2)
        ex_mask[..., layout[:, 0]] = masked
    return ex_mask


def feature_group(column):
    """特征列的分组标签（仅用于审计与可视化，不参与计算）。"""
    if column == TARGET:
        return "target"
    if is_calendar_col(column):
        return "calendar"
    if is_missing_col(column):
        return "missing_flag"
    if is_age_col(column):
        return "availability"
    if is_exogenous_col(column):
        return "exogenous:" + column.split("_", 1)[0]
    return "price"


def feature_schema(schema):
    """生成特征 schema 描述（name/order/dtype/group）。

    dtype 记录的是实际落盘矩阵类型（统一 float32）；语义差异由
    group 承担，避免把 0/1 标记硬说成 bool 而与 mmap 不一致。
    """
    columns = []
    for order, name in enumerate(schema):
        columns.append({
            "name": name,
            "order": order,
            "dtype": "float32",
            "group": feature_group(name),
        })
    return columns


def feature_schema_hash(columns, version=None):
    """稳定 hash：只依赖版本与列描述，不依赖 dict 插入顺序。

    version 缺省为当前版本常量；校验旧产物时传入其存储版本重算，
    使版本升级（新增列）不会把旧导出目录/旧 checkpoint 判为不匹配。
    """
    payload = {
        "version": FEATURE_SCHEMA_VERSION if version is None else version,
        "columns": columns,
    }
    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def feature_schema_doc(schema):
    """可直接写入 JSON 的 schema 文档（含稳定 hash）。"""
    columns = feature_schema(schema)
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "hash": feature_schema_hash(columns),
        "columns": columns,
    }


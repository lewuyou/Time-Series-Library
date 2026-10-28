#!/usr/bin/env bash
# ============================================================================
# 4. 指定宏观品种训练（拉取数据库数据 -> 直接训练）
#
#   单股（示例 601398）+ 指定宏观品种（示例 SSE:000001 SSE:510300
#   SPCFD:SPX，close + volume）的两步流程：
#     a. dataset.py 拉取行情与指定宏观品种并导出；
#     b. train_timexer.py 直接读取导出目录训练。
#   建议先用少数定义明确的品种，不必一开始加入全部宏观数据。
#   全空、训练覆盖不足或训练期为常数的特征组会被删除，记录在
#   preprocessing.json -> dropped_groups；省略 --macro-symbols 时读取
#   查询区间内全部宏观品种并把实际品种列表冻结进 preprocessing.json。
#
#   训练前人工检查数据：导出目录中的 train_preview.csv（逐交易日的标准化
#   特征矩阵 + 当日原始收益率 y_raw + train/val/test 划分）。
#
# 用法：
#   bash 4_train_macro_symbols.sh
#   DATA_DIR=dataset/timexer_macro_002/601398 \
#       TRAIN_OUT=checkpoints/stock_timexer_macro_002 \
#       bash 4_train_macro_symbols.sh
#   # 超参透传给 train_timexer.py：
#   bash 4_train_macro_symbols.sh --d-model 32 --epochs 30
#
# 宏观数据可用时间：macro 来源默认 lag_days=2（保守）。ETF/指数等日频
# 数据盘后即可获取，若确认当天盘后已入库且早于决策时点 08:30，可用
# POLICY=source_policy.json 把对应 "macro:品种:字段" 调成
# lag_days=0 + release_hour=18（lag_days=0 时 release_hour 必须 >= 15）。
# 百度数据延迟：见 1_train_single_stock.sh 头部说明（默认 lag_days=2）。
# 输出目录已存在时 dataset.py 拒绝覆盖：重复运行请换新的 DATA_DIR，
# 或删除旧目录；只重训不重拉数据时直接用 8_train_timexer.sh。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

DB="${DB:-../AutoTS/download/autots.duckdb}"
DATA_DIR="${DATA_DIR:-dataset/timexer_macro_001/601398}"
TRAIN_OUT="${TRAIN_OUT:-checkpoints/stock_timexer_macro_001}"
POLICY="${POLICY:-}"

if [ -d "$DATA_DIR" ]; then
    echo "[ERROR] 数据目录已存在：$DATA_DIR（dataset.py 拒绝覆盖）。" >&2
    echo "        请换新的 DATA_DIR；只重训不重拉数据可用 bash 8_train_timexer.sh" >&2
    exit 1
fi
DATA_ROOT="$(dirname "$DATA_DIR")"

policy_args=()
[ -n "$POLICY" ] && policy_args=(--source-policy "$POLICY")

# ---- 第 1 步：dataset.py 拉取股票 + 指定宏观品种数据并导出 ----
"$PYTHON" dataset.py \
    --db "$DB" \
    --symbol 601398 \
    --macro-symbols SSE:000001 SSE:510300 SPCFD:SPX \
    --macro-fields close volume \
    --train-end 2022-12-31 \
    --val-end 2023-12-31 \
    --as-of 2024-12-31 \
    "${policy_args[@]}" \
    --out-dir "$DATA_ROOT"

echo "训练前数据预览（逐日标准化特征 + y_raw + split）：$DATA_DIR/train_preview.csv"

# ---- 第 2 步：train_timexer.py 直接训练 ----
"$PYTHON" train_timexer.py --data-dir "$DATA_DIR" --out-dir "$TRAIN_OUT" "$@"

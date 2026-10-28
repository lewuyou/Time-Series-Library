#!/usr/bin/env bash
# ============================================================================
# 3. 行业内市值前 10 只训练（示例：电力）（拉取数据库数据 -> 直接训练）
#
#   与 2_train_industry_all.sh 相同的两步流程，仅在选股上增加 --top 10：
#   按股票列表 total_mv 降序取行业前 10 只，每只独立导出、独立训练。
#
#   训练前人工检查数据：每只股票目录下的 train_preview.csv（逐交易日的
#   标准化特征矩阵 + 当日原始收益率 y_raw + train/val/test 划分）。
#
# 用法：
#   bash 3_train_industry_top10.sh
#   INDUSTRY=银行 TOP=20 \
#       DATA_ROOT=dataset/timexer_bank_top20 \
#       TRAIN_OUT=checkpoints/timexer_bank_top20 bash 3_train_industry_top10.sh
#   # 超参透传给每只股票的 train_timexer.py：
#   bash 3_train_industry_top10.sh --d-model 32 --epochs 30
#
# 断点续跑：同 2_train_industry_all.sh（dataset.py 只补缺失股票，
# 训练循环按 metrics.json 跳过已完成者）。
# 百度数据延迟：见 1_train_single_stock.sh 头部说明。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

DB="${DB:-../AutoTS/download/autots.duckdb}"
DATA_ROOT="${DATA_ROOT:-dataset/timexer_power_top10}"
TRAIN_OUT="${TRAIN_OUT:-checkpoints/timexer_power_top10}"
INDUSTRY="${INDUSTRY:-电力}"
TOP="${TOP:-10}"
POLICY="${POLICY:-}"

policy_args=()
[ -n "$POLICY" ] && policy_args=(--source-policy "$POLICY")

# ---- 第 1 步：dataset.py 拉取行业市值前 N 只股票数据并导出 ----
"$PYTHON" dataset.py \
    --db "$DB" \
    --industry "$INDUSTRY" \
    --top "$TOP" \
    --train-end 2022-12-31 \
    --val-end 2023-12-31 \
    --as-of 2024-12-31 \
    --continue-on-error \
    "${policy_args[@]}" \
    --out-dir "$DATA_ROOT"

# ---- 第 2 步：逐只训练（训练前数据预览也在每只股票目录下）----
trained=0 skipped=0 failed=0
for d in "$DATA_ROOT"/*/; do
    [ -f "${d}preprocessing.json" ] || continue
    symbol="$(basename "$d")"
    if [ -f "$TRAIN_OUT/$symbol/metrics.json" ]; then
        echo "[SKIP] $symbol：$TRAIN_OUT/$symbol 已完成（重训请删除该目录）"
        skipped=$((skipped + 1))
        continue
    fi
    echo "==> 训练 $symbol（训练前数据预览：$d/train_preview.csv）"
    if "$PYTHON" train_timexer.py \
        --data-dir "$d" --out-dir "$TRAIN_OUT/$symbol" "$@"; then
        trained=$((trained + 1))
    else
        echo "[ERROR] $symbol 训练失败，继续其余股票" >&2
        failed=$((failed + 1))
    fi
done

echo "行业 $INDUSTRY 前 $TOP 训练完成：成功 $trained | 跳过 $skipped | 失败 $failed"
if [ "$((trained + skipped + failed))" -eq 0 ]; then
    echo "[ERROR] $DATA_ROOT 下没有可训练的股票导出目录，" >&2
    echo "        请检查 $DATA_ROOT/run_summary_*.json 中被跳过的股票。" >&2
    exit 1
fi
[ "$failed" -eq 0 ] || exit 1

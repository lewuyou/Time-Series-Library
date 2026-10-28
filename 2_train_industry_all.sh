#!/usr/bin/env bash
# ============================================================================
# 2. 行业内全部股票训练（示例：电力）（拉取数据库数据 -> 直接训练）
#
#   一个脚本完成两步（替代原 2_industry_all.sh + 逐只跑 8_train_timexer.sh）：
#     a. dataset.py 处理 stock_list 中 industry_l2 = 电力 的全部股票
#        （每只股票独立导出到 $DATA_ROOT/<代码>/，不拼接成一条时间序列；
#        --continue-on-error 跳过样本不足等失败股票，汇总见
#        $DATA_ROOT/run_summary_*.json）；
#     b. 逐只调用 train_timexer.py 直接训练，checkpoint / metrics /
#        test_predictions 输出到 $TRAIN_OUT/<代码>/。
#
#   训练前人工检查数据：每只股票目录下的 train_preview.csv（逐交易日的
#   标准化特征矩阵 + 当日原始收益率 y_raw + train/val/test 划分）。
#
# 用法：
#   bash 2_train_industry_all.sh
#   DATA_ROOT=dataset/timexer_power_002 TRAIN_OUT=checkpoints/timexer_power_002 \
#       bash 2_train_industry_all.sh
#   # 超参透传给每只股票的 train_timexer.py：
#   bash 2_train_industry_all.sh --d-model 32 --epochs 30
#
# 断点续跑：
#   - dataset.py 对已存在的股票目录报错跳过、只补缺失的股票，
#     所以中断后用相同 DATA_ROOT 重跑即可继续导出；
#   - 训练循环按 metrics.json 是否存在跳过已完成的股票，
#     只重训未完成/失败的部分（强行重训请删除对应 $TRAIN_OUT/<代码>）。
#
# 百度数据延迟：见 1_train_single_stock.sh 头部说明（默认 lag_days=2，
# 与百度 API 滞后 1 天 + 09:00 采集晚于 08:30 决策时点的现实一致）。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

DB="${DB:-../AutoTS/download/autots.duckdb}"
DATA_ROOT="${DATA_ROOT:-dataset/timexer_power_001}"
TRAIN_OUT="${TRAIN_OUT:-checkpoints/timexer_power_001}"
INDUSTRY="${INDUSTRY:-电力}"
POLICY="${POLICY:-}"

policy_args=()
[ -n "$POLICY" ] && policy_args=(--source-policy "$POLICY")

# ---- 第 1 步：dataset.py 拉取行业全部股票数据并导出 ----
"$PYTHON" dataset.py \
    --db "$DB" \
    --industry "$INDUSTRY" \
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

echo "行业 $INDUSTRY 训练完成：成功 $trained | 跳过 $skipped | 失败 $failed"
if [ "$((trained + skipped + failed))" -eq 0 ]; then
    echo "[ERROR] $DATA_ROOT 下没有可训练的股票导出目录，" >&2
    echo "        请检查 $DATA_ROOT/run_summary_*.json 中被跳过的股票。" >&2
    exit 1
fi
[ "$failed" -eq 0 ] || exit 1

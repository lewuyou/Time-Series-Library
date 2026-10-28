#!/usr/bin/env bash
# 训练所有符合条件的股票，并在最后 N 个交易日汇总"预测第二天正收益"准确率。
# 条件：市值前 3000；数据量 >= 3 年；辅助数据含 baidu 且含 rzrq（导出后复检）。
# 切分：最后 30 个交易日为纯评估 test；val 再往前 60 个交易日。
# 用法：./10_train_all_qualified.sh [-- 透传给 train_timexer.py 的参数]
# 断点续跑：直接重跑即可（metrics.json 存在的股票自动跳过）。
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-.venv/bin/python}
DB=${DB:-../AutoTS/download/autots.duckdb}
UNIVERSE_CSV=${UNIVERSE_CSV:-dataset/universe_run_001/universe.csv}
DATA_ROOT=${DATA_ROOT:-dataset/universe_run_001}
TRAIN_ROOT=${TRAIN_ROOT:-checkpoints/universe_001}
WORKERS=${WORKERS:-8}
TORCH_THREADS=${TORCH_THREADS:-2}
TEST_DAYS=${TEST_DAYS:-30}
VAL_DAYS=${VAL_DAYS:-60}

# 1) 股票池筛选（清单存在即跳过；重新筛选请删除该文件）
if [ ! -s "$UNIVERSE_CSV" ]; then
  "$PY" select_universe.py --db "$DB" --out "$UNIVERSE_CSV" \
    --top-mv 3000 --min-years 3
fi

# 2) 批量导出 + 训练 + baidu/rzrq 特征复检
"$PY" run_universe.py \
  --db "$DB" \
  --universe-csv "$UNIVERSE_CSV" \
  --data-root "$DATA_ROOT" \
  --train-root "$TRAIN_ROOT" \
  --workers "$WORKERS" \
  --torch-threads "$TORCH_THREADS" \
  --test-days "$TEST_DAYS" \
  --val-days "$VAL_DAYS" \
  "$@"

# 3) 汇总最后 N 日每只股票的次日正收益预测准确率
"$PY" eval_last30.py --train-root "$TRAIN_ROOT" --min-days 20

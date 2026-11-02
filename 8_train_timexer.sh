#!/usr/bin/env bash
# ============================================================================
# 8. 训练 TimeXer（独立训练循环，对应审查建议 3 / 4 / 9 / 10.4 / 12.6）
#
#   对 dataset.py 导出的单股数据训练 TimeXer，下一交易日收益率预测。
#
# 与 run.py 的区别：
#   - StockTimeXerDataset 返回 (x, y) 二元组，本脚本自带训练循环，
#     只复用 models/TimeXer.py 的模型本体；
#   - features="MS"（末列为内生目标）、pred_len=1、label_len=0；
#   - patch_len 缺省自动取 4~16 内 seq_len 的最大因子（seq_len=60 时
#     为 15），并校验整除——run.py 默认 patch_len=16 对 seq_len=60 会
#     丢弃最近 12 天目标历史；
#   - 默认小模型 d_model=64 / d_ff=128 / e_layers=1，避免单股小样本
#     过拟合（可用 --d-model 等覆盖）；
#   - 每轮只看训练/验证损失做早停，测试集只在最后评估一次，
#     同时报告零预测 / 训练期均值 / 持续性（昨日收益率）三个基准。
#
# 用法（统一入口读取导出目录，不再接受单股 --data-dir）：
#   DATA_DIR=dataset/gseq OUT_DIR=checkpoints/timexer_001 bash 8_train_timexer.sh
#   # 自定义超参：bash 8_train_timexer.sh --d-model 32 --patch-len 10
#
# 输出（$OUT_DIR）：checkpoint.pth、metrics.json、test_predictions.csv
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

DATA_DIR="${DATA_DIR:-dataset/gseq}"
OUT_DIR="${OUT_DIR:-checkpoints/stock_timexer_001}"

"$PYTHON" train_stock.py --mode train \
    --data-root "$DATA_DIR" --out-dir "$OUT_DIR" "$@"

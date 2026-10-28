#!/usr/bin/env bash
# ============================================================================
# 6. 使用已有预处理参数生成最新预测窗口（infer 模式）
#
#   该模式：
#     - 从预处理文件读取股票、关键词、特征配置和列顺序；
#     - 不重新拟合均值、中位数或标准差；
#     - 不要求验证集、测试集或未来真实标签；
#     - 生成 latest.npz（[1, seq_len, F]）；
#     - 若最新核心行情窗口无效，不生成预测窗口，
#       并记录原因（manifest.json -> latest）。
#
#   省略 --as-of 时默认取库内最新行情日。
#   默认复用示例 1 的预处理文件，换数据集时可用 PREPROCESSOR 覆盖，例如：
#     PREPROCESSOR=dataset/timexer_macro_001/601398/preprocessing.json \
#         OUT_DIR=dataset/timexer_infer_002 bash 6_infer_latest.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

DB="${DB:-../AutoTS/download/autots.duckdb}"
PREPROCESSOR="${PREPROCESSOR:-dataset/timexer_run_001/601398/preprocessing.json}"
OUT_DIR="${OUT_DIR:-dataset/timexer_infer_001}"

"$PYTHON" dataset.py \
    --mode infer \
    --db "$DB" \
    --preprocessor "$PREPROCESSOR" \
    --as-of 2025-01-03 \
    --out-dir "$OUT_DIR"

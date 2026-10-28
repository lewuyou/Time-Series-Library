#!/usr/bin/env bash
# ============================================================================
# 9. 导出数据验收测试（对应审查建议 12）
#
#   对已导出的股票数据目录做离线一致性校验（不连数据库），通过则退出
#   码为 0，任何失败项退出码为 1，可用于 CI / 训练前检查。
#
# 检查内容（train/val/test 导出 18~19 项；infer 导出 10 项）：
#   1. 文件齐全（含 train_preview.csv）、矩阵有限、形状与列定义一致、
#      日期有序唯一；
#   2. 样本索引与标签一致（target_pos == end_pos + 1、窗口长度、
#      位置与日期一致、目标晚于输入）；
#   3. y_raw 可由相邻后复权收盘价重算，y_scaled 与矩阵末列一致；
#   4. train_preview.csv（训练前数据预览）与 feature_matrix.npy 逐值一致；
#   5. 切分边界：train/val/test 目标日期互不越界；
#   6. 对数收益率 expm1 与简单收益率一致（return-type=log 时）；
#   7. 审计 available <= decision（未来数据不可用断言复核）；
#   8. 验证/测试期数值扰动后重拟合，统计量与特征列不变（无泄露拟合）；
#   9. 未来行扰动后重 transform，历史行矩阵不变；
#  10. latest.npz 形状与有限性。
#
# 用法：
#   DATA_DIR=dataset/timexer_run_001/601398 bash 9_validate_dataset.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

DATA_DIR="${DATA_DIR:-dataset/timexer_run_001/601398}"

"$PYTHON" dataset.py --mode validate --data-dir "$DATA_DIR" "$@"

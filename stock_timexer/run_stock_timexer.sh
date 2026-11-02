#!/usr/bin/env bash
# ============================================================================
# TimeXer 股票预测示例脚本（dry-run 版）
#
# 前提：准备一份日频 CSV（首列 date，其余列 = 外生特征 + 目标列），
#       目标列名通过 --target 指定（代码会自动把目标列移到最后一列）。
#       示例列：date, open, high, low, volume, turnover, index_close, close
#
# 用法：
#   bash stock_timexer/run_stock_timexer.sh <root_path> <data_path> <target>
# 例：
#   bash stock_timexer/run_stock_timexer.sh ./dataset/stock/ 600519_daily.csv close
#
# 说明：
#   - --features MS：多变量输入、只预测目标列（推荐单目标场景）
#   - --enc_in 必须等于 CSV 中非 date 列的总数（含目标列）
#   - seq_len 必须能被 patch_len 整除（模型内有 assert 强校验）
#   - 本脚本为最小验证：1 epoch、小模型；跑通后再放大 --train_epochs/--d_model
#   - 产出：./checkpoints/<setting>/checkpoint.pth
#           ./results/<setting>/{metrics,pred,true}.npy
#           ./test_results/<setting>/*.pdf
#           ./result_long_term_forecast.txt（追加写入）
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # 回到仓库根目录执行 run.py

ROOT_PATH="${1:-./dataset/stock/}"
DATA_PATH="${2:-600519_daily.csv}"
TARGET="${3:-close}"

# 非	date 列数（含目标列）。示例 8 列：open,high,low,volume,turnover,amount,index_close,close
ENC_IN="${ENC_IN:-8}"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id stock_dryrun \
  --model TimeXer \
  --data custom \
  --root_path "$ROOT_PATH" \
  --data_path "$DATA_PATH" \
  --features MS \
  --target "$TARGET" \
  --freq b \
  --seq_len 60 \
  --label_len 0 \
  --pred_len 1 \
  --patch_len 15 \
  --enc_in "$ENC_IN" \
  --dec_in "$ENC_IN" \
  --c_out 1 \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 4 \
  --e_layers 1 \
  --dropout 0.1 \
  --use_norm 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --patience 3 \
  --itr 1 \
  --num_workers 0 \
  --des dryrun

echo "done. 指标见 ./result_long_term_forecast.txt 与 ./results/*/metrics.npy"

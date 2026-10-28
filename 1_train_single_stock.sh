#!/usr/bin/env bash
# ============================================================================
# 1. 单股训练（拉取数据库数据 -> 直接训练 TimeXer）
#
#   一个脚本完成两步（替代原 1_single_stock.sh + 8_train_timexer.sh）：
#     a. dataset.py 从数据库拉取 601398 的行情/百度/估值/两融/宏观数据，
#        对齐、拟合预处理并导出到 $DATA_DIR；
#     b. train_timexer.py 直接读取 $DATA_DIR 训练，输出 checkpoint.pth /
#        metrics.json / test_predictions.csv 到 $TRAIN_OUT。
#
#   训练前人工检查数据：导出目录中的 train_preview.csv（逐交易日的标准化
#   特征矩阵 + 当日原始收益率 y_raw + train/val/test 划分）。
#
# 用法：
#   bash 1_train_single_stock.sh
#   # 换数据/输出目录：
#   DATA_DIR=dataset/timexer_run_002/601398 \
#       TRAIN_OUT=checkpoints/stock_timexer_002 bash 1_train_single_stock.sh
#   # 超参透传给 train_timexer.py：
#   bash 1_train_single_stock.sh --d-model 32 --epochs 30
#   # 需要更多 dataset.py 参数（如 --start 2018-01-01、--tail 1250）时，
#   # 直接加到下面 dataset.py 的命令里。
#
# 百度数据延迟（不需要额外处理，已由来源策略覆盖）：
#   百度指数 API 天然滞后 1 天：search_all / feed 当天都只能取到昨日
#   （见 AutoTS download/sources/baidu.py），且每日采集在 09:00 之后
#   运行（晚于预测决策时点 08:30）。以决策时点 t+1 08:30 为例：日 t
#   的百度数据要到 t+1 09:00 后才入库，决策时库里最新的百度行是 t-1。
#   dataset.py 默认策略把 baidu 设为 lag_days=2（见 DEFAULT_POLICIES）：
#   观测日 d 的值视为 d+2 00:00 起可用。普通交易日行（周一~周四）决策
#   时点恰好只匹配到前一日（t-1）的百度值，即等效"延迟一天"，不会把
#   决策后才入库的 t 行当作可用（那会造成回测泄露）；周五行的决策点
#   在下周一 08:30，会匹配到周六的值（按 1 天滞后 API 周日已发布，
#   属决策时点前的公开信息；09:00 采集晚于决策点入库是已知局限，
#   见 manifest.limitations）。其余来源默认同样保守地滞后 2 天。
#   若确认某来源当天盘后即可入库，可用 POLICY=source_policy.json
#   覆盖（lag_days=0 时 release_hour 必须 >= 15，dataset.py 有硬校验）。
#
# 默认值（同旧脚本，可按数据实际覆盖区间调整）：
#   seq_len=60、pred_len=1、return_type=simple、macro_fields=close volume；
#   需满足 train-end < val-end < as-of。
# 输出目录已存在时 dataset.py 会拒绝覆盖：重复运行请换新的 DATA_DIR，
# 或删除旧目录；只重训不重拉数据时直接用 8_train_timexer.sh。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

DB="${DB:-../AutoTS/download/autots.duckdb}"
DATA_DIR="${DATA_DIR:-dataset/timexer_run_001/601398}"
TRAIN_OUT="${TRAIN_OUT:-checkpoints/stock_timexer_001}"
POLICY="${POLICY:-}"

if [ -d "$DATA_DIR" ]; then
    echo "[ERROR] 数据目录已存在：$DATA_DIR（dataset.py 拒绝覆盖）。" >&2
    echo "        请换新的 DATA_DIR；只重训不重拉数据可用 bash 8_train_timexer.sh" >&2
    exit 1
fi
DATA_ROOT="$(dirname "$DATA_DIR")"

policy_args=()
[ -n "$POLICY" ] && policy_args=(--source-policy "$POLICY")

# ---- 第 1 步：dataset.py 拉取数据库数据并导出（含 train_preview.csv）----
"$PYTHON" dataset.py \
    --db "$DB" \
    --symbol 601398 \
    --train-end 2022-12-31 \
    --val-end 2023-12-31 \
    --as-of 2024-12-31 \
    "${policy_args[@]}" \
    --out-dir "$DATA_ROOT"

echo "训练前数据预览（逐日标准化特征 + y_raw + split）：$DATA_DIR/train_preview.csv"

# ---- 第 2 步：train_timexer.py 直接训练 ----
"$PYTHON" train_timexer.py --data-dir "$DATA_DIR" --out-dir "$TRAIN_OUT" "$@"

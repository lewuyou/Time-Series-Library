# stock_timexer 使用手册（拆分后）

> 适用仓库：`/root/projects/Time-Series-Library`
> 全部命令用仓库内虚拟环境：`.venv/bin/python`
> 默认行情库：`../AutoTS/download/autots.duckdb`（`akshare_tx` 为后复权 `adjust='hfq'`）

## 0. 约定与入口总览

### 0.1 工作目录

所有命令都在仓库根执行：

```bash
cd /root/projects/Time-Series-Library
```

### 0.2 拆分后的入口

根目录只保留两个 Python 入口（其余老脚本已在 `a944176` 删除；`analyze_daily_hit.py` 薄转发入口也已在后续清理中删除，替代命令见第 4 节）：

| 入口 | 作用 |
| --- | --- |
| `dataset.py` | 单股数据引擎：`prepare` / `infer` / `validate` |
| `run.py` | 官方框架入口（与本股票流程无关） |

股票流程的四个主入口全部是模块：

| 模块入口 | 对应环节 |
| --- | --- |
| `python -m stock.selection.universe` | 股票池粗筛 |
| `python -m stock.data.export` | 批量导出（训练 data-root） |
| `python -m stock.training.runner` | 训练 / 重拟合 / 单次推理 |
| `python -m stock.production.daily_pipeline` | 每日预测、选股、落库、发布 |
| `python -m stock.evaluation.backtest` | 收盘→收盘 / 可执行回测 |
| `python -m stock.production.outcome_evaluator` | 真实收益回填与事后评价 |
| `python -m stock.production.readiness` | 发布就绪复检 |
| `python -m stock.evaluation.daily_hit` | 逐日命中分析（模块版） |
| `python -m stock.evaluation.walk_forward` | walk-forward 调度 |

### 0.3 老入口速查

| 已删除的老入口 | 替代写法 |
| --- | --- |
| `select_universe.py` | `python -m stock.selection.universe` |
| `export_global.py` | `python -m stock.data.export` |
| `train_stock.py` / `train_global_timexer.py` | `python -m stock.training.runner` |
| `predict_daily.py` | `python -m stock.production.daily_pipeline` |
| `walk_forward.py` | `python -m stock.evaluation.walk_forward` |
| `8_train_timexer.sh` | `python -m stock.training.runner --mode train` |
| `9_validate_dataset.sh` | `python dataset.py --mode validate --data-dir ...` |
| `1_train_single_stock.sh` | 单股数据用 `dataset.py --mode prepare`；单股训练用「单只 universe → export → runner」（见 1.4 / 2.3） |
| `2_train_industry_all.sh` / `3_train_industry_top10.sh` | 不再有行业批量脚本；把行业股票写成 `universe.csv` 后走 `stock.data.export` + `stock.training.runner` |
| `4_train_macro_symbols.sh` | 单股 `dataset.py --mode prepare --macro-symbols ...`；训练走 2.3 |
| `6_infer_latest.sh` | `python dataset.py --mode infer ...`（生产路径由 `daily_pipeline` 内部调用） |
| `10_train_all_qualified.sh` | 粗筛 → 导出 → 训练 → 回测/评价，见本手册第 1~4 步 |

> 老脚本里的 `train_timexer.py` 在仓库从未存在，因此 `8_train_timexer.sh`（早期版本）与 `2/3/4` 的第二步本来就跑不通；现在统一由 `stock.training.runner` 取代。

---

## 1. 数据导出（DuckDB → data-root）

### 1.1 全池：粗筛股票池（可选，但推荐）

```bash
cd /root/projects/Time-Series-Library

.venv/bin/python -m stock.selection.universe \
  --db ../AutoTS/download/autots.duckdb \
  --out dataset/global_001/universe.csv \
  --top-mv 3000 --min-years 3 --baidu-fresh-days 30
```

- 输入：`../AutoTS/download/autots.duckdb`
- 输出：`dataset/global_001/universe.csv`（列 `symbol,name,industry_l2`）与 `dataset/global_001/universe_meta.json`
- 历史时点粗筛：追加 `--as-of 2024-01-02`（市值排名与简称仍是当前快照，属“部分 PIT”）
- 成功判据：命令退出码 0，终端打印 `as_of=... | 筛选漏斗：{...}`

> 老入口：`select_universe.py` 已删除，替代写法就是本命令。

### 1.2 全池：批量导出（自动倒推切分）

```bash
cd /root/projects/Time-Series-Library

.venv/bin/python -m stock.data.export \
  --db ../AutoTS/download/autots.duckdb \
  --universe-csv dataset/global_001/universe.csv \
  --data-root dataset/global_001 \
  --workers 4 --seq-len 60 --test-days 30 --val-days 60
```

- 输入：行情库 + `dataset/global_001/universe.csv`
- 输出：
  - `dataset/global_001/<symbol>/`：`feature_matrix.npy`、`samples.csv`、`preprocessing.json`、`feature_schema.json`、`latest.npz`、`train_preview.csv`、`manifest.json` 等
  - `dataset/global_001/export_config.json`：本次导出的 `as_of / train_end / val_end / test_start / seq_len` 与脚本哈希
  - `dataset/global_001/export_results.csv`：逐股 `status`
- 成功判据：
  - 退出码 0
  - `export_results.csv` 中没有 `status=error`
  - 至少一只 `status=ok`（`filtered_no_baidu` / `filtered_no_rzrq` 是按设计排除）
  - `export_config.json` 的 `as_of` 为库内最新行情日（本手册实测为 `2026-10-14`）

> 注意：`universe.csv` 必须位于 `--data-root` 目录下（`daily_pipeline` 会从 `data-root/universe.csv` 读名称/行业）。把 `--out` 直接放在 data-root 即可。
> 老入口：`export_global.py` 已删除，替代写法就是本命令。

### 1.3 显式切分（walk-forward 每折 / 固定边界）

```bash
.venv/bin/python -m stock.data.export \
  --db ../AutoTS/download/autots.duckdb \
  --universe-csv dataset/global_001/universe.csv \
  --data-root dataset/global_2024 \
  --workers 4 --seq-len 60 \
  --as-of 2024-12-31 --train-end 2024-06-28 --val-end 2024-09-30
```

规则：`train-end < val-end < as-of <= 库内最新行情日`，三者必须同时给；`--train-start` 可选（rolling 窗口下界）。

### 1.4 单股导出（对应老 `1_train_single_stock.sh` 的第一步）

```bash
.venv/bin/python dataset.py --mode prepare \
  --db ../AutoTS/download/autots.duckdb --symbol 601398 \
  --out-dir dataset/single_001 --seq-len 60 \
  --train-end 2023-12-29 --val-end 2025-06-30 --as-of 2026-10-14
```

- 输出：`dataset/single_001/601398/`（文件同上）
- 成功判据：终端打印 `[OK] 601398 ... | F=... | train .../val .../test ... | latest=True`
- 注意：`dataset.py --mode prepare` 的切分必须显式给 `--train-end` 与 `--val-end`（没有 `--test-days/--val-days` 自动倒推）；`601398` 实测导出状态为 `filtered_no_rzrq`，只适合看数据，不进入共享训练。

导出后验收（对应老 `9_validate_dataset.sh`）：

```bash
.venv/bin/python dataset.py --mode validate --data-dir dataset/single_001/601398
```

- 成功判据：`validate ...: 22/22 项通过`，退出码 0

单股推理窗口（对应老 `6_infer_latest.sh`）：

```bash
.venv/bin/python dataset.py --mode infer \
  --db ../AutoTS/download/autots.duckdb \
  --preprocessor dataset/single_001/601398/preprocessing.json \
  --out-dir dataset/single_infer_001 --as-of 2026-10-14
```

- 输出：`dataset/single_infer_001/601398/latest.npz` 等
- 成功判据：`[OK] ... latest=True`，退出码 0

> 老入口：`1_train_single_stock.sh` / `6_infer_latest.sh` 的 dataset 部分，替代写法就是上面两条；训练部分见 2.3。

---

## 2. 训练

### 2.1 普通训练（阶段 1：Val 早停选择）

```bash
cd /root/projects/Time-Series-Library

.venv/bin/python -m stock.training.runner --mode train \
  --data-root dataset/global_001 \
  --out-dir checkpoints/global_001 \
  --epochs 30 --patience 5 --batch-size 256 --workers 0 \
  --d-model 64 --e-layers 2 --n-heads 4 --patch-len 10 --device auto
```

- 输入：1.2 生成的 `dataset/global_001`（必须有 `export_config.json` 和 `export_results.csv`）
- 输出目录 `checkpoints/global_001/`：
  - `checkpoint.pth`
  - `feature_schema.json`
  - `history.json`
  - `schema.json`
  - `selection_result.json`（`best_epoch`、`ex_mask`、`as_of/train_end/val_end/test_start`）
  - `test_metrics.json`、`test_metrics_by_stock.csv`、`test_predictions.csv`
- 成功判据：终端出现 `共享 checkpoint 已保存：checkpoints/global_001/checkpoint.pth`，退出码 0

> 老入口：`train_stock.py` / `train_global_timexer.py` / `8_train_timexer.sh` 已删除，替代写法就是本命令。
> 训练侧只接受 `stock.data.export` 产出的 data-root；直接拿 `dataset.py --out-dir` 的单股目录训练会因缺少 `export_config.json` / `export_results.csv` 失败。

### 2.2 两阶段 refit（阶段 2：Train+Val 全量重拟合）

第一阶段先用 2.1 跑出 `selection_result.json`，记下 `best_epoch`：

```bash
.venv/bin/python -c "import json;print(json.load(open('checkpoints/global_001/selection_result.json'))['best_epoch'])"
```

第二阶段导出（`train_test` 布局，必须给 `--as-of` 与 `--train-end`，不能给 `--val-end`）：

```bash
.venv/bin/python -m stock.data.export \
  --db ../AutoTS/download/autots.duckdb \
  --universe-csv dataset/global_001/universe.csv \
  --data-root dataset/refit_001 \
  --workers 4 --seq-len 60 \
  --split-layout train_test \
  --as-of 2026-10-14 --train-end 2026-08-25
```

> 这里的 `--train-end` 取第一阶段 `selection_result.json` 里的 `val_end`（本手册实测为 `2026-08-25`），即把原验证区间并入最终训练。

重拟合：

```bash
.venv/bin/python -m stock.training.runner --mode refit \
  --data-root dataset/refit_001 \
  --out-dir checkpoints/refit_001 \
  --epochs 1 --batch-size 256 --workers 0 \
  --d-model 64 --e-layers 2 --n-heads 4 --patch-len 10 --device auto \
  --selection-result checkpoints/global_001/selection_result.json
```

- `--epochs` 必须等于第一阶段 `best_epoch`（提供 `--selection-result` 时会校验；本手册实测 `best_epoch=1` + `--epochs 1` 通过）
- 输出目录 `checkpoints/refit_001/`：`checkpoint.pth`、`feature_schema.json`、`fit_metadata.json`、`history.json`、`schema.json`、`test_metrics.json`、`test_metrics_by_stock.csv`、`test_predictions.csv`
- 成功判据：退出码 0，`fit_metadata.json` 存在
- 说明：阶段 2 的 `test_predictions.csv` 覆盖 `train_end` 之后的区间，属于协议产物，不是严格样本外

> 老入口：`walk_forward.py --refit` 内部的同一套两阶段逻辑，现在手工执行就是上面两步；批量折请用 `stock.evaluation.walk_forward`。

### 2.3 单股训练（对应老 `1_train_single_stock.sh` 的第二步）

共享训练器要求 data-root 是导出目录，因此“单股训练”= 单只股票的 universe：

```bash
mkdir -p dataset/train_single_001
printf 'symbol,name,industry_l2\n000001,平安银行,银行Ⅱ\n' > dataset/train_single_001/universe.csv

.venv/bin/python -m stock.data.export \
  --db ../AutoTS/download/autots.duckdb \
  --universe-csv dataset/train_single_001/universe.csv \
  --data-root dataset/train_single_001 \
  --workers 1 --seq-len 60 --test-days 30 --val-days 60

.venv/bin/python -m stock.training.runner --mode train \
  --data-root dataset/train_single_001 \
  --out-dir checkpoints/single_001 \
  --epochs 30 --patience 5 --batch-size 256 --workers 0 \
  --d-model 64 --e-layers 2 --n-heads 4 --patch-len 10 --device auto
```

- 成功判据：`export_results.csv` 中该股票 `status=ok`；训练打印 `共享 checkpoint 已保存`

> 老入口：`1_train_single_stock.sh` 与 `4_train_macro_symbols.sh` 的训练段。行业批量老脚本（`2_` / `3_`）同理：把行业股票写进 `universe.csv` 即可。

### 2.4 切换模型：TimeXer vs TimesNet

第二个模型 TimesNet 已接入（`stock/adapters/timesnet.py`，网络本体拷贝为 `stock/adapters/timesnet_model.py`）。`--model` 参数在 `stock.training.runner` 与 `stock.evaluation.walk_forward` 都可用，名字大小写不敏感（`--model timesnet` 等价 `--model TimesNet`）。

两模型的关键差异：

| | TimeXer | TimesNet |
|---|---|---|
| checkpoint 格式号 | `global_timexer_typed_v1` | `timesnet_mv_v1` |
| 输入语义 | 内生 patch token（目标）+ 外生 value token，区分内生/外生 | **全特征通道**同质读入（含目标末列、外生值列、`__missing` 缺失标记列），无 token 结构 |
| `ex_mask` | 支持（`--ex-mask`，token 级掩码） | **不支持**（开了 `--ex-mask` 启动即报错，不静默退化） |
| `use_norm` | 固定 0（特征保持导出期尺度） | 默认 0；`--use-norm` 可开（会污染 0/1 缺失标记通道，仅作消融对照） |
| 目标通道 | projection 输出目标 | 输出末列（schema 最后一列＝目标列） |
| 模型专属参数 | `--d-model --e-layers --n-heads --patch-len --dropout --ex-mask` | `--d-model --d-ff --e-layers --top-k --num-kernels --dropout --use-norm` |

TimesNet 参数含义：`--top-k`＝FFT 主周期个数（约束 `top-k ≤ (seq_len+1)//2`，否则启动前报错）；`--num-kernels`＝Inception 卷积核尺寸数（默认 6）；`--d-ff`＝TimesBlock 中间宽度（缺省 `d-model*2`）；`seq_len` 不需要被 `top-k` 整除（TimesBlock 零填充对齐）。

TimesNet 训练（其余与 2.1/2.3 完全相同，只换 `--model` 和专属参数）：

```bash
.venv/bin/python -m stock.training.runner --mode train \
  --model TimesNet \
  --data-root dataset/global_001 \
  --out-dir checkpoints/global_001_timesnet \
  --epochs 30 --patience 5 --batch-size 256 --workers 0 \
  --d-model 64 --e-layers 2 --top-k 5 --num-kernels 6 --device auto
```

TimesNet 的 walk-forward（第 4 节回测链路同理，只是加 `--model`）：

```bash
.venv/bin/python -m stock.evaluation.walk_forward \
  --db ../AutoTS/download/autots.duckdb \
  --universe-csv dataset/global_001/universe.csv \
  --run-root walkforward/timesnet_001 \
  --first-test-date 2025-01-01 --last-test-date 2025-12-31 \
  --val-days 60 --test-days 20 --seq-len 60 \
  --model timesnet --d-model 64 --e-layers 2 --top-k 5 \
  --epochs 30 --patience 5 --device auto
```

预测/每日选股/回测命令**不需要任何改动**：checkpoint 里登记了 `format`（`timesnet_mv_v1`），`predict` / `predict_daily` 自动反查适配器。同一份 data-root 用两个模型训练，产出的 `schema.json` / 特征哈希 / 日期覆盖完全相同（已实测），`test_predictions.csv` 字段结构相同，只是 `y_pred` 不同。

通用路径（runner / predictor / walk_forward / daily_pipeline）不含任何模型专属硬编码：模型构造、参数校验、CLI 透传都由 `stock/adapters/` 的适配器声明（`cli_params` / `validate_args`）。新增第三个模型＝在 `stock/adapters/` 加一个适配器并在 `stock/adapters/__init__.py` 追加一行导入。

---

## 3. 每日预测 / 选股

### 3.1 生成、落库并发布

```bash
cd /root/projects/Time-Series-Library

.venv/bin/python -m stock.production.daily_pipeline \
  --checkpoint checkpoints/global_001/checkpoint.pth \
  --data-root dataset/global_001 \
  --infer-root dataset/infer_001 \
  --run-db predictions/predictions.db \
  --as-of 2026-10-14 \
  --workers 2 --top 5
```

- `--checkpoint`：2.1 或 2.2 的产物
- `--data-root`：训练期导出目录（读 `preprocessing.json` / `universe.csv`）
- `--infer-root`：本次推理窗口输出目录；**每次换新目录**（已存在会报 `FileExistsError`），或显式 `--skip-export` 复用（`--skip-export` 未实测）
- `--run-db`：预测库，唯一事实源；相对路径按仓库根解析
- `--as-of`：推理基准日（t）；缺省取库内最新行情日
- 输出：
  - `predictions/predictions.db`：`prediction_runs`（run 元信息、`publish_status`、`blockers`）+ `predictions`（逐股预测，主键冻结）
  - `predictions/predict_<target_date>.csv`：**只是视图**，首行固定为 `# 视图，以库为准（authoritative: SQLite prediction store）`
- 成功判据：终端打印 `已落库 run_id=...（...；状态 published）`，退出码 0

> 老入口：`predict_daily.py` 已删除，替代写法就是本命令。

### 3.2 发布阻断

不传 `--max-lag-days` 时默认门槛为 5 天；`--min-coverage` 默认 0.8；`--max-failure-ratio` 默认 0.2。

```bash
.venv/bin/python -m stock.production.daily_pipeline \
  --checkpoint checkpoints/global_001/checkpoint.pth \
  --data-root dataset/global_001 \
  --infer-root dataset/infer_blocked_001 \
  --run-db predictions/predictions.db \
  --as-of 2026-10-14 --workers 2 --top 5
```

- 数据滞后超过门槛时：终端打印 `[BLOCK] 数据滞后 N 天 > 允许 5 天`，run 仍以 `blocked` 落库，但**不写 CSV 视图**，退出码 1（已实测）
- 复跑历史数据做演练时可放宽：`--max-lag-days 1000`（已实测 published）

发布后复检（已实测，`ready=True` 退出码 0；坏 checkpoint 时 `ready=False` 退出码 1）：

```bash
.venv/bin/python -m stock.production.readiness \
  --run-db predictions/predictions.db \
  --run-id <run_id> \
  --checkpoint checkpoints/global_001/checkpoint.pth \
  --today 2026-11-03 --max-lag-days 5
```

### 3.3 从库里读结果（不要把 CSV 当事实源）

```bash
.venv/bin/python - <<'PY'
import sqlite3
import pandas as pd

con = sqlite3.connect("predictions/predictions.db")
runs = pd.read_sql_query(
    "SELECT run_id, as_of_date, target_date, publish_status, blockers "
    "FROM prediction_runs ORDER BY created_at DESC LIMIT 5", con)
print(runs.to_string(index=False))

run_id = runs.iloc[0]["run_id"]
print(pd.read_sql_query(
    "SELECT rank, symbol, y_pred, pred_return_pct, status, signal "
    "FROM predictions WHERE run_id = ? ORDER BY rank",
    con, params=(run_id,)).to_string(index=False))
con.close()
PY
```

- 已实测输出：`run_id / target_date / publish_status` 与逐股 `rank,symbol,y_pred,pred_return_pct,status,signal`
- 阻塞原因在 `prediction_runs.blockers`；同一 `target_date` 的多次 run 用 `ORDER BY created_at` 区分
- 需要 CSV 视图时用库导出函数（不要手工改 CSV）：`stock.production.prediction_store.export_view(con, run_id, path, universe=...)`

> 老入口：`predictions/predict_*.csv` 曾是可覆盖的“事实源”，现在只作为视图；老 `6_infer_latest.sh` 的 `dataset.py --mode infer` 仍可单股排查，但生产入口是 `daily_pipeline`。

---

## 4. 回测与事后评价

### 4.1 准备可执行口径的后复权行情

`--prices` 需要 `symbol,date,open,close`，日期范围要覆盖 `target_date` 与持有期：

```bash
cd /root/projects/Time-Series-Library

.venv/bin/python - <<'PY'
import duckdb
import pandas as pd

con = duckdb.connect("../AutoTS/download/autots.duckdb", read_only=True)
pred = pd.read_csv("checkpoints/global_001/test_predictions.csv",
                   dtype={"symbol": str})
symbols = sorted(pred["symbol"].unique())
lo, hi = pred["target_date"].min(), pred["target_date"].max()
sql = (
    "SELECT symbol, date, open, close FROM akshare_tx WHERE adjust='hfq' "
    "AND symbol IN (" + ",".join(["?"] * len(symbols)) + ") "
    "AND date >= ?::DATE - INTERVAL 10 DAY "
    "AND date <= ?::DATE + INTERVAL 10 DAY ORDER BY date, symbol"
)
con.execute(sql, symbols + [lo, hi]).df().to_csv(
    "dataset/backtest_prices.csv", index=False)
print("prices:", len(symbols), "symbols", lo, "->", hi)
PY
```

- 输出：`dataset/backtest_prices.csv`
- 已实测：3 只股票、30 个预测日，生成 111 行有效行情

### 4.2 用训练期测试预测回测

```bash
.venv/bin/python -m stock.evaluation.backtest \
  --predictions checkpoints/global_001/test_predictions.csv \
  --prices dataset/backtest_prices.csv \
  --out-dir checkpoints/global_001/backtest \
  --hold-days 1 --fee-bps 5 --slippage-bps 5
```

- 输入：`symbol,target_date,y_pred[,y_true]` + `symbol,date,open,close`
- 输出目录：
  - `backtest_report.json`（`close_to_close` 与 `executable` 两套口径）
  - `backtest_signals.csv`、`backtest_trades.csv`、`backtest_turnover.csv`
- 成功判据：退出码 0；报告中 `days / signals / signal_days` 非空；`y_true` 存在时输出 `close_to_close`，给了 `--prices` 时输出 `executable`
- 口径：`close_to_close` 用 `y_true`（t 收盘→t+1 收盘）；`executable` 为 t+1 开盘买入、持有 `hold_days` 后收盘卖出，双边扣费与滑点

### 4.3 把库里的 run 喂给 backtest

`backtest` 没有 `--run-db` / `--run-id` 参数，必须先按回测契约从库里导出 CSV（已实测）：

```bash
.venv/bin/python - <<'PY'
import sqlite3
import pandas as pd

run_id = "<从 prediction_runs 查到的 run_id>"
con = sqlite3.connect("predictions/predictions.db")
frame = pd.read_sql_query(
    "SELECT symbol, target_date, y_pred FROM predictions "
    "WHERE run_id = ? AND y_pred IS NOT NULL",
    con, params=(run_id,))
con.close()
frame.to_csv("predictions/run_predictions.csv", index=False)
print("exported", len(frame), "rows for", run_id)
PY
```

然后按 4.2 的命令，把 `--predictions` 换成 `predictions/run_predictions.csv`；`--prices` 按该 run 的 `target_date` 区间准备。

- 已实测：库内 3 条预测 → `run_predictions.csv` → `backtest` 退出码 0
- 该 CSV 只有 `y_pred`，因此只输出信号与可执行口径；如果已在 4.4 回填标签，可在 SQL 中 `LEFT JOIN prediction_outcomes` 取 `realized_return AS y_true` 再算 `close_to_close`（这个 JOIN 写法未单独验证）

### 4.4 回填真实收益并事后评价

标签文件格式：`symbol,target_date,y_true`，`y_true` 是**小数收益率**（不是百分数）：

```bash
.venv/bin/python -m stock.production.outcome_evaluator \
  --run-db predictions/predictions.db \
  --run-id <run_id> \
  --labels dataset/labels.csv \
  --out predictions/eval_<run_id>.json
```

- 输出：库内 `prediction_outcomes` 表（允许标签修订，与冻结的 `predictions` 解耦）+ 可选评价 JSON
- 成功判据：终端打印 `回填 N 条（标签文件 M 条）` 及 `coverage / signal_hit_rate / base_up_rate / signal_excess_hit` 等指标，退出码 0
- 已实测：1 条预测 / 1 条标签，`coverage=1.0`，报告正常输出

> 老入口：`analyze_daily_hit.py` 已删除；逐日命中分析用 `python -m stock.evaluation.daily_hit --run-dir <ckpt目录>`（`--help` 已通过）。批量样本外评估用 `python -m stock.evaluation.walk_forward`（CLI 已通过 `--help`，完整调度未在本手册实跑）。

---

## 5. `dataset.py` 的现状（P5 已拆分为薄入口 + `stock/data/` 职责模块）

### 5.1 拆分后的结构

`dataset.py`（根目录）现在是一个**薄 CLI 入口**：保留原来的全部 CLI 参数与 `--mode prepare|infer|validate` 行为，实现全部按职责拆进 `stock/data/`，旧函数名（`prepare_one`、`load_calendar`、`make_samples`……）在 `dataset` 模块命名空间仍可 import（兼容转发，`import dataset as ds` 的旧代码不受影响）。

| 模块 | 职责（关键符号） |
| --- | --- |
| `stock/data/common.py` | 共享常量与小工具：`VERSION`/`SPLIT_LAYOUTS`/字段清单、`norm_symbol`/`sql_symbol`/`parse_dates`/`as_numeric`/`load_policies` |
| `stock/data/calendar.py` | 交易日历：`load_calendar`/`calendar_fingerprint`/`calendar_dates_sha256` |
| `stock/data/splits.py` | 切分边界倒推：`resolve_split`（自 `stock/data/export.py` 原样迁出，函数体逐字未动，export 侧转发 import） |
| `stock/data/sources.py` | 原始数据读取与序列规格：`select_stocks`/`load_prices`/`load_stock_aux`/`load_macro`/`make_specs_from_long` |
| `stock/data/pit.py` | Point-in-time 对齐与审计：`align_one`/`source_watermarks`/`history_drift_report` |
| `stock/data/features.py` | 特征构造：`fixed_transform`/`build_features` |
| `stock/data/targets.py` | 目标列（此前已存在）：`target_series`/`target_spec`/`label_valid` |
| `stock/data/preprocessing.py` | 预处理：`fit_preprocessing`/`transform` |
| `stock/data/windows.py` | 样本窗口：`make_samples` |
| `stock/data/builder.py` | 单股导出编排：`prepare_one`/`save_latest`/`load_asof`/`coverage_report` |
| `stock/data/validation.py` | 离线验收：`validate_dataset`（`--mode validate` 的全部检查项） |
| `stock/data/export.py` | 批量导出（P3 已存在）；`resolve_split` 已迁往 `splits.py` 并从那里转发 |
| `dataset.py` | 薄入口：docstring + 兼容转发 + `parse_args`/`main` + `StockTimeXerDataset`（单股独立训练循环） |

### 5.2 `dataset_script_sha256` 兼容规则（重要）

- 该哈希 = sha256(根目录 `dataset.py`)，登记在 `export_config.json`（export 写入）与 checkpoint/walk-forward manifest（runner 登记）。它只是**导出脚本的身份指纹，不是数据契约**。
- 旧 data-root 的 `--mode validate` / `--mode infer` **不比对该哈希**：脚本改动（含 P5 拆分）后旧目录照常可用。
- 只有**原地重导**（`--out-dir` 指向已存在的导出目录）会被 `prepare_one` 的 `FileExistsError` 拒绝——口径变更必须换新目录，避免新旧代码产物混在同一 data-root。
- 判断两份导出是否同口径，比对 manifest 的 `feature_schema`/日历指纹/config，不要用该哈希。
- 上述规则也写在 `dataset.py` 头部注释与 `tests/test_dataset_equiv.py`。

### 5.3 等价性验证（P5 实测）

拆分前 HEAD `f71203f` 与拆分后，同一只股票（601398）、同一组参数（`--as-of 2026-10-14 --train-end 2023-12-29 --val-end 2025-06-30 --seq-len 60`）各导一份：`feature_matrix.npy` / `dates.npy` / `samples.csv` / `aligned_values.csv` / `features_unscaled.csv` **逐字节一致**，`preprocessing.json` 既有键与值一致。固化在 `tests/test_dataset_equiv.py`（`-m slow`，本地已实跑 5/5 通过；基线对比需 `P5_EQ_BASELINE` 指向旧代码导出目录）。

### 5.4 它在新结构里的角色

- `--mode prepare|infer|validate` 单股排查入口（本手册第 1 节的底层）。
- 被 `python -m stock.data.export` 调用（批量导出复用 `dataset.prepare_one`）。
- **不被训练脚本直接调用**：训练侧读 data-root，经 `stock.data.panel.load_catalog` / `stock.data.schema` 装载。

---

## 6. 本手册命令的验证状态

| 命令 | 状态 |
| --- | --- |
| 1.1 `stock.selection.universe` | `--help` 通过；完整筛选未在本轮重跑 |
| 1.2 `stock.data.export`（自动倒推） | 实跑通过（1 只 / 3 只股票） |
| 1.3 `stock.data.export`（显式边界） | `--help` 通过；参数规则来自模块文档 |
| 1.4 `dataset.py --mode prepare` / `validate` / `infer` | 实跑通过（`601398`，22/22 项） |
| 2.1 `runner --mode train` | 实跑通过（1 epoch） |
| 2.2 `runner --mode refit` | 实跑通过（1 epoch，`best_epoch=1`） |
| 2.3 单只 universe 导出 + 训练 | 实跑通过（`000001`） |
| 2.4 `runner --mode train --model TimesNet` | 实跑通过（1 epoch，schema/特征哈希/日期覆盖与 TimeXer 逐字节一致） |
| 2.4 `runner --mode predict --model TimesNet checkpoint` | 实跑通过（`status=ok`） |
| 2.4 `walk_forward --model timesnet` | 实跑通过（1 折，`fold_000`，5 个测试日） |
| 5.3 数据层拆分 A/B（拆分前 HEAD vs 拆分后，601398 同参数） | 实跑通过（核心产物逐字节一致） |
| 5.3 `tests/test_dataset_equiv.py -m slow`（含 `P5_EQ_BASELINE` 基线对比） | 实跑通过（5 passed） |
| 3.1 `daily_pipeline` 发布 | 实跑通过（`published`，`run_id=run-20261015-20261103T084918-07f5dd6d`） |
| 3.2 阻断 | 实跑通过（`blocked`，退出码 1，不写 CSV） |
| 3.3 读库 sqlite3 + pandas | 实跑通过 |
| 3.2 `readiness` 复检 | 实跑通过（`ready=True`，退出码 0） |
| 4.1 生成 prices CSV | 实跑通过 |
| 4.2 `backtest` | 实跑通过（`close_to_close` + `executable`） |
| 4.3 库 run → CSV → `backtest` | 实跑通过 |
| 4.4 `outcome_evaluator` | 实跑通过 |
| `stock.evaluation.walk_forward` | `--help` 通过；完整折调度未实跑 |
| `stock.evaluation.daily_hit`（原 `analyze_daily_hit.py`，已删） | `--help` 通过；未在本轮实跑 |
| `--skip-export` | 未验证 |
| `prediction_outcomes` JOIN 取 `y_true` | 未验证 |

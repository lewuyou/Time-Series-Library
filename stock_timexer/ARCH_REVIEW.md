# 股票预测平台化重构建议 · 落地评审（以现有代码为准）

> 评审对象：`/root/projects/Time-Series-Library`
> 实际 HEAD `72ea19d`（提交信息 "11"），`deee9cb` 是其父提交；`git status` 为空（描述中的"未提交改动"未观察到，可能已被提交或权限差异）。
> 只读探查，未改仓库任何既有文件，未运行训练。

## 0. 总判断

**必须做（收益高、成本低、当前就有错误结论风险）**
1. 新增/补全**预测产物契约**（`symbol/as_of_date/target_date/model_id/dataset_id/y_pred/status`）与**预测落库**（SQLite，写一次不覆盖）。
2. 把**可执行回测**补上，并与 `predict_daily` / `analyze_daily_hit` 共用同一个选股函数；当前只有"收盘→收盘"统计口径，信号却在 t+1 08:30 才形成，不能当收益结论。
3. 给**特征 schema 加版本与 hash**并写进训练/推理产物；给**标签定义**收敛成显式规范（目标、`label_valid`、可用时点），不再散落在 `dataset.py` 与文档里。
4. 修掉 `walk_forward.py` 里写死的 `--model TimeXer`，否则"支持第二模型"是假的。

**可以忽略（当前无需求，属于过度设计）**
- `data/repository.py` 的 DuckDB 抽象层、`ensemble`、`monitoring`、`attribution`、`tuning`、`experiments/<id>/` 完整目录制、一次性拆 30+ 文件、真正的 PIT 改造（库内无历史快照）。
- 建议里"已经存在"的模块不要重复新建：`models/base` ≈ `stock/adapters/base.py`，`models/registry` ≈ `stock/adapters/registry.py`，`PanelWindows` ≈ `stock/data/panel.py`，`validation` ≈ `dataset.validate_dataset`，选股规则 ≈ `stock/selection/rules.py`。
- 建议对现状的最大误判：以为 `train_global_timexer.py` 还是一大坨，实际它已是 40 行 shim，主体在 `stock/training/runner.py`（616 行）。

**一句话方针**：保留 `stock/` 现有分层与入口 shim，先补契约和评估闭环，再补生产持久化，最后才考虑拆 `dataset.py`；不要新起根目录 `data/ models/ evaluation/`。

## 1. 现状骨架

### 1.1 仓库与数据存储

- 数据存储是 **DuckDB**：默认 `../AutoTS/download/autots.duckdb`（`dataset.py:209`）。实测表：`akshare_tx`（15,920,867 行）、`baidu`（26,058,514）、`rzrq`（5,730,726）、`stock_list`（5,224）、`holiday_calendar`（5,113），另有 `guzhi`、`nbjb`、`stock_features`、`tradingview_macro*`、`tradingview_study`。
- 没有历史市值/历史简称/发布时间/修订版本表；`select_universe.py` 的 `pit_status.snapshot_criteria` 已明确承认（`stock/selection/universe.py:274-296`）。
- 导出产物目录 `dataset/` 当前只有 `gseq/universe.csv` 与 `universe_meta.json`（此前跑过的数据目录不在仓库内）。`predictions/` 为空。

### 1.2 `stock/` 包（24 个 py，4197 行）

| 子包 | 文件（行数） | 职责 |
|---|---|---|
| `stock/data` | `schema.py(136)`、`panel.py(350)`、`export.py(577)` | 列布局单一事实源；导出目录装载 + `PanelWindows`；批量导出调度 |
| `stock/adapters` | `base.py(37)`、`registry.py(52)`、`timexer.py(87)`、`__init__.py(33)` | 模型适配器 ABC / 注册表 / TimeXer 接入 |
| `stock/training` | `runner.py(616)`、`trainer.py(119)`、`artifacts.py(55)` | train/refit 主流程、epoch 循环与测试预测、环境登记 |
| `stock/inference` | `predictor.py(155)` | 单股 `predict` + 全池 `score` |
| `stock/selection` | `rules.py(33)`、`universe.py(317)` | 混合选股规则；股票池粗筛（部分 PIT） |
| `stock/evaluation` | `metrics.py(80)`、`daily_hit.py(151)`、`walk_forward.py(1088)` | 指标/测试报告、逐日命中、walk-forward 调度 |
| `stock/production` | `daily_pipeline.py(217)` | 每日 infer 导出 + 打分 + 选股 + CSV |
| `stock/utils.py` | 48 | 原子写 JSON / hash / 日期 |

### 1.3 根目录入口与 `dataset.py`

| 文件 | 行数 | 现状 |
|---|---|---|
| `dataset.py` | 2303 | **仍是单体数据引擎**：DuckDB→价格/辅助对齐→特征→样本→预处理→导出→`validate`→`infer` |
| `export_global.py` | 26 | 转发 `stock/data/export.py` |
| `train_global_timexer.py` | 40 | 转发 `stock/training/runner.py` + 重导出旧符号 |
| `train_stock.py` | 21 | 统一训练入口（`--model`） |
| `walk_forward.py` | 20 | 转发 `stock/evaluation/walk_forward.py` |
| `predict_daily.py` | 15 | 转发 `stock/production/daily_pipeline.py` |
| `select_universe.py` | 15 | 转发 `stock/selection/universe.py` |
| `analyze_daily_hit.py` | 14 | 转发 `stock/evaluation/daily_hit.py` |
| `8_train_timexer.sh` | 34 | **已坏**：第 34 行调用不存在的 `train_timexer.py` |

`dataset.py` 内部关键实现（拆分的真实难点）：
- 交易日历与校准 `load_calendar`（`dataset.py:283`）、日历指纹 `calendar_fingerprint`（`1117`）。
- 辅助数据 PIT 近似对齐 `align_one`（`639-703`）：`available = observed + lag_days + release_hour`，`merge_asof(direction="backward")`，并断言 `available <= decision`。
- 特征构造 `build_features`（`713-847`）；**标签** `prev=close.shift(1)`、`simple_return=close/prev-1`（`752-756`），`core_valid = valid_price & target.notna()`（`845`）。
- 样本与切分 `make_samples`（`849-942`）：按 **target_date** 分 train/val/test，`label_valid` 独立字段。
- 预处理拟合/冻结/变换 `fit_preprocessing`（`944-1019`）、`transform`（`1022-1038`）。
- 最新窗口 `save_latest`（`1041-1062`）；覆盖统计 `coverage_report`（`1082`）。
- 离线验收 `validate_dataset`（`1641-1990`），9 类检查含 `available<=decision`、扰动重拟合不变性。
- `manifest` 已登记 `as_of`、`forecast_time`、`source_watermarks`、`quality`、`limitations`（`1454-1519`）。

### 1.4 已有抽象（建议里声称要新建、其实已存在）

1. **模型契约/注册**：`ModelAdapter` ABC（`stock/adapters/base.py:6-37`），按名/按 checkpoint format 取适配器（`registry.py:9-52`），TimeXer 注册在 `timexer.py:49-87`。
2. **特征布局单一事实源**：`stock/data/schema.py:21-136`（`TARGET`、伴随列命名、`align_features`、`exogenous_mask_layout`、`compute_ex_mask`）。
3. **面板 Dataset**：`PanelWindows`（`stock/data/panel.py:159-350`），mmap + 每 epoch 股票等权（`281-288`）。
4. **导出目录强校验**：`load_catalog`（`panel.py:25-156`）检查 `export_config.json`、每股 `preprocessing.json`、`seq_len/train_end/val_end/sample_selection` 一致。
5. **选股规则单一实现**：`hybrid_signal`（`stock/selection/rules.py:12-33`），`predict_daily` 与 `analyze_daily_hit` 都已调用（`daily_pipeline.py:164`、`daily_hit.py:47`）。
6. **推理契约严格校验**：checkpoint format 反查适配器、预处理逐字段相等、窗口维度/有限值检查（`stock/inference/predictor.py:39-45`、`55-58`）。
7. **walk-forward 协议**：折生成 `build_folds`（`walk_forward.py:206`）、DB 指纹 `read_db_state/ensure_db_state`（`167-204`）、两阶段 refit、协议对照 `protocol_comparison`（`413`）。
8. **原始尺度测试指标**：`y_true=y_raw`、`y_pred=pred_scaled*scale+mean`（`stock/training/trainer.py:88-111`），并非标准化尺度。

### 1.5 产物现状

- 导出目录每股：`feature_matrix.npy`、`dates.npy`、`samples.csv`、`preprocessing.json`、`manifest.json`、`latest.npz`、`aligned_values.csv`、`features_unscaled.csv`、`alignment_audit.csv`、`train_preview.csv`。
- 训练目录：`checkpoint.pth`（含 `format/model/model_config/schema/export_config/preprocessing/train_mean`，refit 另含 `refit`）、`history.json`、`schema.json`、`selection_result.json`、`fit_metadata.json`、`test_predictions.csv`、`test_metrics.json`、`test_metrics_by_stock.csv`。
- 生产目录：`predictions/predict_<target_date>.csv`。
- **无** `experiments/`、`configs/`、SQLite、`prediction_run_id`、`dataset_id`、`model_id`。

## 2. 新旧模块映射表

| 建议路径 | 现状对应 | 证据 | 处置 |
|---|---|---|---|
| `data/repository.py` | 无（`dataset.py` 内直连 DuckDB） | `dataset.py:218/301/1046` | 不新建 |
| `data/universe.py` | `stock/selection/universe.py` | 317 行，部分 PIT | 已存在，位置保留 |
| `data/calendar.py` | `dataset.load_calendar` / `calendar_fingerprint` | `dataset.py:283/1117` | 先不拆 |
| `data/splits.py` | `export.resolve_split` + `make_samples` + `PanelWindows` 校验 | `export.py:175`、`dataset.py:849`、`panel.py:231-251` | 部分存在，暂不拆 |
| `data/pit.py` | `align_one` + 审计 + `validate` 遗嘱断言 + `pit_status` | `dataset.py:639/1920`、`universe.py:274` | 只补能力标记 |
| `data/features.py` | `build_features` | `dataset.py:713` | 随 `dataset.py` 拆 |
| `data/feature_schema.py` | `stock/data/schema.py`（仅命名/对齐） | 136 行 | 增强，不新建 |
| `data/targets.py` | `schema.TARGET` + 内联公式 + `label_valid` | `schema.py:21`、`dataset.py:752-756/911` | **采纳（小）** |
| `data/windows.py` | `make_samples` + `PanelWindows` | `dataset.py:849`、`panel.py:159` | 已覆盖 |
| `data/preprocessing.py` | `fit_preprocessing/transform` | `dataset.py:944/1022` | 先不拆 |
| `data/builder.py` | `prepare_one` | `dataset.py:1223` | 已覆盖 |
| `data/exporter.py` | `stock/data/export.py` | 577 行 | 已存在 |
| `data/panel_dataset.py` | `stock/data/panel.py` | 350 行 | 已存在 |
| `data/artifacts.py` | `stock/training/artifacts.py` + manifest | 55 行、`dataset.py:1454` | 增强 |
| `data/validation.py` | `validate_dataset` | `dataset.py:1641-1990` | 已覆盖 |
| `models/base.py` | `stock/adapters/base.py` | 37 行 | 已存在 |
| `models/registry.py` | `stock/adapters/registry.py` | 52 行 | 已存在 |
| `models/timexer/{network,embedding,adapter}.py` | `models/TimeXer.py` + `adapters/timexer.py` | embedding 在 `timexer.py:19-36` | 已拆分，network 不迁 |
| `models/patchtst/*` | 无 | — | 等第二模型 |
| `models/baselines/*` | 测试内基准列 | `trainer.py:104-108` | 不新建 |
| `training/runner.py` | `stock/training/runner.py` | 616 行 | 已存在 |
| `training/trainer.py` | `stock/training/trainer.py` | 119 行 | 已存在 |
| `training/refit.py` | `runner.train_refit` | `runner.py:242-523` | 已存在 |
| `training/losses.py` | 内联加权 MSE | `trainer.py:49-50` | 不拆 |
| `training/sampling.py` | `PanelWindows.stock_weights` | `panel.py:281-288` | 不拆 |
| `training/checkpoints.py` | 内联 save/load | `runner.py:141-152/408-416` | 暂不拆 |
| `training/experiment.py` | 无（out-dir 约定） | — | 只补 ID/manifest |
| `training/tuning.py` | 无 | — | 不采纳 |
| `inference/latest.py` | `save_latest` | `dataset.py:1041` | 不新建 |
| `inference/predictor.py` | `stock/inference/predictor.py` | 155 行 | 已存在 |
| `inference/contracts.py` | 无统一契约（`note` 混用） | `predictor.py:109-120` | **采纳** |
| `inference/ensemble.py` | 无 | — | 不采纳 |
| `selection/rules.py` | `stock/selection/rules.py` | 33 行 | 已存在 |
| `selection/filters.py` | 可并入 rules | — | 不拆 |
| `selection/portfolio.py` | 无 | — | 与回测一起 |
| `selection/config.py` | argparse | `daily_pipeline.py:60-77` | 不采纳 |
| `evaluation/metrics.py` | 已存在 | 80 行 | 已存在 |
| `evaluation/daily_hit.py` | 已存在 | 151 行 | 已存在 |
| `evaluation/walk_forward.py` | 已存在 | 1088 行 | 已存在 |
| `evaluation/folds.py` | `build_folds` 内嵌 | `walk_forward.py:206` | 不拆 |
| `evaluation/state_store.py` | JSON plan/fold_metrics | `walk_forward.py:1037-1063` | 推迟 |
| `evaluation/compare.py` | `protocol_comparison` 内嵌 | `walk_forward.py:413` | 不拆 |
| `evaluation/report.py` | `save_test_report` | `metrics.py:32-80` | 已存在 |
| `evaluation/backtest.py` | **无** | — | **采纳** |
| `evaluation/attribution.py` | 无 | — | 不采纳 |
| `production/daily_pipeline.py` | 已存在 | 217 行 | 已存在 |
| `production/readiness.py` | 仅打印数据滞后告警 | `daily_pipeline.py:169-176` | 小改 |
| `production/model_registry.py` | checkpoint format 注册表已覆盖 | `adapters/registry.py:35` | 推迟 |
| `production/prediction_store.py` | **无**（CSV 按 target_date 覆盖） | `daily_pipeline.py:207-212` | **采纳** |
| `production/monitoring.py` | 无 | — | 推迟 |
| `production/outcome_evaluator.py` | 无 | — | 推迟 |
| `production/retrain.py` | 无 | — | 推迟 |
| `production/alerts.py` | 无 | — | 不采纳 |

## 3. 逐条判定（2.1–2.9）

### 2.1 数据层

| 建议 | 判定 | 依据 |
|---|---|---|
| `repository.py` | **不采纳** | DuckDB 查询已集中在 `dataset.py`/`export.py`/`universe.py`，都是只读连接；抽象层只会把 2303 行数据引擎重写一遍，且当前无第二数据源需求。 |
| `universe.py` | **采纳（已在 `stock/selection/`）** | 已有 317 行，含 `--as-of` 部分 PIT 与 `pit_status` 元信息（`universe.py:274-296`）。放到 `data/` 反而与"选股资格"语义脱节。 |
| `calendar.py` | **推迟（等拆 `dataset.py`）** | 日历推导 + 市场宽度校准 + 指纹是一体逻辑（`dataset.py:283-406/1117`），单独抽出无收益。 |
| `splits.py` | **推迟（等下一次必须重导数据）** | 边界解析在 `export.py:175-284`，落到样本在 `dataset.py:918-924`，装载校验在 `panel.py:231-251`。合并会改 `export_script_sha256`，现有 data-root 立即不可续跑（`export.py:512-520`）。 |
| `pit.py` | **改造成本低先做（只做能力标记）** | 已有 `align_one` 近似可用时间 + `available<=decision` 审计（`dataset.py:639-703/1920-1923`），manifest 已写明"无实际发布时间、可能修订"（`dataset.py:1496-1503`）。真 PIT 需要历史快照表，库内没有。 |
| `features.py` | **推迟** | `build_features` 与价格/辅助 specs 强耦合，单独抽出意义有限。 |
| `feature_schema.py` | **采纳** | `schema.py` 只有命名函数，没有 schema 版本、列类型、分组、hash；训练侧 `schema.json` 只是列表（`runner.py:217`），schema hash 只在 refit 的 `fit_metadata`（`runner.py:487-489`）。 |
| `targets.py` | **采纳** | 目标常量在 `schema.py:21`，公式内联在 `dataset.py:752-756`，`label_valid` 在 `dataset.py:911-915`，切分在 `918-924`；定义分散，改口径易漏。 |
| `windows.py` | **不采纳** | `make_samples` + `PanelWindows` 已覆盖；移动会破坏 samples.csv 契约。 |
| `preprocessing.py` | **推迟** | fit/freeze/transform 已在 `dataset.py:944-1038`，且被 `infer` 冻结复用；拆分风险大于收益。 |
| `builder.py` / `exporter.py` | **不采纳（已覆盖）** | `prepare_one`（`dataset.py:1223`）+ `stock/data/export.py`。 |
| `panel_dataset.py` | **不采纳（已存在）** | `stock/data/panel.py:159-350`。 |
| `artifacts.py` | **改造成本低先做** | 现有只有环境信息与首个预测时刻；缺实验/模型 ID 与统一 manifest。 |
| `validation.py` | **不采纳（已存在）** | `validate_dataset`（`dataset.py:1641-1990`）已做 9 类检查，且 `--mode validate` 是稳定入口。 |

三条约束：
- (a) **采纳**：在 `targets.py` 显式写清"次日涨跌幅 = 后复权收盘 close(t+1)/close(t)-1，停牌日无标签，标签用下一交易日"，并把"预测标签"与"可执行收益（t+1 开盘/成本）"分开；后者由回测实现。现状只在文档与 manifest 声明（`dataset.py:1516-1518`、`daily_pipeline.py:36-40`），无代码级规范。
- (b) **已满足，无需改**：切分按 target_date（`dataset.py:918-924`），samples.csv 同时有 `input_date/target_date`（`dataset.py:928-933`），`train_end` 之前不需要未来价格才能形成的标签。`PanelWindows` 还会纵深校验（`panel.py:231-251`）。
- (c) **采纳（只做标记）**：`align_one` 用的是库内**当前修订值** + 假设 lag，不是严格 PIT；`infer` 会用当前库重算历史并只报 `history_drift`（`dataset.py:70-75`、`manifest.limitations`）。在 manifest/评估报告里把能力标成 `approximate_pit`，不宣称 PIT。真 PIT **推迟**到有快照表。

### 2.2 模型层

| 建议 | 判定 | 依据 |
|---|---|---|
| `models/base.py` | **不采纳（已有等价实现）** | `stock/adapters/base.py:6-37` 已定义 `model_config/build_model/forward/checkpoint_format`。 |
| `models/registry.py` | **不采纳（已有）** | `stock/adapters/registry.py:9-52`，含按 checkpoint format 反查与旧格式回退。 |
| `models/timexer/{network,embedding,adapter}` | **不采纳新建** | `TypedExogenousEmbedding` 已在 `adapters/timexer.py:19-36`，network 保留上游 `models/TimeXer.py`（fork 兼容性优先）。 |
| `models/patchtst/*`、`models/baselines/*` | **推迟（等第二个模型立项）** | 目前只注册 TimeXer（`adapters/__init__.py:25`）；零/均值/持续性基准已在 `trainer.py:104-108`。 |
| "新增模型只需 5 步且不改三处" | **改造成本低先做** | 适配层做到了，但 `walk_forward.train_command` 写死 `"--model","TimeXer"`（`walk_forward.py:323`），且继续传 `d_model/e_layers/patch_len`（`324-341`）。 |
| 训练循环两层（共享 Trainer / 自带 fit） | **推迟** | 只有一个 torch 模型；引入 `fit()` 分支属无需求抽象。runner 已模型无关地调用 `adapter.forward`。 |

### 2.3 训练层

| 建议 | 判定 | 依据 |
|---|---|---|
| `runner/trainer/refit` | **不采纳（已存在）** | `runner.py` 616 行、`trainer.py` 119 行；refit 在 `runner.py:242-523`；`train_global_timexer.py` 已是 40 行 shim。 |
| `losses.py` | **不采纳** | 唯一损失是股票等权加权 MSE（`trainer.py:49-50`），拆文件只增加跳转。 |
| `sampling.py` | **不采纳** | 采样权重是 `PanelWindows` 的语义一部分（`panel.py:281-288`），拆开会破坏 Dataset 契约。 |
| `checkpoints.py` | **推迟** | 原子写 + 失败即报错已实现（`runner.py:141-152/408-416`）；等需要跨模型加载/迁移时再抽。 |
| `experiment.py` | **改造成本低先做（只做 ID+manifest）** | 现以 `--out-dir` 为实验目录；见 2.7。 |
| `tuning.py` | **不采纳** | 无超参搜索需求，walk-forward 已固定预算。 |

建议中"测试集只做最终评估"**已满足**：`train` 先早停选模型，再从 checkpoint 重载做测试（`runner.py:219-236`）；`refit` 在预算完成后才测试（`runner.py:423-443`）。

### 2.4 推理/选股层

| 建议 | 判定 | 依据 |
|---|---|---|
| `inference/latest.py` | **不采纳** | 最新窗口由数据引擎 `save_latest`（`dataset.py:1041-1062`）产出，推理层不应再管窗口生成。 |
| `inference/predictor.py` | **采纳（已存在）** | `predict`/`score` 已抽出（155 行），checkpoint 格式反查。 |
| `inference/contracts.py` | **采纳** | 现在 `score` 的 `note` 混用 `ok/no_infer_output/bad_shape/non_finite`（`predictor.py:109-120`），生产又往里写 `signal`（`daily_pipeline.py:167`）；输出无 `model_id/dataset_id/prediction_status`。 |
| `inference/ensemble.py` | **不采纳** | 无多模型/集成分工需求。 |
| `inference/artifacts.py` | **推迟** | 与 2.7 重复，合并处理。 |
| `selection/rules.py` | **采纳（已存在）** | `hybrid_signal` 已统一 `predict_daily` 与 `analyze_daily_hit`。 |
| `selection/filters.py` | **不采纳** | 过滤逻辑是规则的一部分，拆成两文件易漂移。 |
| `selection/portfolio.py` | **推迟（与 backtest 同步做）** | 目前没有组合构建/成本/成交价，单独先做是无本之木。 |
| `selection/config.py` | **不采纳** | argparse + 显式参数已是单一来源，再加配置文件会出现两套默认值。 |
| "三处必须同一个选股函数" | **改造成本低先做** | 生产与命中分析**已共用**；但不存在"第三处组合回测"，`walk_forward` 只算回归/IC（`metrics.py:50-60`），不调用选股。需补回测并统一。 |
| floor 作用于预测还是实际、top q% 分母、并列、取整、空组合 | **采纳（写进契约+测试）** | floor 明确作用于**预测值**（`rules.py:12-17`、`daily_pipeline.py:160-164`）；分位分母生产是剔除 stale 后的 `valid`（`daily_pipeline.py:150-152`），`daily_hit` 是当日全部测试行（`daily_hit.py:43-48`），两者不一致；并列由 `sort_values` 稳定排序隐式决定（`rules.py:23-26`），跨调用方顺序不同；空组合允许（打印"建议空仓"）。 |

### 2.5 评估层

| 建议 | 判定 | 依据 |
|---|---|---|
| `metrics.py` | **不采纳（已存在）** | 80 行，含 pooled/stock-equal/日均 RankIC。 |
| `daily_hit.py` | **不采纳（已存在）** | 151 行，逐日命中 + 多 floor 对照。 |
| `walk_forward.py` | **不采纳（已存在）** | 1088 行，含两阶段 refit、DB 指纹、协议对照。 |
| `folds.py` | **不采纳** | `build_folds`（`walk_forward.py:206`）已独立函数，拆文件只改相对导入。 |
| `state_store.py` | **推迟** | walk-forward 目前明确不实现断点恢复（`walk_forward.py:55-60` 文档），用 JSON plan/fold_metrics；等需要长跑续跑时再上 SQLite。 |
| `compare.py` | **不采纳（已存在）** | `protocol_comparison`（`walk_forward.py:413`）负责 A/B 对照。 |
| `backtest.py` | **采纳（真缺口）** | 只有"收盘→收盘"统计命中，没有成交价/成本/组合净值；输入信号无法按 t 收盘成交（`daily_pipeline.py:36-40`、`dataset.py:1516-1518`）。 |
| `report.py` | **不采纳（已存在）** | `save_test_report`（`metrics.py:32-80`）。 |
| `attribution.py` | **不采纳** | 无归因需求，且需先有可执行回测。 |
| 多模型比较共享股票池/日期/标签/选股协议 | **推迟（等第二模型+回测）** | 现在只比 refit vs selection，同一模型（`walk_forward.py:413`）。 |

### 2.6 生产层

| 建议 | 判定 | 依据 |
|---|---|---|
| `daily_pipeline.py` | **不采纳（已存在）** | 217 行，含逐股 infer、失败继续、打分、选股、CSV。 |
| `readiness.py` | **改造成本低先做** | 只有打印式新鲜度告警（`daily_pipeline.py:169-176`），不阻断；没有 schema/checkpoint/股票池覆盖的硬门槛。 |
| `model_registry.py` | **推迟** | checkpoint format 注册已能防混用（`registry.py:35-48`）；版本晋升/回滚等需求未出现。 |
| `prediction_store.py` | **采纳** | 预测按 `predict_<target_date>.csv` 写（`daily_pipeline.py:207-212`），同日重跑直接覆盖；失败股票只打印前 5 条（`139-140`），不入文件。 |
| `monitoring.py` | **推迟** | 无监控告警渠道与 SLA。 |
| `outcome_evaluator.py` | **推迟（store 之后一步）** | 有了落库与标签，才能对账。 |
| `retrain.py` | **推迟** | 当前人工调度 walk-forward/训练脚本。 |
| `alerts.py` | **不采纳** | 无需求。 |
| 幂等 / `--as-of` 补跑 / 失败隔离 / 回滚 / 预测不可覆盖 | **采纳（小步）** | `dataset infer` 已支持 `--as-of`（`dataset.py:1064-1078`），但生产 CLI 没暴露（`daily_pipeline.py:60-77`）；失败隔离有（`run_infer:100-113`、`score:106-120`），持久化与不可覆盖没有。 |

### 2.7 产物与版本

| 建议 | 判定 | 依据 |
|---|---|---|
| 四类 ID | **改造成本低先做** | 现在只有目录名与哈希；无 `dataset_id/experiment_id/model_id/prediction_run_id`。 |
| `experiments/<id>/` 固定目录 | **不采纳** | 现有 `--out-dir` 已被 walk-forward 与脚本广泛依赖（`walk_forward.py:822-1010`）；改目录会同时改训练、评估、文档与历史产物路径。改为在 out-dir 内写 `model_manifest.json`。 |
| manifest 记录模型/数据/特征/标签/窗口/预处理/代码版本 | **改造成本低先做** | checkpoint 已有 `model_config/schema/export_config/preprocessing`，`refit` 已有 `schema_hash`（`runner.py:487-489`）；缺 selection 阶段的 schema hash、feature schema 版本、target 定义、dataset_id。 |
| 推理契约不匹配报错不静默 | **不采纳（已满足）** | `predictor.py:39-45` 预处理逐字典相等、`55-58` 维度/有限值；`adapter_for_checkpoint` 未知格式直接报错（`registry.py:35-48`）。 |

### 2.8 测试

| 建议 | 判定 | 依据 |
|---|---|---|
| `test_pit.py` | **推迟** | 真 PIT 未实现；现在能测的只有 `available<=decision` 与 manifest 能力字段，`validate_dataset` 已覆盖前者（`dataset.py:1920-1923`）。 |
| `test_targets.py` | **改造成本低先做** | 目标公式与 `label_valid` 目前无独立测试。 |
| `test_splits.py` | **改造成本低先做** | 边界/标签日期规则值得钉死。 |
| `test_preprocessing.py` | **推迟** | `validate_dataset` 已做扰动不变性（`dataset.py:1641-1990`），可后续迁成 pytest。 |
| `test_windows.py` | **推迟** | 窗口位置已由 `PanelWindows` 装载期硬校验（`panel.py:215-222`）。 |
| `test_model_contract.py` | **改造成本低先做** | 已有 `tests/test_embed_and_attention.py`，但只测层/注意力；缺 adapter 输出的 `[B,1,1]` 与 checkpoint 重载。 |
| `test_selection.py` | **采纳（高优先）** | 并列、floor 单位、分母口径无测试；本地 `verify_equiv.py` 只覆盖"新旧实现等价"，且**不在远端仓库**。 |
| `test_checkpoint_reload.py` | **改造成本低先做** | 现有第 13 行声明有 checkpoint save/load 回归，但只针对模型层。 |
| `test_daily_pipeline.py` | **推迟（等 prediction_store）** | 无 store 时只能测 CSV 覆盖行为。 |
| `test_walk_forward.py` | **推迟** | 1088 行调度器做单测成本高，先用小样本端到端冒烟。 |

**现状测试盘点**：远端只有 `tests/test_embed_and_attention.py`（模型层 + TimeXer ex_mask/save-load）。`verify_equiv.py` 位于本地 `/Users/lwy/Documents/Trae-CN/default/tsl_refactor/verify_equiv.py`，远端没有；它已覆盖：`hybrid_signal` vs 旧内联（30 组随机含并列）、逐日信号、`align_features/ex_mask` 20 组随机列子集、`write_json` 逐字节、`export.py` 五个函数源码与 `orig/` 逐字一致。**这是最值得先搬进仓库的回归资产。**

### 2.9 实施顺序与"优先新增"

| 建议 | 判定 | 依据 |
|---|---|---|
| 原四阶段顺序（先第二模型，再拆数据引擎） | **不采纳（按本文第 6 节重排）** | 契约与生产持久化才是当前风险；第二模型没有立项；拆 `dataset.py` 会改导出脚本哈希，必须等重导窗口。 |
| "最终优先新增 8 个文件" | **改造采纳** | `models/base`、`models/registry`、`training/runner`、`inference/predictor`、`selection/rules` **已存在**；真正要新增的是 `feature_schema`（增强）、`targets`、`inference/contracts`、`production/prediction_store`、`evaluation/backtest`。 |
| 保留脚本入口、抽共享模块、先契约后模型最后拆大文件 | **采纳** | 与现状的 shim 策略一致（`train_global_timexer.py:19-37` 等）。 |

## 4. 冲突点与推荐目录方案

### 4.1 冲突点

1. **`train_global_timexer.py` 已不是单体**：40 行转发（`train_global_timexer.py:19-37`），主体在 `stock/training/runner.py`。建议里的"拆 train→trainer、predict→inference"已完成，重复拆只会制造第三套入口。
2. **目录名冲突**：根目录已有上游 `models/`（`models/TimeXer.py` 等）、`data_provider/`、`dataset/`（导出产物）。建议的 `data/`、`models/`、`evaluation/` 会与它们混淆；`stock/data` 与根 `dataset/` 也必须分清（前者代码，后者产物）。
3. **选股"三处重复"不成立**：生产与命中分析已共用 `hybrid_signal`；真正缺的是第三层"组合回测"，不是第三个重复实现。
4. **`targets.yaml` 会制造第二事实源**：现有 `preprocessing.json/manifest.json/checkpoint` 已承载目标与配置；再加 yaml 需要同步。应把目标规范写成 Python 常量 + 产物字段，而不是新配置文件。
5. **测试资产错位**：`verify_equiv.py` 只在本地，远端仓库没有；建议的 10 个测试文件如果照搬，会先于功能存在，维护成本高于收益。

### 4.2 推荐目录方案（唯一方案，不并行）

保留 `stock/` 包为唯一业务层，根目录只保留兼容 shim 与 `dataset.py`（过渡期）：

```
stock/
  data/          schema.py  panel.py  export.py
                 + targets.py  feature_schema 相关（后续可拆 splits/preprocessing/pit/features/validation）
  adapters/      base.py  registry.py  timexer.py        （后续加 patchtst.py）
  training/      runner.py  trainer.py  artifacts.py
  inference/     predictor.py  contracts.py
  selection/     rules.py  universe.py                   （后续加 portfolio.py）
  evaluation/   metrics.py  daily_hit.py  walk_forward.py  backtest.py
  production/   daily_pipeline.py  prediction_store.py  readiness.py
根目录：dataset.py（数据引擎，长期可迁到 stock/data/builder.py 并留 shim）
        train_stock.py / export_global.py / walk_forward.py / predict_daily.py /
        select_universe.py / analyze_daily_hit.py（薄转发，不变）
产物：dataset/（数据导出）、checkpoints/<run>/（模型产物）、predictions/（CSV 视图）
       predictions.db（SQLite，事实源）
```

不采纳根目录 `data/`、`models/`、`evaluation/`。

## 5. 真缺口与最小修法（按"是否会导致错误结论"排序）

| # | 缺口 | 现状证据 | 为什么会导致错误结论 | 最小修法 |
|---|---|---|---|---|
| 1 | **无可执行回测**，统计命中被当成收益 | 只有 `y_true` 收盘→收盘与命中率（`metrics.py:15-29`、`daily_hit.py:43-89`）；信号 t+1 08:30 才形成，无法按 t 收盘成交（`daily_pipeline.py:36-40`） | 会把"方向准确/IC"误读为"可赚钱"；未计成本、开盘滑点、停牌不可成交 | `stock/evaluation/backtest.py`：复用 `hybrid_signal`，t+1 开盘（或可成交价）买入、持有期与成本显式，输出净值/换手/回撤；同时在报告里并列"收盘→收盘"与"开盘→开盘" |
| 2 | **预测可被覆盖、失败不留痕** | `predictions/predict_<target_date>.csv`（`daily_pipeline.py:207-212`），同日重跑覆盖；失败只打印前 5 条（`139-140`），`score` 的 `note` 不落库 | 无法复盘"当时到底预测了什么"；不同 checkpoint 的结果被静默替换；后验评估失真 | SQLite `prediction_runs/predictions`，唯一键 `(run_id,symbol,target_date)`，`status` 枚举（ok/stale/no_infer/non_finite/signal），插入即冻结；CSV 变为导出视图 |
| 3 | **特征 schema 无版本/hash（训练侧）** | `schema.json` 只是列表（`runner.py:217`）；hash 只在 refit 的 `fit_metadata`（`runner.py:487-489`） | 跨 run/跨模型比较时列序或列集变化不会被显式拒绝，可能把不同 schema 的指标放一起 | `stock/data/schema.py` 增 `FEATURE_SCHEMA_VERSION`，由 `fit_preprocessing` 生成 `feature_schema.json`（name/order/dtype/group/hash），写入每个导出目录、checkpoint、`test_metrics.json`；推理/评估比对 hash，不匹配即报错 |
| 4 | **标签定义散落** | `TARGET` 在 `schema.py:21`，公式在 `dataset.py:752-756`，`label_valid` 在 `911-915`，文档另写（`dataset.py:27-31`） | 改收益率口径/复权/停牌处理时，训练、评估、回测三处口径可能漂移 | 新增 `stock/data/targets.py`：`TARGET_SPEC`（复权、价格口径、目标交易日、停牌规则）+ `label_valid()`；`dataset.py` 引用它并写进 `preprocessing.json`/`manifest.json`；回测读同一规范 |
| 5 | **选股边界未定义（并列/分母/单位）** | `rules.py:23-26` 稳定排序隐式定并列；生产分母是剔除 stale 的 `valid`（`daily_pipeline.py:150-164`），命中分析是当日全量（`daily_hit.py:43-48`）；floor 单位由调用方换算（`rules.py:17`） | 同一 q% 在分析/生产选出不同股票数；并列跨边界时结果不稳定；单位错了会差 100 倍 | `hybrid_signal` 显式 `tie-break=(pred desc, symbol asc)`、`floor_decimal` 命名、`eligible_universe` 入参；生产与命中分析传入同一 eligible 集合；加单测覆盖并列/空集/单位 |
| 6 | **窗口重叠导致样本不独立** | `make_samples` 从 `length-1` 到 `len(index)-2` 逐行推进（`dataset.py:849-942`），相邻样本共享 `seq_len-1` 行 | MAE/RMSE/方向准确率的样本数被放大，置信区间偏窄；早停的 val 曲线也相关 | 报告同时给"按 target_date 聚合"的日均指标与窗口数；文档明确 `n` 是重叠窗口数；重要结论用按日 block bootstrap |
| 7 | **PIT 限制未传播到评估层** | `select_universe` 已标 `snapshot_criteria`（`universe.py:274-296`）；walk-forward 只固定当前 universe 或部分 PIT（`walk_forward.py` 文档"已知局限"） | 容易把 walk-forward 结果称为"严格 PIT 回测" | 把 `pit_status` 写进 `walk_forward_plan.json`/`oos_summary.json`，报告标题标"approximate_pit"；真 PIT 等历史快照表 |
| 8 | **walk-forward 写死 TimeXer** | `walk_forward.py:323`，并继续传 `d_model/e_layers/patch_len`（`324-341`） | 接入第二模型时不是"只加适配器"，而是改调度器；参数也会错传 | 加 `--model`（默认 TimeXer），模型专属参数由适配器声明/校验；`train_command` 只传通用参数 |
| 9 | **`8_train_timexer.sh` 已坏** | 第 34 行调用不存在的 `train_timexer.py`（仓库无此文件） | 新人按脚本跑必失败，误以为环境问题 | 改为 `train_stock.py --mode train`，或删除该脚本并更新文档 |
| 10 | **仓库内缺数据/选股回归测试** | 仅 `tests/test_embed_and_attention.py`；`verify_equiv.py` 只在本地 | 拆分/改口径后无自动回归，等价性靠人工 | 把 `verify_equiv.py` 收进 `tests/test_equiv.py`，再补 `test_selection/test_targets/test_splits/test_checkpoint_reload` |

> 澄清：建议中的"test 指标在标准化尺度"**不是缺口**。`test_predictions` 已把预测还原到原始收益率尺度（`trainer.py:88-111`），测试指标用的就是原始 `y_true`。

## 6. 分阶段落地顺序（渐进，不照搬原四阶段）

### P0 卫生与回归基线（不改数据/模型逻辑）
- 改：`8_train_timexer.sh` 指向 `train_stock.py`；根目录/README 的入口说明。
- 增：`tests/test_equiv.py`（迁入本地 `verify_equiv.py`）。
- 验证：`.venv/bin/python -m pytest tests` + `verify_equiv` 全 PASS。
- **不做**：不改 `dataset.py`、不动导出产物、不重命名 `stock/`。

### P1 契约与口径（核心，仍不拆大文件）
- 增：`stock/data/targets.py`；`stock/inference/contracts.py`；`stock/data/schema.py` 增 schema 版本与 hash 函数。
- 改：`dataset.py` 的 `fit_preprocessing/prepare_one` 写出 `feature_schema.json` 并登记 `target_spec`；`runner.py` 把 schema hash 写入 selection 阶段与 checkpoint；`predictor.py` 输出统一字段与 `status`；`rules.py` 明确并列/分母/单位；`walk_forward.py` 加 `--model`。
- 验证：任选一只股票用**新 data-root** 重导，比较 `feature_matrix.npy/dates.npy/samples.csv` 与旧目录逐值一致（新增文件除外）；用旧 `test_predictions.csv` 重放 `daily_hit`，命中数逐日一致；`test_equiv` 仍 PASS。
- **不做**：不改选股语义（除显式 tie-break）、不改模型、不拆 `dataset.py`、不上第二模型。

### P2 生产持久化
- 增：`stock/production/prediction_store.py`（SQLite，manifest 表 + predictions 表）、`stock/production/readiness.py`。
- 改：`daily_pipeline.py` 暴露 `--as-of`（透传 `dataset.py --as-of`），写入 `prediction_run_id/model_id/dataset_id`，失败股票与 stale 全部落库，CSV 保留为视图；同 target_date 二次运行新增 run 而不覆盖。
- 验证：对同一 target_date 连跑两次（同/不同 checkpoint），确认两条 run 均存在、旧 run 可读；断网/坏 checkpoint 时 readiness 阻断而非产出半份结果。
- **不做**：不做自动重训、告警、监控面板。

### P3 可执行回测与事后评估
- 增：`stock/evaluation/backtest.py`、`production/outcome_evaluator.py`。
- 改：`daily_hit`/回测都调用同一 `hybrid_signal` 与同一 eligible 集合；报告并列收盘→收盘 / 开盘→开盘、成本与换手。
- 验证：历史 `test_predictions.csv` 上，回测的信号集合与 `daily_hit` 信号集合按 `(symbol,target_date)` 完全一致；给定零成本、开盘=收盘时净值与收盘口径一致（自洽性检查）。
- **不做**：不做归因、不做多策略组合优化。

### P4 第二模型接入
- 增：`stock/adapters/patchtst.py` 或线性/树模型适配器；`stock/adapters/__init__.py` 注册一行。
- 改：`walk_forward` 的模型参数改由适配器提供；`train_command` 只传通用参数；比较协议固定股票池/日期/标签/选股。
- 验证：两个模型在同一导出目录、同一折上产出相同 schema 与日期覆盖；`test_metrics` 可并排比较；`test_model_contract` 通过。
- **不做**：不做 ensemble、不做超参搜索。

### P5（可选）拆数据引擎
- 仅在 `dataset.py` 变更频率高时做：按 `targets/splits → preprocessing/windows → pit/features → builder/exporter/validation` 顺序抽到 `stock/data/`，每步一次小样本重导，比较 `feature_matrix` 哈希、`samples.csv`、`preprocessing.json` 统计量。
- 验证：每次迁移后跑 `validate_dataset` + 目录级哈希对照；`export_script_sha256` 变化必须显式换 data-root。
- **不做**：不搬 `export.py` 已稳定的五个函数；不做 DuckDB repository。

## 7. 风险与反例（照做会出问题的地方）

1. **先拆 `dataset.py` 会立刻废掉现有 data-root**
   导出配置登记了 `dataset_script_sha256/export_script_sha256`（`stock/data/export.py:512-513`），复用时严格比对，不一致直接报"导出配置已变化，请使用新的 data-root"（`export.py:516-520`）。同时本地 `verify_equiv.py` 用 `inspect.getsource` 逐字比对 `export.py` 五个函数与 `orig/`（`verify_equiv.py:307-315`）。反例：为"整洁"把 `resolve_split` 挪走，会让所有历史导出不可续跑、等价性校验失败。正确做法：先版本号升级 + 明确换目录 + 重导，再迁移。

2. **"真 PIT"在现有库上做不出来，硬做会得到虚假自信**
   `align_one` 的 `available` 只是 `observed + lag_days + release_hour`（`dataset.py:664-668`），库内没有 `published_at/version`；`infer` 用当前库重算历史，只能报 `history_drift`（`dataset.py:70-75`）；`select_universe` 明确市值/简称/存续名单是快照（`universe.py:282-295`）。反例：把 `merge_asof` 改名为 PIT 并在报告中宣称"严格 point-in-time"，会让回测结论无法审计。正确做法：manifest 标 `approximate_pit`，把限制传播到 `oos_summary.json`。

3. **过早把模型专属参数塞进通用 registry，会让 walk-forward 更难定位**
   现状 `train_command` 写死 `--model TimeXer` 且无条件传 `d_model/e_layers/patch_len`（`walk_forward.py:323-341`）。反例：只把 `--model` 改成变量就宣称"支持多模型"，第二模型会收到不认识的参数或用了错误默认值，错误发生在子进程里、日志被折叠。正确做法：适配器声明可接受的参数集，调度器按模型组装命令并校验。

4. **新增 SQLite 但保留覆盖式 CSV，会形成两个事实源**
   `predict_daily` 用 `predict_<target_date>.csv` 覆盖写（`daily_pipeline.py:207-212`），且 `note` 同时承载 `ok/no_infer_output/bad_shape/non_finite/signal`（`predictor.py:109-120`、`daily_pipeline.py:167`）。反例：回测读 CSV、监控读 SQLite，两者对"某股票当天是否发信号"给出不同答案。正确做法：SQLite 为唯一事实源，CSV 只做导出；`status` 与 `signal` 拆成两列。

5. **选股规则"顺手统一单位"可能改掉 100 倍口径**
   `hybrid_signal` 要求 `floor` 与 `pred` 同单位（`rules.py:12-17`），生产传百分数 `0.5`（`daily_pipeline.py:164`），命中分析传 `0.5/100`（`daily_hit.py:47`）。反例：重构时把 `daily_hit` 的 `/100` 删掉，会把 floor 变成 0.5%（本意 0.5 个百分点）之外的 50%，信号数骤变而无人察觉。正确做法：参数改名 `floor_decimal`/`floor_pct` 并在单测中钉死两组等价关系。

6. **把 10 个测试文件一次性铺开，会先制造维护负担**
   远端目前只有 `tests/test_embed_and_attention.py`；`verify_equiv.py` 尚未入库。反例：先写 `test_pit.py`/`test_daily_pipeline.py` 去测尚不存在的功能，测试只能 mock，随后功能一变全部重写。正确做法：先把已有的 `verify_equiv` 收进仓库，再按 P1/P2 的实际改动补 4 个核心测试。

---

**评审结论落点**：现状已经是"契约先行"的半成品（适配器、schema 单一事实源、PanelWindows、统一选股、refit 协议、manifest 限制声明都在），下一阶段真正的增量不是"再拆文件"，而是**把预测写成不可覆盖的事实、把评估落到可执行收益、把 schema/标签版本钉死、把模型硬编码去掉**。`dataset.py` 的拆分应排在最后，且必须伴随一次显式重导。

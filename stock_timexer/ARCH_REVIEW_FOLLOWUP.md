# 股票预测平台化重构建议 · 续评（P0–P5 落地后的剩余条目）

> 基线：`stock_timexer/ARCH_REVIEW.md`（评审对象 HEAD `72ea19d`）。
> 本续评对象：该报告中处置为「推迟 / 小改 / 待第二模型 / 待 store」的条目，以及真缺口表中未被 P0–P5 覆盖的项。
> 当前 HEAD `02e9e7d`（P0 `41fb7ee` → P1 `000885b` → P2 `cf303e5` → P3 `cf303e5` 前序 → P4 `f71203f` → P5 `001c5ce` → 清理 `02e9e7d`），全部测试 104 passed + 1 skipped。
> 只读核实，本轮未改任何功能代码。

## 1. 原评审「必须做」与真缺口 #1–#5/#8–#10：全部落地，不再属于剩余

| 原条目 | 落地证据 |
|---|---|
| 预测落库（SQLite、不可覆盖） | `stock/production/prediction_store.py`（`prediction_runs`/`predictions`，主键 `(run_id,symbol,target_date)`，触发器禁止原地更新）；CSV 仅视图 |
| 可执行回测共用同一选股 | `stock/evaluation/backtest.py`（`executable_returns`：t+1 开盘入场、成本/滑点/换手/组合净值；`compute_signals` 复用 `hybrid_signal`） |
| 特征 schema 版本+hash | `stock/data/schema.py` + 导出目录 `feature_schema.json`；checkpoint 登记且双模型逐字节一致（P4 实测） |
| 标签定义收敛 | `stock/data/targets.py`（`target_series/target_spec/label_valid`），写入 `preprocessing.json`/`manifest.json` |
| walk_forward 写死 TimeXer | P4：`--model` + 适配器 `cli_params` 透传，`--model timesnet` 1 折实测通过 |
| readiness 阻断 / `--as-of` / 失败落库 | P2：`stock/production/readiness.py`，阻断实测退出码 1 |
| 四类 ID | `prediction_runs.model_id/dataset_id/run_id`（checkpoint 指纹）；checkpoint 含 `dataset_id/feature_schema_hash`。`experiment_id` 未单独建字段——由 `run_id` 与 out-dir 目录名承载，**实质满足、形式差异**（原报告也反对 `experiments/<id>/` 目录制） |
| `inference/contracts.py` | 已存在（`PredictionStatus` 枚举，`status` 与 `signal` 拆列落库） |
| 回归测试资产入库 | `tests/` 已有 13 个测试文件（含 `test_equiv.py`/`test_selection.py`/`test_targets.py`/`test_splits.py`/`test_prediction_store.py`/`test_readiness.py`/`test_backtest.py`/`test_outcome_evaluator.py`/`test_dataset_equiv.py`/`test_timesnet_adapter.py`） |
| `8_train_timexer.sh` 已坏 | 随老脚本清理删除（`a944176`），USAGE 有替代命令 |
| 数据层拆分（原 P5 可选项） | `001c5ce`：`dataset.py` 薄入口 + `stock/data/{common,calendar,splits,sources,pit,features,preprocessing,windows,builder,validation}.py`，A/B 逐字节等价（`test_dataset_equiv.py` 固化） |
| `selection/portfolio.py`（原「与回测同步做」） | backtest 已含 `_portfolio/_equity_records/_turnover`（组合净值/换手），**关闭，不单独建** |

## 2. 剩余条目逐条续评

| # | 原条目（原判定） | 现状核实 | 续判定 |
|---|---|---|---|
| R1 | 真缺口 #6：窗口重叠样本不独立（原：报告按日聚合 + 文档声明 + block bootstrap） | 部分完成：`metrics.py` 已有 `mean_daily_rank_ic`/`rank_ic_days`（RankIC 按日聚合）；但 MAE/RMSE/方向准确率的 `n` 仍是**重叠窗口数**，`test_metrics.json` 与 USAGE 未声明该口径；block bootstrap 未做 | **小步现在做**：`test_metrics.json` 加口径说明字段（如 `n_semantics: "overlapping_windows"`）+ USAGE 一句话声明；bootstrap 属研究需求，**继续推迟** |
| R2 | 真缺口 #7：PIT 限制未传播到评估层（原：`pit_status` 写进 `walk_forward_plan.json`/`oos_summary.json`，报告标 `approximate_pit`） | **未做**：`universe.py:274` 已产出 `pit_status`，但 `walk_forward.py` 的 plan/oos_summary（`walk_forward.py:806/1076`）未登记 | **建议现在做**（成本低、防错误解读）：折调度时把 universe 的 `pit_status` 原样写进 plan 与 summary；测试并入 `test_pit` 的断言（见 R8） |
| R3 | `training/checkpoints.py` 抽取（原：推迟，等跨模型加载） | 原子写已实现；两模型经同一 `adapter_for_checkpoint` 反查加载，未出现迁移/跨格式需求 | **继续推迟**（条件未出现） |
| R4 | `evaluation/state_store.py`：walk-forward 断点恢复（原：推迟，等长跑续跑需求） | 仍为 JSON plan/fold_metrics，无续跑需求 | **继续推迟** |
| R5 | `production/model_registry.py`（原：推迟） | checkpoint format 反查（`registry.py`）已防两模型混用；版本晋升/回滚需求未出现 | **继续推迟** |
| R6 | `production/monitoring.py` / `retrain.py` / `alerts.py`（原：推迟/不采纳） | 无监控渠道与自动重训需求；`readiness` 阻断 + `outcome_evaluator` 对账已覆盖人工复盘路径 | **维持原判**（推迟/不采纳） |
| R7 | `inference/ensemble.py` / `tuning.py` / `attribution.py` / `data/repository.py`（原：不采纳） | 条件未变（单库、无集成/归因/超参需求） | **维持不采纳** |
| R8 | `test_pit.py` / `test_preprocessing.py` / `test_windows.py`（原：推迟） | 能力已被其他资产覆盖：`validate_dataset` 22 项含 `available<=decision` 与窗口索引一致性；`test_dataset_equiv.py` 固化扰动重拟合不变性。单独建文件属重复 | **关闭，不单独建**；仅当 R2 落地时补 `pit_status` 传播断言（可放 `test_readiness.py` 或 walk-forward 冒烟） |
| R9 | `test_daily_pipeline.py`（原：推迟，等 prediction_store） | **前置条件已满足**：store 已存在且有 `test_prediction_store.py`，但「pipeline 端到端（infer→score→signal→落库→阻断）」无独立测试，目前靠 USAGE 实测记录 | **升级为值得做**（中优先）：用临时 data-root + 1 股 checkpoint 冒烟，断言 run 落库、CSV 视图、readiness 阻断路径 |
| R10 | `test_walk_forward.py`（原：推迟，用小样本端到端冒烟） | 已有 1 折双模型实测记录（P4），但未固化成测试（需真实 DB，同 `test_dataset_equiv.py` 的 slow/skipif 形态） | **可选**：按 `test_dataset_equiv.py` 模式固化 1 折冒烟（依赖 DB，默认 skip）；优先级低于 R9 |
| R11 | 多模型比较共享协议（原：推迟，等第二模型+回测） | 条件已满足：P4 已验证同 data-root 双模型 schema/哈希/日期覆盖一致；`test_metrics` 可并排 | **条件已备齐，属使用而非开发**：跑两组 `walk_forward --model TimeXer/TimesNet` 同折对照即可；不需要新代码 |
| R12 | `inference/artifacts.py` / `model_manifest.json` 独立文件（原：与 2.7 合并） | 信息已由 checkpoint 元数据 + `prediction_runs` 承载 | **关闭**（避免第二事实源） |
| R13 | `data/pit.py` 能力标记 `approximate_pit`（原：改造成本低先做） | manifest `limitations` 已声明"无真实发布时间、可能修订"（`dataset.py` 头注/`builder.py` 迁出后语义不变），但**没有机器可读的能力字段**；与 R2 同根 | **并入 R2**：在 manifest 加 `"pit_capability": "approximate_pit"` 一并传播 |

## 3. 结论与建议顺序

1. **建议现在做（一次小改可合并提交）**：R2+R13（`pit_status`/`pit_capability` 传播到 plan/oos_summary/manifest）+ R1（指标口径声明字段）。两者都是"防止结论被误读"的低成本项，与原评审「必须做」同性质。
2. **其次**：R9（`test_daily_pipeline.py` 端到端冒烟），R10 可选。
3. **继续搁置**：R3/R4/R5/R6（需求未出现），R7（无需求），R8/R12（已被现有资产覆盖/关闭）。
4. R11 不需要开发，直接用现成入口跑对照即可。

> 未验证项说明：本续评为只读评审，R1–R13 的"现状核实"基于源码检查与 P0–P5 的既有实测记录（见各 commit message 与 USAGE 第 6 节），未运行新的长任务（如全池 walk-forward 双模型对照）。

## 4. 落地记录（本节随实施 commit 更新）

| 条目 | 落地方式 | 实测 |
|---|---|---|
| R1 口径声明 | `metrics.py` `save_test_report` 的 summary 顶层新增 `_semantics`（`n_semantics: "overlapping_windows"` + 两条 note），随 `test_metrics.json` / `oos_summary.json` 一并落盘；USAGE §2.1 补口径说明 | timesnet 1 折冒烟：`oos_summary.json` 含 `_semantics.n_semantics=overlapping_windows` |
| R2 oos_summary PIT 标记 | `walk_forward._pit_summary()`：按 `--pit-universe` 与否写 `universe_mode`/`pit_capability`（`approximate_pit` / `conditional_history`）+ criteria/note；非 refit 写入 `summary["_pit"]`，refit 路径顶层加 `"pit"` 键。`walk_forward_plan.json` 原有 pit_universe 段未动 | 固定池分支端到端实测通过（timesnet 1 折冒烟，`conditional_history`）；`--pit-universe` 分支单元级断言通过（`_pit_summary(True)`），全池端到端因筛选导出耗时过长未跑（**未验证**，原因：全池 777+ 只逐折导出预计 40 分钟以上） |
| R13 manifest 能力字段 | `builder.py` manifest 新增机器可读 `"pit_capability": {"level": "approximate_pit", ...}`（`limitations` 之外的单字段版） | 1 折导出 manifest 实测含该键 |
| R9 管线端到端测试 | 新增 `tests/test_daily_pipeline.py`（slow+skipif 真库）：临时 data-root + 契约完整随机权重 checkpoint；场景 1 发布（run 落库 published、predictions 行、CSV 视图）；场景 2 `--max-lag-days 5` 触发 readiness「数据滞后」阻断（blocked 留档、exit 1、无 CSV） | 1 passed（16s） |

其余条目维持第 3 节判定不变（R3–R7 推迟/不采纳，R8/R10–R12 维持原判）。

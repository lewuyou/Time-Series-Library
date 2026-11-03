# 日历特征设计说明（P6 新约束版）

本文档记录日历特征进入 TimeXer 的方式、默认行为、被拒绝的建议，
以及对上游 `models/TimeXer.py` 的逐条改动清单。
结论先行：**日历信息默认走官方 `x_mark` 通道（`calendar_mode=time_marks`），
预测头默认拼接目标日星期 one-hot（`head_calendar_dim=5`），
两者可用独立开关关闭做 A/B。**

---

## 1. 一个流传很广的推论，在本仓库不成立

建议文本里有一句："官方 TimeXer 对 `x_mark` 列做共享投影，交换周一、
周二列输出不变，所以把星期 one-hot 放进 `x_mark` 有语义漂移风险。"

**这个推论在本仓库不成立**，原因有二：

1. **每个外生 token 有身份编码。** 本仓库的
   `TypedExogenousEmbedding`（`stock/adapters/timexer.py`）为每个外生
   变量位置维护独立的可学习 `feature_id` 向量
   （`[1, n_ex_tokens, d_model]`），与值嵌入相加后才进入注意力。
   交换两列，两个 token 的身份编码跟着交换，注意力输出一般会变化——
   不存在"共享投影导致列可任意置换"的对称性。
2. **列顺序被 schema 固定。** `stock/data/schema.py` 的
   `feature_schema_hash` 把显式列清单（含顺序）哈希进导出目录与
   checkpoint；装载时列集合或顺序不一致直接报错。列序不是运行时
   可以漂移的东西。

**但它指出的真问题是真的：输入日 / 目标日错位一天。**
本任务预测的是 t+1 个交易日收益，若第 t 行日历值取自输入日 t，
模型看到的星期/节假日信息与标签日差一天。本实现统一用
**目标日语义**解决：第 t 行日历值 = t+1 交易日的日历值
（`stock/data/features.py: target_calendar_features`），
`x_mark`、旗标列、预测头 one-hot 三处全部遵守同一语义。

## 2. 日历信息放在哪里：官方通道优先，列做补充

三种进入方式，互斥开关 `--calendar-mode {time_marks, columns, none}`：

| 模式 | 星期/月内位置/年内位置 | 5 个旗标列 | 说明 |
|---|---|---|---|
| `time_marks`（默认） | `x_mark_enc` 3 个 token | 留在输入列 | 官方通道 |
| `columns` | 全部留在输入列 | 留在输入列 | 模型无关通道 |
| `none` | 不进入模型 | 不进入模型 | 基线 |

为什么默认 `time_marks`：

- `x_mark_enc` 是 TimeXer 论文处理时间协变量的**官方通道**
  （`DataEmbedding_inverted` 把 mark 与变量并排投影成 token 参与
  全局注意力），走官方路径最少引入自创语义。
- 标记值沿用官方日频归一化写法（对应官方 `hour/23 - 0.5` 的日频版）：
  `dow/6 - 0.5`、`(day-1)/30 - 0.5`、`(dayofyear-1)/365 - 0.5`。

为什么 5 个旗标列（`target_gap_days_log`、`target_is_gap_ge4`、
`target_is_pre_long_holiday`、`target_is_month_start_3`、
`target_is_month_end_3`）**不**塞进 `x_mark` 而留在普通外生列：

- `x_mark` 的语义是"时间位置标记"，3 个 token 已覆盖官方三特征；
  把派生旗标混进去会稀释通道语义，也让 `columns` 模式的对照不干净。
- 列是**模型无关**通道：TimesNet、未来的树模型都能直接消费；
  `x_mark` 是 TimeXer 专属。旗标列放列里，换模型不用重做特征。
- 原 `target_dow_*` 5 个 one-hot 列**已删除**：星期信息已由
  `x_mark` 的 DayOfWeek token（或 `columns` 模式下的
  `target_mark_dow`）与预测头 one-hot 承载，不做双份编码。

日历列的三条硬规则：

- **全部目标日语义、纯交易日历推导。** 节假日后下一交易日来自
  `holiday_calendar` 表，禁止"星期加一"；月初/月末 3 个交易日按
  交易日历计数，禁止"日期 ≥ 28"近似。
- **不参与标准化、不参与 `ex_mask`、无 `__missing` 伴随列。**
  日历永远存在，标记列已归一到 [-0.5, 0.5]、旗标列为 0/1；
  标准化只会破坏语义。训练期恒为常数的日历列也不允许被丢弃
  （补零对齐会把"缺列"和"常 0 列"混淆）。
- **不同日历规格的导出目录禁止混用。** `load_catalog` 发现
  股票间日历列集合不一致直接报错，不做补零对齐。

## 3. 预测头日历向量（`head_calendar_dim=5`，默认开启）

`models/TimeXer.py` 的 `FlattenHead` 前，把目标日星期 one-hot（5 维）
`torch.cat` 到展平表示，再过 `Linear`。`--no-head-calendar` 关闭
（`head_calendar_dim=0`，结构与上游逐键一致）。

**与 `x_mark` 的 DayOfWeek token 信息重叠是有意的：**

- token 路径里，星期信息要经过 60 天序列的 patch 化、线性投影、
  两层注意力压缩，是有损的；
- head 路径是 5 维显式输入，直接进最后一层线性分类/回归，
  对"星期几"这种与标签日严格对齐的离散信息是无损捷径。
- 两条路径可分别关闭（`--calendar-mode none` 关 token 侧、
  `--no-head-calendar` 关 head 侧），A/B 可分离各自贡献。

## 4. 对 `gap_days` 的预期管理

`target_gap_days_log` / `target_is_gap_ge4` 描述的是"目标日距上一
交易日隔了多少个自然日"。长假后首个交易日主要影响的是**波动**
（信息积压一次性释放、成交放大），而不是**带符号收益的方向**。
因此不应指望这组特征提升日度 RankIC（带符号收益率的秩相关）；
它的合理收益场景是波动/幅度类目标或风控过滤。评估时若 RankIC
无变化，属预期内，不是实现错误。

## 5. 被拒绝的建议及理由

| 建议 | 结论 | 理由 |
|---|---|---|
| 季末旗标（`target_is_quarter_end` 等） | 拒绝 | A 股季末效应主要由资金面/考核驱动，与交易日历位置弱相关；样本内约 2400 个独立交易日，季末每季 1~3 天，独立事件 < 40 次，估计不出稳定效应 |
| 衍生品到期日（期权/期指交割） | 拒绝 | 需要额外维护交割日历数据源；月度到期日与月初/月末旗标高度共线，增量信息存疑，复杂度不划算 |
| 月份 one-hot / 年内季节性（12 维、365 维） | 拒绝 | 约 2400 个独立交易日摊到 12 个月 / 365 天，每格样本极少；`target_mark_doy` 已提供平滑的年内位置，离散月份 one-hot 是"准日期 ID"，过拟合风险高于收益 |

## 6. `x_mark_dec`：本任务不接入

TimeXer 的预测头直接作用在编码器输出上，解码器侧输入不参与
`forecast` 路径，`x_mark_dec` 对本模型**完全无用**，不为对齐官方
接口而造数据。

但以下模型**确实会用** `x_mark_dec`（它们把时间标记喂给解码器）：
`Informer`、`Autoformer`、`Non-stationary Transformer`
（以及 `Transformer`/`Reformer` 等 decoder-encoder 家族）。
将来若把日历特征接入这些模型，需要单独实现解码器侧目标日标记
（解码窗口对应的目标日语义要与编码器侧一致），不能直接复用
`x_mark_enc`。

## 7. 开关、默认值与产物契约

| 开关 | 默认 | 语义 |
|---|---|---|
| `--calendar-mode` | `time_marks` | `time_marks`：3 个官方时间标记进 `x_mark_enc`；`columns`：全部日历列留在输入列；`none`：关闭全部日历信息 |
| `--head-calendar` / `--no-head-calendar` | 开启 | 预测头拼接目标日星期 one-hot（5 维） |
| `--calendar-features` | `all` | 导出期日历列集合（`all`/`none`/逗号子集），同一导出目录内所有股票必须一致 |

- checkpoint 格式号：`global_timexer_typed_v1`（head 关）/
  `global_timexer_typed_v2`（head 开）；`model_config` 记录
  `calendar_mode`、`calendar_columns`、`calendar_mark_columns`、
  `calendar_mark_dim`、`head_calendar_dim`，推理端按 checkpoint
  复现，不由命令行覆盖。
- schema 版本：`stock_features_v3`（删除 `target_dow_*`，
  新增 3 标记列 + 5 旗标列）。

## 8. `models/TimeXer.py` 相对上游的逐条改动清单

上游基准：thuml/Time-Series-Library commit `9d582c5` 的
`models/TimeXer.py`；只读参考实现见
`tests/reference_timexer_upstream.py`（同提交内联上游
`layers/Embed.py`、`layers/SelfAttention_Family.py`、
`utils/masking.py`）。**开关全关（`head_calendar_dim=0`、
`ex_mask=None`、`mask=None`）时数值逐位等价**，由
`tests/test_calendar_features.py::test_upstream_equivalence_*`
守护（实测最大绝对差 0.0）。

行号以当前仓库 `models/TimeXer.py` 为准：

| # | 行号 | 改动 | 生效条件 |
|---|---|---|---|
| 1 | L17-27 | `FlattenHead.forward` 增加可选参数 `extra`：展平后把 `[B, extra_dim]` 向量广播拼接到线性层输入 | 仅当 `extra is not None`（即 `head_calendar_dim>0` 且传入 `head_calendar`）；`extra=None` 时与原实现逐位一致 |
| 2 | L131-139 | `Model.__init__` 增加断言：`seq_len` 必须被 `patch_len` 整除 | 恒生效（构造期）。上游 `unfold(step=P)` 会静默丢弃末尾 `seq_len%P` 个**最新**点，对预测是最坏损失，改为快速失败；整除时行为不变 |
| 3 | L168-174 | 新增 `self.head_calendar_dim`（`getattr(configs, "head_calendar_dim", 0)`，缺省 0）；`FlattenHead` 输入维改为 `head_nf + head_calendar_dim` | `head_calendar_dim=0`（默认/旧 checkpoint）时结构与上游完全相同，state_dict 逐键一致 |
| 4 | L177-190 | 新增 `_check_ex_mask` 静态方法：校验 `ex_mask` 为 `[B, n_ex_tokens]` bool | 仅当传入 `ex_mask` |
| 5 | L194-195, L208-214 | `forecast` 增加 `ex_mask`/`head_calendar` 参数；`ex_mask` 转为 `[B,1,1,S]` 的 `cross_mask` 传给编码器 | `ex_mask=None` 时 `cross_mask=None`，与上游 `encoder(en_embed, ex_embed)` 相同 |
| 6 | L220-234 | `forecast` 头部：`head_calendar_dim>0` 时校验并经 `extra` 拼接头输入；`=0` 时拒绝误传的 `head_calendar` | `head_calendar_dim=0` 且未传 `head_calendar` 时走原 `self.head(enc_out)` 分支 |
| 7 | L245-249, L262-268 | `forecast_multi` 增加 `ex_mask` 支持（同 #4/#5）；`head_calendar_dim>0` 时报错（多变量任务不支持头部日历） | 本仓库只用 `features=MS`，`forecast_multi` 路径默认不触发 |
| 8 | L284-305 | `forward` 增加 `ex_mask`/`head_calendar` 参数；显式拒绝 `mask is not None`（上游静默忽略该参数，本仓库无 imputation 任务，防止误用） | `mask=None`（上游调用约定）时不触发；其余分支透传 #5/#6 |

未改动的部分：`EnEmbedding`、`ExEmbedding`（含 `glb_token` 初始化）、
`Encoder`/`EncoderLayer` 的数值路径、`forecast` 的
归一化/反归一化、patch 化与重整形状——与上游逐字一致。
仓库 `layers/Embed.py`、`layers/SelfAttention_Family.py` 相对上游
仅有校验/参数重排类改动（数值路径由乘法交换律保证一致），
由 `tests/test_embed_and_attention.py` 与上游等价测试联合守护。

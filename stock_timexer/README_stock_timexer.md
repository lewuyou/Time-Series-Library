# Time-Series-Library（TimeXer）理解报告 + 股票预测落地方案

仓库：`/root/projects/Time-Series-Library`（HEAD `deee9cb` "增加参数能否被整除校验"，工作区干净，remote 为 `lewuyou/Time-Series-Library`，是 thuml/Time-Series-Library 的 fork，含本地股票方向改造）。

## 0. 总体结论

这套库是一个「统一的多任务时序预测实验框架」：`run.py` 只做参数解析和实验编排，把 `task_name` 映射到 `exp/` 里的 Exp 类；`exp/` 负责训练循环（train/vali/test）、早停、checkpoint 保存和指标落盘；`data_provider/` 负责按 70/10/20 固定切分构造滑窗样本；`models/` 下每个文件是一个可插拔模型，只实现 `forward(x_enc, x_mark_enc, x_dec, x_mark_dec)` 接口。

TimeXer 在这套框架里跑 `long_term_forecast` 任务时是一个 **encoder-only** 模型：目标序列（endogenous）被切成 patch + 1 个全局 token 做 self-attention，外生变量（exogenous）每个变量整段压成 1 个 token，由全局 token 通过 cross-attention 聚合。**它完全忽略 decoder 输入（`label_len`、`x_dec`、`x_mark_dec`、`dec_in` 对它无效）**，输出头是一个 Flatten + Linear 直接映射到 `pred_len`。

评估口径要特别注意：**框架原生不是 walk-forward / rolling，而是「一次固定 70/10/20 切分 + 测试段内 stride=1 滑窗多步预测，模型只训一次、不重训」**。run.py 全部参数里没有任何 rolling/incremental 开关。该仓库作者显然意识到了这一点，在框架**外**另行写了 `walk_forward.py`（1088 行，支持 expanding 默认 / `--train-days` 滚动窗口 / `--refit` 两阶段重拟合）来补 walk-forward 评估。

改股票预测的最小路径：`--data custom` + 一份 `date + 特征列 + 目标列` 的 CSV + `--features MS`（多变量输入、单目标输出），`enc_in` = 非 date 列总数，日频用 `--freq b`（business day），`seq_len` 必须被 `patch_len` 整除。dry-run 一条命令即可验证链路（见 §5.6）。指标建议补方向准确率与累计收益，在 `exp/exp_long_term_forecasting.py` 的 test 落盘处加（见 §6）。

## 1. 项目结构导览与一次 run.py 调用的执行链路

### 1.1 各目录职责

| 路径 | 职责 | 关键证据 |
|---|---|---|
| `run.py`（283 行） | argparse 参数解析、设备选择、`task_name → Exp 类` 映射、生成实验名 `setting`、调用 `exp.train()` / `exp.test()` | `run.py:182-202`（任务映射）、`run.py:237/240`（train/test 调用）、`run.py:219-222`（setting 命名格式） |
| `exp/` | 每种任务一个 Exp 类：训练循环、vali/test 评估、早停、checkpoint 与结果保存 | `exp/exp_long_term_forecasting.py:76-166`（train）、`42-74`（vali）、`168-268`（test）；`exp/exp_basic.py:20`（`model_dict` 懒加载所有模型） |
| `models/TimeXer.py` | TimeXer 模型本体（详见 §2） | 全文 271 行 |
| `data_provider/data_factory.py` | `data_dict` 注册数据集类；`data_provider(args, flag)` 统一构造 Dataset + DataLoader（train/val shuffle、test 不 shuffle） | `data_factory.py:15-23`（data_dict）、`21`（`shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True`）、`78-86`（forecast 分支构造） |
| `data_provider/data_loader.py` | 各数据集 Dataset：ETTh/ETTm（内置硬编码切分）与 `Dataset_Custom`（70/10/20 切分 + 标准化 + 滑窗 `__getitem__`） | `data_loader.py:63-66`（ETT 边界）、`272-276`（custom 切分）、`313-324`（滑窗取样）、`326`（窗口数） |
| `utils/` | `tools.py`：早停 + lr 调度；`metrics.py`：MAE/MSE/RMSE/MAPE/MSPE；`timefeatures.py`：按 `--freq` 生成时间特征 | `utils/tools.py:32-61`（EarlyStopping）、`12-29`（adjust_learning_rate）、`utils/timefeatures.py:99-115`（freq→特征映射） |
| `scripts/long_term_forecast/` | 各模型 × 各数据集的标准基准脚本（多变量 `features M`，多模型对比） | 如 `ETT_script/TimeXer_ETTh1.sh`（`--features M`） |
| `scripts/exogenous_forecast/` | TimeXer 论文的「外生变量预测」基准：固定 `--features MS`，按数据集分目录（详见 §4） | `ECL/TimeXer.sh` 等 8 个脚本 |

### 1.2 执行链路（一次 `run.py --is_training 1`）

1. **参数解析**：`run.py:19-160` argparse；`run.py:163-177` 选设备（cuda/mps/cpu）。
2. **选 Exp 类**：`run.py:182-202`，`long_term_forecast → Exp_Long_Term_Forecast`（未知 task_name 也兜底到它）。
3. **构造实验**：`run.py:210-235`，`Exp(args)` 在 `exp_basic.py:20-23` 里懒加载 `models/<model>.py` 并 `.to(device)`；setting 字符串含 task/model/data/features/超参，用作 checkpoint 与结果目录名。
4. **数据构建**：`exp.train()` 里 `data_provider(args, 'train'/'val'/'test')`（`exp_long_term_forecasting.py:77-79`）→ `Dataset_Custom.__read_data__`（`data_loader.py:253-299`）：读 CSV → 目标列移到最后（`:269-271`）→ 70/10/20 切分（`:272-276`）→ **StandardScaler 只在 train 段 fit**（`:287-288`）→ 时间特征（`:290-299`）。
5. **训练/验证**：`exp_long_term_forecasting.py:96-161`：逐 epoch，构造 `dec_inp = [batch_y[:, :label_len, :], zeros]`（`:111-112`），前向 → MSE loss（`:38`）→ 反传；每 epoch 调 `vali()`（`:151`）按验证损失早停/存 `checkpoint.pth`（`utils/tools.py:57-60`）；注意 `:152` **每个 epoch 也顺带打印 test loss（仅监控，不用于选择）**。
6. **测试与指标保存**：训练完 load 最优权重（`:163-164`）→ `exp.test(setting)`（`:168-268`）：test loader 逐 batch 前向（`--inverse` 时反标准化，`:204-209`）→ 拼接全部窗口 → `metric()`（`:255`，`utils/metrics.py:34-40`）→ 结果写入 `result_long_term_forecast.txt`（`:257-261`）、`./results/<setting>/{metrics,pred,true}.npy`（`:264-266`）、可视化 PDF 到 `./test_results/<setting>/`（`:176-178`，每 20 个 batch 存一张）。

## 2. TimeXer 模型细节（models/TimeXer.py）

### 2.1 输入拆分与三种组件

以 MS 模式（`forecast()`，`TimeXer.py:183-217`）为例：

- **内生序列（endogenous）**：只取最后一列 `x_enc[:, :, -1]`（`:193`，目标列已被 Dataset 移到最后）→ `EnEmbedding`（`:24-49`）：
  - `unfold(size=patch_len, step=patch_len)` 切成 `patch_num = seq_len // patch_len` 个不重叠 patch（`:41`）；
  - 每个 patch 经 `Linear(patch_len, d_model)` + 位置编码（`:44`）；
  - 拼上 1 个**可学习全局 token** `glb_token`（`:31`，`Parameter(1, n_vars, 1, d_model)`），token 数 = `patch_num + 1`。
- **外生变量（exogenous）**：`x_enc[:, :, :-1]` + 时间标记 `x_mark_enc` → `DataEmbedding_inverted`（`:194`）：**每个外生变量把整段 `seq_len` 序列（连同拼接的时间特征）经同一个 `Linear(seq_len, d_model)` 压成 1 个 token**（`layers/Embed.py:180-211`，`:209` 显式 `torch.cat([x, x_mark.permute(0,2,1)], 1)` 把时间特征当作额外"变量 token"拼进去）。
- **Attention 结构**（`EncoderLayer.forward`，`:85-112`）：
  - self-attention：patch tokens + 全局 token 之间（`:86-90`）；
  - cross-attention：**以全局 token 为 query，对外生 tokens 做 attention**（`:93-101`，`x_glb = x[:, -1, :]`）；
  - 之后 FFN（conv1d 1×1 实现，`d_model→d_ff→d_model`，`:73-74`）。
- **输出头**：`FlattenHead`（`:9-22`）把 `[bs, n_vars, d_model*(patch_num+1)]` 摊平后 `Linear(→pred_len)`，即输出 `[bs, n_vars, pred_len]`。

M 模式（`forecast_multi`，`:219-252`）不同点：**全部 `enc_in` 个通道都作为 endogenous patch**（`:229`），同时全部通道又作为 ex tokens（`:230`），反标准化用各通道自己的统计量（`:247-249`）；而 MS 反标准化只用目标列的统计量（`:211-215`，`stdev[:, 0, -1:]`）。

本 fork 的本地增强（非上游原版）：`ex_mask` 支持屏蔽缺失外生变量 token（`:167-182, 196-202`）；`forecast_multi` 分支（上游 MS-only）；`seq_len % patch_len` 整除 assert（`:124-133`）。

### 2.2 参数含义与推荐取值

| 参数 | 实际含义（TimeXer 语境） | 证据 | 推荐取值（股票日频起步） |
|---|---|---|---|
| `enc_in` | 输入变量总数（含目标列）。**M 模式必须精确等于非 date 列数**（否则 `glb_token` 形状不匹配运行时报错）；**MS 模式下在本实现中实际上不被校验、几乎无效**（详见 §2.3 坑 1） | `TimeXer.py:134,136,163` | = CSV 非 date 列数 |
| `dec_in` | **对 TimeXer 完全无效**：模型签名接收 `x_dec, x_mark_dec` 但从未使用（`:183,219,254-270` 全文无引用） | 同左 | 随便填（脚本里都填成 enc_in） |
| `c_out` | **对 TimeXer 实际无效**：`FlattenHead` 虽然接收 `enc_in` 作 `n_vars`，但 forward 里 `n_vars` 未参与输出宽度，输出宽度由实际输入 `n_vars` 决定（MS=1、M=enc_in） | `TimeXer.py:9-22, 163` | 填 1（MS）或 enc_in（M），不影响结果 |
| `seq_len` | 回看窗口长度；**必须被 patch_len 整除，否则构造时 assert 直接失败**（不整除时会静默丢弃最新的 `seq_len % patch_len` 个点——对预测最致命的一段） | `TimeXer.py:124-133`、`run.py:141-143` | 日频 60（约 3 个月）或 120 |
| `label_len` | decoder 起始 token 长度，**TimeXer 忽略 decoder 输入，故对 TimeXer 无影响**；只影响 `dec_inp` 构造（`exp:111-112`）和 `__getitem__` 里 `seq_y` 的形状 | `exp_long_term_forecasting.py:111-112`；`TimeXer.py:183-270` | 0 或默认 48 均可 |
| `pred_len` | 预测步数 | — | 1（次日收益）/ 5（一周） |
| `n_heads` | 注意力头数；`d_model` 必须能被它整除（`AttentionLayer` 内 `d_model // n_heads`） | `layers/SelfAttention_Family.py` | 4 或 8 |
| `e_layers` | encoder 层数（每层 = self-attn + cross-attn + FFN） | `TimeXer.py:152-162` | 1-2（小数据防过拟合；EPF 脚本用到 3） |
| `d_model` / `d_ff` | 隐藏维度 / FFN 维度 | `TimeXer.py:71-79` | d_model 64-128，d_ff 128-256（脚本区间 128-512） |
| `factor` | **无效参数**：本仓库 `FullAttention.__init__` 根本不存储 `factor`，缩放固定为 `1/sqrt(E)`。脚本里的 `--factor 3` 是从 Informer/ProbAttention 时代遗留的习惯 | `layers/SelfAttention_Family.py:186-196`（`self.scale = scale`，无 `self.factor`） | 忽略 |
| `patch_len` | 内生 patch 长度；`patch_num = seq_len // patch_len` | `TimeXer.py:41`、`run.py:141-143` | seq_len=60 → 15（=每周5日×3）或 10/20 |
| `use_norm` | 1=窗口内标准化（Non-stationary Transformers 式：对 `x_enc` 按时间维减均值除标准差，预测后反变换），**不是全数据集标准化** | `TimeXer.py:184-191, 211-215` | 1（股价非平稳，强烈建议开） |
| `dropout` / `activation` | 常规含义 | `:78-79` | 0.1 / gelu |
| `use_amp` / `lradj` / `learning_rate` / `patience` | 混合精度 / lr 调度（type1=每 epoch 减半）/ 早停耐心 | `exp:93-94,141-147,161`；`tools.py:12-29` | lr 1e-4，patience 3 |

### 2.3 「外生变量数量」相关的坑

1. **MS 模式下 `enc_in` 配错不报错**：ex tokens 数量 = `enc_in - 1` 个变量 token + 时间特征 token（`freq=b/d` → 3 个，`h` → 4 个，`t` → 5 个，见 `utils/timefeatures.py:104-111`）。`DataEmbedding_inverted` 对变量数完全无感（每个变量共享同一个 `Linear(seq_len, d_model)`），`FlattenHead` 的 `n_vars` 又不参与计算 → **MS 下 enc_in 是"哑参数"，唯一影响是 checkpoint 命名和 head 的一个未用形参**。想保证正确只能自己数列。M 模式下配错会运行时崩溃（`glb_token` 与实际通道数 cat 不上）。
2. **时间特征会被当成外生 token**：`Embed.py:209` 把 `x_mark` 拼进变量维，所以换 `--freq` 会改变外生 token 数量，但不改任何需要声明的形状参数——静默变化。
3. **目标列位置**：Dataset 会把 `--target` 列移到最后一列（`data_loader.py:269-271`），而 `TimeXer.py:193` 硬编码取 `x_enc[:,:,-1]`、`exp:62-64` 用 `f_dim=-1` 切最后一通道做损失/评估。**CSV 原始列顺序无所谓，但 `--target` 必须是真实存在的列名**。
4. **`--features S`**：Dataset 只保留目标列（`data_loader.py:282-283`），TimeXer 走与 MS 相同的 `forecast()` 分支（`TimeXer.py:268`），此时外生 token 只剩时间特征。可用，但脚本从未这么用。

## 3. 关键问题：训练/验证/测试的评估方式（附代码证据）

**结论：三种都不是"滚动/扩展重训"。原生框架是「一次固定 70/10/20 时序切分 + 模型只训一次 + val/test 各自段内 stride=1 滑窗、每个窗口一次性多步预测」，窗口彼此重叠，不存在任何 rolling/incremental 开关。**

证据链：

**(a) 固定一次性切分** — `data_provider/data_loader.py`（Dataset_Custom）：

```python
# data_loader.py:272-276
num_train = int(len(df_raw) * 0.7)
num_test = int(len(df_raw) * 0.2)
num_vali = len(df_raw) - num_train - num_test
border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
border2s = [num_train, num_train + num_vali, len(df_raw)]
```

- train `[0, 0.7N)`、val `[0.7N - seq_len, 0.8N)`、test `[0.8N - seq_len, N)`。val/test 的 `border1` 前移 `seq_len`，意味着**验证/测试窗口允许回看前一段的最后 seq_len 行作为输入**（只是历史输入，不含前段标签，无标签泄漏，但跨段）。
- 标准化统计量只来自 train 段（`data_loader.py:287-288`：`train_data = df_data[border1s[0]:border2s[0]]; self.scaler.fit(train_data.values)`）。
- ETT 数据集是硬编码边界：`data_loader.py:63-66`（`12*30*24` 个月训练、4 个月验证、4 个月测试）。

**(b) 段内滑窗、步长 1、窗口重叠** — 同文件：

```python
# data_loader.py:313-324 (__getitem__)
s_begin = index; s_end = s_begin + self.seq_len
r_begin = s_end - self.label_len; r_end = r_begin + self.label_len + self.pred_len
seq_x = self.data_x[s_begin:s_end]; seq_y = self.data_y[r_begin:r_end]
# data_loader.py:326 (__len__)
return len(self.data_x) - self.seq_len - self.pred_len + 1
```

每个样本是 `seq_len → pred_len` 的一次性多步预测；test 段内每个可对齐位置都评一次（`pred_len > 1` 时相邻窗口的预测区间互相重叠）。

**(c) 只训一次，不重训、不滚动** — `exp/exp_long_term_forecasting.py`：

- `train()`（`:76-166`）：单个模型、单个 train 段，逐 epoch 训练；`vali()`（`:42-74`）每 epoch 评估一次仅用于早停和选 checkpoint（`:156-158`，`utils/tools.py:32-61`）；训练结束后加载最优权重（`:163-164`）。
- `test()`（`:168-268`）：对**同一个**训好的模型，把 test 段所有滑窗预测一遍后一次性算指标（`:255`）、存 `pred.npy/true.npy`（`:264-266`）。测试循环内没有任何"推进到下一折就重新训练"的逻辑。

**(d) 无 rolling 开关**：`run.py:19-160` 的完整 argparse 已核对，不存在 rolling / walk-forward / incremental / refit 类参数；`scripts/` 下所有脚本也没有。**框架原生不支持 walk-forward。**

**(e) 框架外的补充**（本仓库作者自己补的，不在 run.py 链路里）：

- `walk_forward.py`（1088 行，模块 docstring `:1-80`，参数 `:514-577`）：一个 walk-forward 调度器。协议明确写着：「测试块互不重叠，滚动步长 = `--test-days`(默认20)；**expanding（默认）**：训练目标日为起点至本折 train-end 的全部历史；**rolling（`--train-days N`）**：限制为最近 N 个交易日」；`--refit` 实现两阶段（选择 + 重拟合）协议；`--val-days 60`。
- `train_global_timexer.py`（1450 行）：绕过 run.py 的独立训练器（自带 Dataset 与训练循环，`d_model=64, e_layers=2, n_heads=4` 等小模型默认值，`:1382-1391`）。
- 这佐证了原生框架确实缺少滚动评估，且已有现成的 walk-forward 实现可以复用。

**含义**：如果直接用 run.py 评估股票策略，得到的是「一次切分、单模型、重叠窗口」的指标；要论文/实盘意义上的滚动样本外评估，必须用 `walk_forward.py` 这类外置调度器（已存在）或自己包一层。

## 4. scripts/exogenous_forecast 逐脚本说明

所有脚本固定 `--model TimeXer --features MS --task_name long_term_forecast --itr 1`，逐数据集跑 4 个 pred_len（96/192/336/720，EPF 除外）。与 `long_term_forecast/` 的差别：**exogenous 目录全部用 `features MS`（多变量进、单目标出），并给出 TimeXer 专属超参（d_model/d_ff/e_layers/batch_size 按数据集调）；long_term_forecast 里的 `TimeXer_ETTh1.sh` 等用的是 `--features M`（多变量同预测），用于多模型统一基准对比。**

| 脚本 | 命令要点 | 在做什么 |
|---|---|---|
| `ETTh1/TimeXer.sh` | `--data ETTh1 --features MS --seq_len 96 --label_len 48 --e_layers 2 --factor 3 --enc_in/dec_in/c_out 7 --d_model 512/128...` | ETTh1（7 列电力温度）上以 OT 列为目标，pred_len 96/192/336/720 四组；注意 `--data ETTh1` 走硬编码月度切分而非 70/10/20 |
| `ETTh2/TimeXer.sh` | 同上，`e_layers 1-2`，`d_model 128-256` | ETTh2 四组 |
| `ETTm1/TimeXer.sh`、`ETTm2/TimeXer.sh` | `--data ETTm1/2 --e_layers 1 --d_model 128-512` | 分钟级 ETT 四组 |
| `ECL/TimeXer.sh` | `--data custom --features MS --enc_in/dec_in/c_out 321 --e_layers 1(720 时 3) --batch_size 4-32` | 电价 321 列，目标列 OT；`--data custom` → 70/10/20 切分 |
| `Traffic/TimeXer.sh` | `--data custom --enc_in 862 --e_layers 1 --d_layers 1 --d_model 512 --batch_size 4` | 交通 862 列，四组 pred_len |
| `Weather/TimeXer.sh` | `--data custom --enc_in 21 --e_layers 1 --d_model 128` | 气象 21 列，四组 |
| `EPF/TimeXer.sh` | `--data custom --enc_in 3 --c_out 1 --seq_len 168 --patch_len 24 --pred_len 24 --e_layers 1-3 --d_model 512 --d_ff 512` | 电价预测（NP/PJM/BE/FR/DE 五个市场）：`enc_in 3`（两外生 + 目标）、`c_out 1` 是全目录唯一正确的 c_out 写法；168h 回看 → 24h 预测，`patch_len 24`=一天一个 patch。**这是股票最接近的模板** |

注意：这些脚本里大量出现 `c_out=7/321/862`（与 MS 实际输出 1 通道不符），因 §2.3 坑 1-2 不报错也不生效，属于脚本自身的陈旧配置。

## 5. 改成股票预测：具体操作方案

### 5.1 数据格式与加载配置

`Dataset_Custom`（`data_loader.py:223-328`）对 CSV 的要求：

- 首列 `date`（`pd.to_datetime` 可解析，如 `2024-01-05`）；
- 其余列 = 任意顺序的特征列 + 一列目标（`--target` 指定其列名，代码会移到最后一列，`data_loader.py:269-271`）；
- 不允许有缺失日期以外的 NaN（无填补逻辑）；停牌/假日建议直接不写行（配合 `--freq b`）。

示例：

```csv
date,open,high,low,volume,turnover,pe,index_close,close
2020-01-02,1150.0,1160.0,1145.0,30000,3.4e9,32.1,3096.0,1152.3
2020-01-03,...
```

配置：`--data custom --root_path ./dataset/stock/ --data_path 600519.csv --features MS --target close --freq b`。

`--freq` 证据：`run.py:46-49`（help 明确列出 `d:daily, b:business days`）；`utils/timefeatures.py:104-105`（`d` 与 `b` 都映射到 [DayOfWeek, DayOfMonth, DayOfYear] 3 个特征）；分钟频用 `t`（5 个特征，`:107-111`）。日频建议 `b`。

### 5.2 单目标 + 多外生（`--features MS`）vs 多变量同预测（`--features M`）

- **MS（推荐主方案）**：CSV 列 = 目标股行情 + 技术指标 + 指数/同业/宏观等外生。模型只把目标列当 endogenous patch，其余全部列（含时间特征）当 ex tokens；损失与指标只算目标列（`exp:62-64` 的 `f_dim=-1`）。`enc_in` = 非 date 列总数（例中 9 列 → `enc_in 9`）。
- **M（横截面/多标的）**：所有列都被 patch 成 endogenous 且互相作为 ex tokens（`TimeXer.py:229-230`），输出 `enc_in` 个通道各自的预测。适合「一批股票一起预测」的横截面设置；此时 `enc_in` 必须 = 列数（会真的校验崩溃）。注意 M 模式下目标列与其他股票地位相同，指标会平均所有通道（`exp:62-64` 的 `f_dim=0`）。
- 多只股票做外生：直接把其他股票的 close/return 作为列放进同一张 CSV（按交易日对齐）。**enc_in/dec_in 填列总数，c_out 填什么都行**（见 §2.3），但建议 MS 填 1、M 填 enc_in 以保持 setting 命名可读。

### 5.3 归一化、label_len 与防泄漏

- **两层归一化**：(1) Dataset 层 StandardScaler，**只 fit train 段**（`data_loader.py:287-288`）→ 无全样本泄漏；(2) TimeXer 层 `use_norm=1` 的窗口内标准化（`TimeXer.py:184-191`）→ 缓解股价非平稳。保持 `--use_norm 1`。
- **`label_len` 对 TimeXer 无作用**（模型忽略 decoder 输入，§2.2），`--label_len 0` 即可；`exp:111-112` 构造的 `dec_inp` 只是占位。
- **泄漏注意点**：
  1. 特征工程里的任何 scaler/滚动统计必须在 train 段内 fit（框架只保证 Dataset 内置 StandardScaler 不泄漏，CSV 里预计算的特征不管）；
  2. 复权、成分股名单、财报字段要注意 PIT（point-in-time）——`walk_forward.py` docstring 专门讨论了 `--pit-universe`；
  3. val/test 窗口回看前段最后 `seq_len` 行是设计允许的历史输入（非泄漏）；
  4. 测试段窗口重叠会高估样本独立性——做策略指标时建议只取不重叠的窗口（步长 = pred_len）或用 walk_forward.py；
  5. test 指标默认在**标准化尺度**上计算（`exp:204-209` 只有 `--inverse` 才反变换），跨实验比较没问题，但换算回价格/收益率尺度需 `--inverse` 或自行 `inverse_transform`。

### 5.4 超参起点（TimeXer 必需/易错项）

`--seq_len 60 --patch_len 15`（60/15 整除，约 3 个月回看、每 patch 3 个交易日）；`--pred_len 1`；`--d_model 64 --d_ff 128 --n_heads 4 --e_layers 1 --dropout 0.1`（小样本起步）；`--use_norm 1`；`--label_len 0`。

### 5.5 完整训练命令（dry-run 最小验证）

已写入远程 `stock_timexer/run_stock_timexer.sh`（见 §7）。等价命令：

```bash
cd /root/projects/Time-Series-Library
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id stock_dryrun --model TimeXer \
  --data custom --root_path ./dataset/stock/ --data_path 600519.csv \
  --features MS --target close --freq b \
  --seq_len 60 --label_len 0 --pred_len 1 --patch_len 15 \
  --enc_in 9 --dec_in 9 --c_out 1 \
  --d_model 64 --d_ff 128 --n_heads 4 --e_layers 1 --dropout 0.1 \
  --use_norm 1 --batch_size 32 --learning_rate 0.0001 \
  --train_epochs 1 --itr 1 --patience 3 --num_workers 0 --des dryrun
```

- 最小验证就用 `--train_epochs 1 --itr 1`（`run.py:211-235` 的 itr 循环只控制重复实验次数）；数据量小再补 `--no_use_gpu` 可强制 CPU。
- 跑通后把 `--train_epochs` 提到 10、`--d_model` 提到 128-256。

### 5.6 预期输出

- checkpoint：`./checkpoints/<setting>/checkpoint.pth`（`exp:81-83`，`utils/tools.py:60`）；setting 形如 `long_term_forecast_stock_dryrun_TimeXer_custom_ftMS_sl60_ll0_pl1_dm64_nh4_el1_...`（`run.py:219-222`）。
- 测试指标与预测：`./results/<setting>/metrics.npy`（[mae,mse,rmse,mape,mspe]）、`pred.npy`、`true.npy`（`exp:264-266`）；文本汇总追加到仓库根 `result_long_term_forecast.txt`（`exp:257-261`）。
- 可视化：`./test_results/<setting>/<batch>.pdf`（`exp:176-178, 239-241`）。
- 训练日志：stdout（`-u` 无缓冲），无单独日志文件。

### 5.7 前人已留下的股票资产（本仓库）

`walk_forward.py`（walk-forward 调度）、`export_global.py`/`select_universe.py`（从数据库导出 + 股票池）、`train_global_timexer.py`（独立训练器）、`predict_daily.py`/`analyze_daily_hit.py`（日频推理 + 命中分析）、`8_train_timexer.sh`（注释里写了 pred_len=1、label_len=0、patch_len 取 seq_len 最大因子的经验）。**注意：`8_train_timexer.sh` 调用的 `train_timexer.py` 在仓库根目录不存在（已验证 `ls: No such file or directory`），该脚本目前跑不起来——复用前需先确认文件去向。**

## 6. 评估口径与股票场景建议补充的指标

### 6.1 现有指标的计算范围

`metric()`（`utils/metrics.py:34-40`）输入是 test 段**全部滑窗拼接**后的数组（`exp:228-233`，形状 `[num_windows, pred_len, channels]`，MS 只保留最后一通道，`exp:211-212`），对**整个数组逐元素平均**：

- MSE（`metrics.py:18-19`）、MAE（`:14-15`）、RMSE（`:22-23`）：全局逐点平均，**默认在 StandardScaler 标准化尺度上**（`exp:204-209`，仅 `--inverse` 时反变换）；
- MAPE（`:26-27`）、MSPE（`:30-31`）：`|（true-pred)/true|`，**true 接近 0 时爆炸**——对「收益率」目标基本不可用，对「价格」目标可用但需注意停牌长横盘。

训练中的早停与选型只看 MSE（`exp:38, 69`），指标不参与训练。

### 6.2 股票场景建议补充 + 加在哪

建议补：**方向准确率**（sign(pred diff) vs sign(true diff)）、**Top-K 命中率**、**逐日累计收益 / 夏普**（把每个窗口第一步预测当作次日持仓信号）、**不重叠窗口版指标**。

加码位置（二选一，均已验证行号）：

1. **零侵入**：直接后处理 `./results/<setting>/pred.npy`、`true.npy`（标准化尺度下 sign 与原尺度一致，因为 scaler 是线性正变换；若要价格尺度用 `test_data.inverse_transform`）。
2. **改框架**：在 `exp/exp_long_term_forecasting.py:255`（`mae, mse, ... = metric(preds, trues)` 之后、`f = open(...)` 之前）插入自定义指标并一并写进 `metrics.npy`/txt。示意：

```python
# exp_long_term_forecasting.py:255 之后
mae, mse, rmse, mape, mspe = metric(preds, trues)
# —— 股票指标（preds/trues: [N, pred_len, C]，MS 时 C=1）——
last_true = trues[:, -1, -1]   # 各窗口最后一个时刻的目标（标准化尺度）
last_pred = preds[:, -1, -1]
prev_true = np.concatenate([trues[0, 0, -1:], trues[:-1, -1, -1]])  # 用上一窗口末值近似昨收
dir_acc = np.mean(np.sign(last_pred - prev_true) == np.sign(last_true - prev_true))
```

更严格的方向/收益评估建议直接接 `walk_forward.py` 的 `oos_summary.json`（其 docstring 声明已输出"原始收益率尺度"的样本外总指标）。

## 7. 本次交付物与未验证项

**交付物**（新建目录，未改动任何既有文件）：
- `/root/projects/Time-Series-Library/stock_timexer/README_stock_timexer.md` —— 本报告；
- `/root/projects/Time-Series-Library/stock_timexer/run_stock_timexer.sh` —— §5.5 的 dry-run 训练脚本。

**明确未验证/未执行的事项**：
- 未实际运行任何训练/推理命令（任务约束），输出路径、指标数值均来自代码静态证据，未实测；
- `train_timexer.py` 缺失导致 `8_train_timexer.sh` 不可用（已验证文件不存在，但脚本注释描述的行为未验证）；
- `--factor` 在**上游原版** TimeXer 是否生效未比对（本 fork 中确认无效）；
- HuggingFace 远程数据集回退分支（`data_loader.py` 中 `load_dataset(HUGGINGFACE_REPO, ...)`）未实测。

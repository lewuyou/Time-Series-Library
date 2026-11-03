# -*- coding: utf-8 -*-
"""生成 TimeXer_step_by_step.ipynb：把上游 TimeXer 的每次数据变换展开成独立 cell。"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    })


# ============================================================ 标题
md("""# TimeXer 原版逐步拆解（thuml/Time-Series-Library）

**目的**：把上游 TimeXer 在 ETTh1（`features='MS'`, 96→96）上的**每一次数据变换**拆成独立 cell，
从原始 CSV 一直到 loss 反向，逐步打印张量形状，供逐格运行研究。

**原版文件**（2026-11-03 从 `https://github.com/thuml/Time-Series-Library/tree/main` 下载，一字未改）：

| 文件 | 作用 |
|---|---|
| `models/TimeXer.py` | 模型本体（EnEmbedding / Encoder / FlattenHead / forecast） |
| `layers/Embed.py` | `DataEmbedding_inverted`（外生变元 embedding）、`PositionalEmbedding` |
| `layers/SelfAttention_Family.py` | `FullAttention` / `AttentionLayer` |
| `utils/masking.py`、`utils/timefeatures.py` | 注意力掩码、时间特征 |
| `data_provider/data_loader.py` | `Dataset_ETT_hour`（切分/标准化/滑窗的**原版口径**，本 notebook 手动复现而不 import 它——它顶部 import 了 `datasets/sktime/huggingface_hub` 等重依赖） |
| `run.py`、`data_provider/data_factory.py` | 参数默认值参照 |
| `ETTh1.csv` | 数据（17420 行 × 8 列，来自官方数据仓库 zhouhaoyi/ETDataset） |

**运行方式**：在本目录（`timexer_upstream/`）启动 Jupyter 内核，从上往下逐格运行。
依赖：`torch / pandas / numpy / scikit-learn`；`reformer_pytorch / einops` 缺失时下方 cell 会自动打桩（TimeXer 用不到它们）。

**与多股票面板版（`stock/`）的差异**不在本 notebook 展开范围，最后一节给速查表。""")

# ============================================================ 环境
code("""import sys, types, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# 定位本 notebook 所在目录（timexer_upstream/，应同时包含
# models/TimeXer.py、layers/Embed.py 与 ETTh1.csv）。
# 三者同查是刻意的：若在 Time-Series-Library 仓库根目录启动内核，
# 向上回溯会命中仓库自己的 models/TimeXer.py（本地改过的版本），必须排除。
def _is_upstream_dir(p):
    return ((p / 'models' / 'TimeXer.py').exists()
            and (p / 'layers' / 'Embed.py').exists()
            and (p / 'ETTh1.csv').exists())

ROOT = Path.cwd()
while not _is_upstream_dir(ROOT):
    if ROOT == ROOT.parent:
        raise RuntimeError('找不到 timexer_upstream 目录（需同时有 models/TimeXer.py、'
                           'layers/Embed.py、ETTh1.csv）；请在 timexer_upstream/ 下启动内核')
    ROOT = ROOT.parent
print('上游文件根目录:', ROOT)

# layers/SelfAttention_Family.py 顶部 import 了 reformer_pytorch 与 einops，
# 但 TimeXer 只用 FullAttention / AttentionLayer；缺包时打桩，保证原版文件一行不改就能 import。
try:
    import reformer_pytorch  # noqa: F401
except ImportError:
    _m = types.ModuleType('reformer_pytorch')
    class _Missing:
        def __init__(self, *a, **k):
            raise ImportError('reformer_pytorch 未安装（仅 ReformerLayer 需要，TimeXer 不需要）')
    _m.LSHSelfAttention = _Missing
    sys.modules['reformer_pytorch'] = _m
    print('提示: reformer_pytorch 未安装，已打桩（不影响 TimeXer）')

try:
    import einops  # noqa: F401
except ImportError:
    _m = types.ModuleType('einops')
    def _missing(*a, **k):
        raise ImportError('einops 未安装（仅 ProbAttention/Reformer 等路径需要）')
    _m.rearrange, _m.repeat = _missing, _missing
    sys.modules['einops'] = _m
    print('提示: einops 未安装，已打桩（不影响 TimeXer）')

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from math import sqrt

sys.path.insert(0, str(ROOT))
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.TimeXer import Model as TimeXer
import utils.timefeatures as tf

torch.manual_seed(42); np.random.seed(42)
print('torch', torch.__version__, '| pandas', pd.__version__, '| numpy', np.__version__)""")

md("""---
# 一、喂模型之前：数据侧（`data_provider/data_loader.py` 的 Dataset_ETT_hour 口径）

数据形态总览：**一个数据集 = 一条时间轴上的单张表**。
`ETTh1.csv` 是 `T×8`（1 列 date + 7 个数据列），`features='M'/'MS'` 时取后 7 列为 `df_data`，
全流程只有这一条序列，所有窗口都是它的切片。""")

# ============================================================ 读数据
code("""df_raw = pd.read_csv(ROOT / 'ETTh1.csv')
print('df_raw:', df_raw.shape)      # (17420, 8) = 1 列 date + 7 列数据
print('列名:', list(df_raw.columns))
df_raw.head(3)""")

md("""## 第 1 步：features 列选择（`data_loader.py:68-72`）

`features='M'`（多变量预测多变量）→ 取全部 7 列；`'MS'`（多变量预测单变量）→ 同样取 7 列输入，
但**目标是末列 OT**（模型侧再拆）。`'S'` → 只取 target 1 列。""")

code("""features, target = 'MS', 'OT'   # 本 notebook 用 MS；'M' 的对照见最后一节
cols_data = df_raw.columns[1:]     # 丢掉 date 列
df_data = df_raw[cols_data]
print('df_data:', df_data.shape)   # (17420, 7)
print('列顺序（末列 OT 为目标）:', list(df_data.columns))""")

md("""## 第 2 步：切分边界 —— ETT 专用的 12/4/4 月固定切分（`data_loader.py:63-66`）

train = 前 12 个月×30×24 = 8640 小时；val = 4 个月 = 2880；test = 4 个月 = 2880。
注意 val/test 的**起点往前多算 `seq_len` 行**（`border1s[1]=8640-96`），
这是为了让 val/test 的第一个窗口也凑够 96 行输入——代价是 val 的前 96 行与 train 的末尾重叠。""")

code("""seq_len, label_len, pred_len = 96, 48, 96     # ETTh1_96_96 脚本参数
border1s = [0, 12*30*24 - seq_len, 12*30*24 + 4*30*24 - seq_len]
border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 4*30*24 + 4*30*24]
for name, b1, b2 in zip(['train', 'val', 'test'], border1s, border2s):
    print(f'{name:5s}: 行号 [{b1:5d}, {b2:5d})  共 {b2 - b1} 行  （起点比"纯 split"提前 {seq_len if b1 > 0 else 0} 行回看）')""")

md("""## 第 3 步：全局 StandardScaler —— 只用 train 段拟合（`data_loader.py:74-77`）

一个 scaler 管整条序列的 7 列（每列一组 mean/std）。
**fit 只用 train 行**（无泄漏），然后 transform 全序列，再按 border 切片。""")

code("""scaler = StandardScaler()
scaler.fit(df_data[border1s[0]:border2s[0]].values)          # 只用 train 拟合
data = scaler.transform(df_data.values)                      # transform 全序列
print('data:', data.shape)                                    # (17420, 7)
print('train 段逐列 |mean| 最大值:', np.abs(data[border1s[0]:border2s[0]].mean(0)).max(), '（≈0）')
print('train 段逐列 std 最小值:', data[border1s[0]:border2s[0]].std(0).min(), '（≈1）')
print('test 段逐列 mean:', np.round(data[border2s[2] - 2880:].mean(0), 3), '← 非 0：test 分布相对 train 漂移')""")

md("""## 第 4 步：时间特征（`timeenc=1, freq='h'`）

`run.py` 里 `--embed` 默认 `'timeF'` → `data_factory.py:24` 算出 `timeenc=1` → 走
`utils/timefeatures.py:147` 的 `time_features(dates, freq='h')`，按 `freq='h'` 查表（`timefeatures.py:106`）
得到 **4 个特征**：[HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]，每个都线性归一到 [-1, 1] 附近。""")

code("""df_stamp = df_raw[['date']].copy()
df_stamp['date'] = pd.to_datetime(df_stamp.date)
data_stamp_full = tf.time_features(pd.to_datetime(df_stamp['date'].values), freq='h')
data_stamp_full = data_stamp_full.transpose(1, 0)          # [T, 4]
print('data_stamp_full:', data_stamp_full.shape)
print('首行 4 个值:', np.round(data_stamp_full[0], 3), '← 对应', df_stamp['date'].iloc[0], '(周五 0 点)')""")

md("""## 第 5 步：按 split 切片（`data_loader.py:93-99`）

`data_x = data_y = data[border1:border2]`（同一份矩阵，窗口切法不同），
`data_stamp` 同步切片。训练集 8640 行，窗口数 = `T - seq_len - pred_len + 1`。""")

code("""set_type = 0                                             # 0=train, 1=val, 2=test
border1, border2 = border1s[set_type], border2s[set_type]
data_x    = data[border1:border2]
data_y    = data[border1:border2]
data_stamp = data_stamp_full[border1:border2]
print('data_x / data_y:', data_x.shape, '| data_stamp:', data_stamp.shape)
print('__len__ =', len(data_x) - seq_len - pred_len + 1, '个训练窗口')   # 8640-96-96+1 = 8449""")

md("""## 第 6 步：`__getitem__` 滑窗（`data_loader.py:101-112`）—— 原版源码摘录 + 手动复现

关键结构：`seq_y` 不是纯未来，而是 **label_len(48) 历史 + pred_len(96) 未来 = 144 行**，
其中前 48 行与 `seq_x` 的后 48 行完全重叠（decoder 的 "start token"）。""")

code("""_dl = ROOT / 'data_provider' / 'data_loader.py'
if _dl.exists():
    src = _dl.read_text().splitlines()
    print('\\n'.join(f'{i+1:4d} {l}' for i, l in enumerate(src[100:115], start=100)))
else:
    print('（未找到 data_provider/data_loader.py，跳过源码摘录；该文件仅用于展示）')""")

code("""def getitem(index):                       # 逐行复现 data_loader.py:101-112
    s_begin = index                       # 窗口起点
    s_end   = s_begin + seq_len           # 输入窗终点（不含）
    r_begin = s_end - label_len           # seq_y 起点 = 与输入窗重叠 48 行
    r_end   = r_begin + label_len + pred_len
    return (data_x[s_begin:s_end], data_y[r_begin:r_end],
            data_stamp[s_begin:s_end], data_stamp[r_begin:r_end])

seq_x, seq_y, seq_x_mark, seq_y_mark = getitem(0)
print('seq_x', seq_x.shape, '| seq_y', seq_y.shape,
      '| seq_x_mark', seq_x_mark.shape, '| seq_y_mark', seq_y_mark.shape)
print('重叠验证: seq_y[:48] 与 seq_x[-48:] 相同 →', np.allclose(seq_y[:48], seq_x[-48:]))
print('相邻窗口(index 0 vs 1)共享 95/96 个点 →', np.allclose(seq_x[1:], getitem(1)[0][:-1]))""")

md("""## 第 7 步：DataLoader batch 组装（`data_factory.py:26-45`：train `shuffle=True, drop_last=False`）

一个 batch 全是**同一条序列**的 4 个（准）随机窗口。""")

code("""class ETTSingleSeries(Dataset):
    def __init__(self, data_x, data_y, data_stamp):
        self.data_x, self.data_y, self.data_stamp = data_x, data_y, data_stamp
    def __len__(self):
        return len(self.data_x) - seq_len - pred_len + 1
    def __getitem__(self, index):
        return getitem(index)            # 复用上面的滑窗函数

train_loader = DataLoader(ETTSingleSeries(data_x, data_y, data_stamp),
                          batch_size=4, shuffle=True, drop_last=False)
batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
batch_x, batch_y = batch_x.float(), batch_y.float()
batch_x_mark, batch_y_mark = batch_x_mark.float(), batch_y_mark.float()
print('batch_x', batch_x.shape, '| batch_y', batch_y.shape,
      '| batch_x_mark', batch_x_mark.shape, '| batch_y_mark', batch_y_mark.shape)""")

md("""## 第 8 步：decoder 输入 `dec_inp`（`exp/exp_long_term_forecasting.py:111-112`）

`label_len` 段真值 + `pred_len` 段零。
**注意**：TimeXer 是 encoder-only，`forecast()` 的签名收下 `x_dec/x_mark_dec` 但根本不用（`models/TimeXer.py:157`），
这里构造它只是为了兼容上游 exp 的统一调用协议。""")

code("""dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float()
print('dec_inp:', dec_inp.shape)          # [4, 144, 7] = 48 真值 + 96 零
print('后 96 步全零:', bool((dec_inp[:, -pred_len:, :].abs().max() == 0).item()))""")

md("""---
# 二、喂进模型之后：模型侧（`models/TimeXer.py`，MS 模式）

构造模型。官方脚本 `TimeXer_ETTh1.sh`（96→96 档）用 `d_model=256, e_layers=1, batch_size=4`；
本 notebook 缩小为 `d_model=64` 以便 CPU 快速运行，**结构完全一致**，只改宽度。""")

code("""import argparse
configs = argparse.Namespace(
    task_name='long_term_forecast', features='MS',
    seq_len=96, label_len=48, pred_len=96,
    enc_in=7, dec_in=7, c_out=7,
    d_model=64, d_ff=128, e_layers=1, n_heads=4, factor=3,
    dropout=0.1, embed='timeF', freq='h', activation='gelu',
    patch_len=16, use_norm=1,
)
model = TimeXer(configs).float()
print(f'patch_num = {model.patch_num}  (= 96 // 16)')
print(f'n_vars    = {model.n_vars}   (MS → 1：只有末列 OT 是内生变量)')
print(f'head_nf   = {model.head_nf}  (= d_model × (patch_num+1) = 64×7)')
print(f'总参数量  = {sum(p.numel() for p in model.parameters()):,}')""")

md("""## 第 9 步：use_norm —— 窗口内实例归一化（RevIN 风格，`models/TimeXer.py:158-163`）

对**每个窗口、每个通道**分别减均值除标准差（沿时间维 dim=1），`detach()` 保证统计量不参与梯度。
反归一化在第 18 步用**目标通道（OT）的** mean/std 还原。""")

code("""x_enc = batch_x.clone()
means  = x_enc.mean(1, keepdim=True).detach()                                    # [B, 1, 7]
x_enc  = x_enc - means
stdev  = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, 7]
x_enc  = x_enc / stdev
print('means:', means.shape, '| stdev:', stdev.shape)
print('归一化后每个窗口每通道 |mean| 最大:', x_enc.mean(1).abs().max().item(), '（≈0）')""")

md("""## 第 10 步：内生 / 外生拆分（`models/TimeXer.py:167-168`）

MS 模式：**末列 OT 是内生变量**（1 条），其余 6 列是外生变量。
- 内生走 patch 路线（EnEmbedding）
- 外生走 inverted 路线（每列整个窗口压成 1 个 token）""")

code("""en_in = x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1)   # [B, 1, 96]  末列 OT
ex_in = x_enc[:, :, :-1]                                  # [B, 96, 6]  其余 6 列
print('en_in（内生）:', en_in.shape, '| ex_in（外生）:', ex_in.shape)
print('验证 en_in 就是 OT 通道:', torch.allclose(en_in[:, 0], x_enc[:, :, -1]))""")

md("""## 第 11 步：EnEmbedding —— 内生变量切成 patch + 1 个全局 token（`models/TimeXer.py:24-48`）

`unfold(size=16, step=16)` 把 96 点切成 6 个不重叠 patch；
每个 patch 经**共享** `Linear(16→64)` + 位置编码变成 1 个 token；
再 cat 1 个可学习的全局 token（`glb_token`）→ 共 **7 个内生 token**。""")

code("""model.eval()                                   # 分析阶段关 dropout，保证可复现
with torch.no_grad():
    en_embedding = model.en_embedding
    x = en_in
    n_vars = x.shape[1]
    glb = en_embedding.glb_token.repeat((x.shape[0], 1, 1, 1))
    print('glb_token 参数:', tuple(en_embedding.glb_token.shape), '→ repeat:', tuple(glb.shape))

    x_unfold = x.unfold(dimension=-1, size=en_embedding.patch_len, step=en_embedding.patch_len)
    print('1) unfold(16,16):', tuple(x_unfold.shape))            # [B, 1, 6, 16]
    print('   第 0 样本 patch#0 的 16 个原始点:', np.round(x_unfold[0, 0, 0].numpy(), 3))

    x_r = torch.reshape(x_unfold, (x_unfold.shape[0]*x_unfold.shape[1],
                                   x_unfold.shape[2], x_unfold.shape[3]))
    print('2) reshape:', tuple(x_r.shape))                      # [B, 6, 16]

    x_v = en_embedding.value_embedding(x_r) + en_embedding.position_embedding(x_r)
    print('3) Linear(16→64)+pos:', tuple(x_v.shape))            # [B, 6, 64]

    x_v = torch.reshape(x_v, (-1, n_vars, x_v.shape[-2], x_v.shape[-1]))
    x_v = torch.cat([x_v, glb], dim=2)
    print('4) cat glb_token:', tuple(x_v.shape))                # [B, 1, 7, 64]

    en_embed_manual = torch.reshape(x_v, (x_v.shape[0]*x_v.shape[1], x_v.shape[2], x_v.shape[-1]))
    print('5) 最终 en_embed:', tuple(en_embed_manual.shape))    # [B, 7, 64]

    en_embed_direct, _ = en_embedding(en_in)                    # 模块直接调用
    print('手动 vs 模块输出一致:', torch.allclose(en_embed_manual, en_embed_direct, atol=1e-6))""")

md("""## 第 12 步：DataEmbedding_inverted —— 外生变量 + 时间标记各成 1 个 token（`layers/Embed.py:129-143`）

**本仓库里最容易看漏的一步**：`forward(x, x_mark)` 在 `x_mark` 非 None 时把它
**cat 进变元维**（`Embed.py:141`），所以 6 个外生值列 + 4 个时间特征列 = **10 个外生 token**，
全部共享同一个 `Linear(96→64)`（把每列整段 96 点压成 1 个 64 维 token）。""")

code("""with torch.no_grad():
    ex_embedding = model.ex_embedding
    x_p    = ex_in.permute(0, 2, 1)                     # [B, 6, 96]
    mark_p = batch_x_mark.permute(0, 2, 1)              # [B, 4, 96]
    x_cat  = torch.cat([x_p, mark_p], 1)                # [B, 10, 96]
    print('值列 permute:', tuple(x_p.shape), '| 时间标记 permute:', tuple(mark_p.shape))
    print('cat 后（变元维 6+4=10）:', tuple(x_cat.shape), '← 时间特征变成第 7~10 个变元 token！')

    ex_embed = ex_embedding(ex_in, batch_x_mark)        # 模块直接调用
    print('ex_embed:', tuple(ex_embed.shape))           # [B, 10, 64]
    print('手动 Linear vs 模块一致:', torch.allclose(ex_embedding.value_embedding(x_cat), ex_embed, atol=1e-6))
    print('token 顺序: [HUFL, HULL, MUFL, MULL, LUFL, LULL, 小时, 星期, 月内日, 年内日]')""")

md("""## 第 13 步：EncoderLayer ① self-attention（`models/TimeXer.py:87-92` + `SelfAttention_Family.py:48-75,179-212`）

7 个内生 token 之间做标准多头注意力（`FullAttention(mask_flag=False)` → 无因果掩码），
残差连接 + LayerNorm。注意力矩阵是 `[B, H, 7, 7]`——**token 对 token**，不是时间对时间。""")

code("""with torch.no_grad():
    layer = model.encoder.layers[0]
    sa = layer.self_attention                    # AttentionLayer(FullAttention, d_model, n_heads)
    B = en_embed_manual.shape[0]
    H = sa.n_heads
    D = sa.query_projection.out_features // H    # 每头维度 64/4 = 16

    q = sa.query_projection(en_embed_manual).view(B, 7, H, D)
    k = sa.key_projection(en_embed_manual).view(B, 7, H, D)
    v = sa.value_projection(en_embed_manual).view(B, 7, H, D)
    print('q/k/v:', tuple(q.shape))                              # [B, 7, H, D]

    scores = torch.einsum('blhe,bshe->bhls', q, k)               # [B, H, 7, 7]
    print('scores:', tuple(scores.shape), '| scale = 1/sqrt(D) =', round(1/sqrt(D), 4))
    A = torch.softmax(scores / sqrt(D), dim=-1)
    print('attention 权重 A:', tuple(A.shape), '| 每行和=1:', bool(torch.allclose(A.sum(-1), torch.ones_like(A.sum(-1)), atol=1e-5)))
    V = torch.einsum('bhls,bshd->blhd', A, v)
    out = sa.out_projection(V.reshape(B, 7, H * D))
    print('self-attn 输出:', tuple(out.shape))                    # [B, 7, 64]

    x1 = layer.norm1(en_embed_manual + out)
    sa_direct, _ = sa(en_embed_manual, en_embed_manual, en_embed_manual, attn_mask=None)
    print('手动 vs AttentionLayer 直接调用一致:', torch.allclose(out, sa_direct, atol=1e-5))""")

md("""## 第 14 步：EncoderLayer ② cross-attention —— 全局 token 查外生 token（`models/TimeXer.py:94-104`）

只取 self-attn 之后的**全局 token（第 7 个，`x[:, -1, :]`）**当 query，
对 10 个外生 token 做 1 query × 10 keys 的注意力，残差 + LayerNorm。
这就是 TimeXer 名义上的"外生信息注入点"，且**只更新全局 token**。""")

code("""with torch.no_grad():
    x_glb_ori = x1[:, -1, :].unsqueeze(1)          # 全局 token = 序列最后一个
    print('query（全局 token）:', tuple(x_glb_ori.shape))         # [B, 1, 64]

    ca = layer.cross_attention
    q_g = ca.query_projection(x_glb_ori).view(B, 1, H, D)
    k_g = ca.key_projection(ex_embed).view(B, 10, H, D)
    scores_g = torch.einsum('blhe,bshe->bhls', q_g, k_g)
    A_g = torch.softmax(scores_g / sqrt(D), dim=-1)
    print('cross-attn 权重:', tuple(A_g.shape), '（1 个 query 对 10 个外生 token）')
    print('第 0 个样本各头对 10 个外生 token 的平均注意力:',
          np.round(A_g[0].mean(0)[0].numpy(), 3))

    x_glb_attn, _ = layer.cross_attention(x_glb_ori, ex_embed, ex_embed, attn_mask=None)
    x_glb = layer.norm2(x_glb_ori + x_glb_attn)
    print('更新后的全局 token:', tuple(x_glb.shape))              # [B, 1, 64]""")

md("""## 第 15 步：EncoderLayer ③ FFN + 拼回 7 token（`models/TimeXer.py:106-111`）与整个 Encoder

把 6 个 patch token（`x[:, :-1, :]`）和更新后的全局 token 拼回 `[B, 7, 64]`，
过 Conv1d(64→128)→GELU→Conv1d(128→64) 的逐 token FFN，残差 + LayerNorm。
`Encoder`（`TimeXer.py:51-67`）循环 e_layers 层后接最终 LayerNorm。""")

code("""with torch.no_grad():
    y_cat = torch.cat([x1[:, :-1, :], x_glb], dim=1)
    print('拼接 6 patch token + 新全局 token:', tuple(y_cat.shape))   # [B, 7, 64]

    y = layer.dropout(layer.activation(layer.conv1(y_cat.transpose(-1, 1))))
    y = layer.dropout(layer.conv2(y).transpose(-1, 1))
    layer_out = layer.norm3(y_cat + y)
    print('EncoderLayer 输出:', tuple(layer_out.shape))

    enc_out = model.encoder(en_embed_manual, ex_embed)               # 整个 Encoder（含最终 LayerNorm）
    print('Encoder 整体输出:', tuple(enc_out.shape))                  # [B, 7, 64]
    print('token 记账: 内生 7（6 patch + 1 glb）+ 外生 10（6 值列 + 4 时间列）= 17 个 token')""")

md("""## 第 16 步：reshape / permute 到 head 输入（`models/TimeXer.py:171-174`）""")

code("""with torch.no_grad():
    enc_r = torch.reshape(enc_out, (-1, 1, enc_out.shape[-2], enc_out.shape[-1]))
    print('reshape 引入 n_vars 维:', tuple(enc_r.shape))            # [B, 1, 7, 64]
    enc_p = enc_r.permute(0, 1, 3, 2)
    print('permute 到 [B, nvars, d_model, patch_num+1]:', tuple(enc_p.shape))  # [B, 1, 64, 7]""")

md("""## 第 17 步：FlattenHead —— 拍平投影成预测（`models/TimeXer.py:9-21, 153-155`）

`d_model×(patch_num+1) = 64×7 = 448` 维拍平后一个 `Linear(448→96)` 直接输出 96 步预测。""")

code("""with torch.no_grad():
    flat = model.head.flatten(enc_p)
    print('flatten(d_model × (patch_num+1)):', tuple(flat.shape))   # [B, 1, 448]
    dec = model.head.linear(flat)
    print('Linear(448→96):', tuple(dec.shape))                      # [B, 1, 96]
    dec_out = dec.permute(0, 2, 1)
    print('dec_out:', tuple(dec_out.shape))                         # [B, 96, 1]""")

md("""## 第 18 步：de-normalization（`models/TimeXer.py:179-182`）

用第 9 步存的**目标通道（OT，末列）**的 mean/std 把预测还原回标准化前的尺度。""")

code("""with torch.no_grad():
    dec_out_denorm = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, pred_len, 1))
    dec_out_denorm = dec_out_denorm + (means[:, 0, -1:].unsqueeze(1).repeat(1, pred_len, 1))
    print('还原用的 stdev[:, 0, -1]:', tuple(stdev[:, 0, -1:].shape), '=', stdev[:, 0, -1].flatten().tolist())
    print('反归一化后 dec_out:', tuple(dec_out_denorm.shape))""")

md("""## 第 19 步：整体 forward 一次 + loss 口径（`exp/exp_long_term_forecasting.py:119-131`）

`forward` 返回后 exp 再切 `[:, -pred_len:, f_dim:]`：
- `MS`：`f_dim=-1` → 只取目标通道，loss 是 `[B, 96, 1]` 的 MSE
- `M` ：`f_dim=0` → 全部 7 通道，loss 是 `[B, 96, 7]` 的 MSE（下一节对照）""")

code("""with torch.no_grad():
    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    print('model 输出:', tuple(outputs.shape))                       # MS: [4, 96, 1]
    f_dim = -1 if configs.features == 'MS' else 0
    outputs_ms = outputs[:, -pred_len:, f_dim:]
    batch_y_ms = batch_y[:, -pred_len:, f_dim:]
    print('MS 切片后:', tuple(outputs_ms.shape))
    print('MSE loss:', nn.functional.mse_loss(outputs_ms, batch_y_ms).item())""")

md("""## 第 20 步：一个完整训练 step（复现 `exp_long_term_forecasting.py:104-147` 的单步）

`zero_grad → forward → MSE → backward → Adam step`。原版默认优化器是 Adam，无梯度裁剪。""")

code("""model.train()                                   # 重新打开 dropout
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

def train_step():
    optimizer.zero_grad()
    out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    f_dim = -1 if configs.features == 'MS' else 0
    loss = criterion(out[:, -pred_len:, f_dim:], batch_y[:, -pred_len:, f_dim:])
    loss.backward()
    optimizer.step()
    return loss.item()

print('step 1 loss:', train_step())
print('step 2 loss:', train_step(), '← 同一批数据连训两步，loss 应下降')""")

md("""---
# 三、对照：`features='M'` 走 forecast_multi（`models/TimeXer.py:187-214`）

M 模式下 7 列**全部是内生变量**（`n_vars=7`）：每列各自切 6 patch + 1 glb token；
外生侧只剩**时间标记 4 个 token**（`ex_embedding(x_enc, x_mark)` 里 `x_enc` 7 列 permute 后与 4 列时间 cat 成 11 个 token——
注意此时外生 token = 7 值列 + 4 时间列 = 11）。head 对 7 个变元各自输出 96 步。""")

code("""configs_m = argparse.Namespace(**{**vars(configs), 'features': 'M'})
model_m = TimeXer(configs_m).float().eval()
with torch.no_grad():
    out_m = model_m(batch_x, batch_x_mark, dec_inp, batch_y_mark)
print('M 模式 n_vars:', model_m.n_vars)
print('M 模式输出:', tuple(out_m.shape))          # [4, 96, 7]
print('M 模式 token 记账: 每变元 7 内生 token（共 7×7）+ 外生 11 token（7 值列 + 4 时间列）')""")

md("""---
# 四、速查：原版（本 notebook）vs 多股票面板版（远程仓库 `stock/`）

| 维度 | 原版（ETTh1 单序列） | `stock/` 面板版 |
|---|---|---|
| 数据 | 1 条 `17420×7` 单表 | N 只股票各一条 `Tᵢ×C`，schema 全局统一 |
| 切分 | 行号 12/4/4 月 | 按 `target_date` 全局日期（`stock/data/windows.py:76-83`） |
| 标准化 | 1 个全局 scaler（train 拟合，`data_loader.py:74-77`） | per-symbol per-column，导出期固化（`stock/data/preprocessing.py:44-72`） |
| 滑窗 | 单轴滑窗，`seq_y=144` 行含 48 行重叠 | 每股独立滑窗、`pred_len=1` 单步目标日，样本带 `symbol/split/label_valid`（`windows.py:63-93`） |
| batch | 同序列窗口 `[4,96,7]` + 时间标记 | 跨股票混装 `[256,60,C]`，六元组含 `sid/y_raw/ex_mask`（`panel.py:326-350`） |
| 时间特征 | 4 个时间列 cat 成外生 token（`Embed.py:141`） | 不用时间 token；`x_mark` 恒 None（`adapters/timexer.py:98-99`），星期语义进 `target_dow_*` 特征列 |
| 外生 token | 共享 `Linear(96→64)` | 共享 `Linear(60→64)` + `feature_id` 身份编码（`timexer.py:19-36`）+ `ex_mask` 屏蔽 |
| use_norm | 窗口内 RevIN（=1） | 0：保留导出期尺度（`timexer.py:87`） |
| loss | 全通道全点 MSE | 单目标 per-sample MSE × 股票等权（`trainer.py:49-50`） |
| 优化 | Adam、可选 AMP、无裁剪 | AdamW + clip 1.0（`trainer.py:54-61`） |

**原版文件清单**（本目录）：
```
timexer_upstream/
├── models/TimeXer.py            # 模型本体（未改动）
├── layers/SelfAttention_Family.py
├── layers/Embed.py
├── utils/masking.py
├── utils/timefeatures.py
├── data_provider/data_loader.py # Dataset_ETT_hour（未 import，仅摘源码）
├── data_provider/data_factory.py
├── run.py                       # 参数默认值参照
├── ETTh1.csv                    # 17420×8
└── make_notebook.py             # 本 notebook 的生成脚本
```""")

# ============================================================ 写出
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "TimeXer_step_by_step.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"OK: {out}  ({len(cells)} cells, {sum(c['cell_type']=='code' for c in cells)} code)")

import sys, types, warnings
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
print('torch', torch.__version__, '| pandas', pd.__version__, '| numpy', np.__version__)

df_raw = pd.read_csv(ROOT / 'ETTh1.csv')
print('df_raw:', df_raw.shape)      # (17420, 8) = 1 列 date + 7 列数据
print('列名:', list(df_raw.columns))
df_raw.head(3)

features, target = 'MS', 'OT'   # 本 notebook 用 MS；'M' 的对照见最后一节
cols_data = df_raw.columns[1:]     # 丢掉 date 列
df_data = df_raw[cols_data]
print('df_data:', df_data.shape)   # (17420, 7)
print('列顺序（末列 OT 为目标）:', list(df_data.columns))

seq_len, label_len, pred_len = 96, 48, 96     # ETTh1_96_96 脚本参数
border1s = [0, 12*30*24 - seq_len, 12*30*24 + 4*30*24 - seq_len]
border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 4*30*24 + 4*30*24]
for name, b1, b2 in zip(['train', 'val', 'test'], border1s, border2s):
    print(f'{name:5s}: 行号 [{b1:5d}, {b2:5d})  共 {b2 - b1} 行  （起点比"纯 split"提前 {seq_len if b1 > 0 else 0} 行回看）')

scaler = StandardScaler()
scaler.fit(df_data[border1s[0]:border2s[0]].values)          # 只用 train 拟合
data = scaler.transform(df_data.values)                      # transform 全序列
print('data:', data.shape)                                    # (17420, 7)
print('train 段逐列 |mean| 最大值:', np.abs(data[border1s[0]:border2s[0]].mean(0)).max(), '（≈0）')
print('train 段逐列 std 最小值:', data[border1s[0]:border2s[0]].std(0).min(), '（≈1）')
print('test 段逐列 mean:', np.round(data[border2s[2] - 2880:].mean(0), 3), '← 非 0：test 分布相对 train 漂移')

df_stamp = df_raw[['date']].copy()
df_stamp['date'] = pd.to_datetime(df_stamp.date)
data_stamp_full = tf.time_features(pd.to_datetime(df_stamp['date'].values), freq='h')
data_stamp_full = data_stamp_full.transpose(1, 0)          # [T, 4]
print('data_stamp_full:', data_stamp_full.shape)
print('首行 4 个值:', np.round(data_stamp_full[0], 3), '← 对应', df_stamp['date'].iloc[0], '(周五 0 点)')

set_type = 0                                             # 0=train, 1=val, 2=test
border1, border2 = border1s[set_type], border2s[set_type]
data_x    = data[border1:border2]
data_y    = data[border1:border2]
data_stamp = data_stamp_full[border1:border2]
print('data_x / data_y:', data_x.shape, '| data_stamp:', data_stamp.shape)
print('__len__ =', len(data_x) - seq_len - pred_len + 1, '个训练窗口')   # 8640-96-96+1 = 8449

_dl = ROOT / 'data_provider' / 'data_loader.py'
if _dl.exists():
    src = _dl.read_text().splitlines()
    print('\n'.join(f'{i+1:4d} {l}' for i, l in enumerate(src[100:115], start=100)))
else:
    print('（未找到 data_provider/data_loader.py，跳过源码摘录；该文件仅用于展示）')

def getitem(index):                       # 逐行复现 data_loader.py:101-112
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
print('相邻窗口(index 0 vs 1)共享 95/96 个点 →', np.allclose(seq_x[1:], getitem(1)[0][:-1]))

class ETTSingleSeries(Dataset):
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
      '| batch_x_mark', batch_x_mark.shape, '| batch_y_mark', batch_y_mark.shape)

dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float()
print('dec_inp:', dec_inp.shape)          # [4, 144, 7] = 48 真值 + 96 零
print('后 96 步全零:', bool((dec_inp[:, -pred_len:, :].abs().max() == 0).item()))

import argparse
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
print(f'总参数量  = {sum(p.numel() for p in model.parameters()):,}')

x_enc = batch_x.clone()
means  = x_enc.mean(1, keepdim=True).detach()                                    # [B, 1, 7]
x_enc  = x_enc - means
stdev  = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, 7]
x_enc  = x_enc / stdev
print('means:', means.shape, '| stdev:', stdev.shape)
print('归一化后每个窗口每通道 |mean| 最大:', x_enc.mean(1).abs().max().item(), '（≈0）')

en_in = x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1)   # [B, 1, 96]  末列 OT
ex_in = x_enc[:, :, :-1]                                  # [B, 96, 6]  其余 6 列
print('en_in（内生）:', en_in.shape, '| ex_in（外生）:', ex_in.shape)
print('验证 en_in 就是 OT 通道:', torch.allclose(en_in[:, 0], x_enc[:, :, -1]))

model.eval()                                   # 分析阶段关 dropout，保证可复现
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
    print('手动 vs 模块输出一致:', torch.allclose(en_embed_manual, en_embed_direct, atol=1e-6))

with torch.no_grad():
    ex_embedding = model.ex_embedding
    x_p    = ex_in.permute(0, 2, 1)                     # [B, 6, 96]
    mark_p = batch_x_mark.permute(0, 2, 1)              # [B, 4, 96]
    x_cat  = torch.cat([x_p, mark_p], 1)                # [B, 10, 96]
    print('值列 permute:', tuple(x_p.shape), '| 时间标记 permute:', tuple(mark_p.shape))
    print('cat 后（变元维 6+4=10）:', tuple(x_cat.shape), '← 时间特征变成第 7~10 个变元 token！')

    ex_embed = ex_embedding(ex_in, batch_x_mark)        # 模块直接调用
    print('ex_embed:', tuple(ex_embed.shape))           # [B, 10, 64]
    print('手动 Linear vs 模块一致:', torch.allclose(ex_embedding.value_embedding(x_cat), ex_embed, atol=1e-6))
    print('token 顺序: [HUFL, HULL, MUFL, MULL, LUFL, LULL, 小时, 星期, 月内日, 年内日]')

with torch.no_grad():
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
    print('手动 vs AttentionLayer 直接调用一致:', torch.allclose(out, sa_direct, atol=1e-5))

with torch.no_grad():
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
    print('更新后的全局 token:', tuple(x_glb.shape))              # [B, 1, 64]

with torch.no_grad():
    y_cat = torch.cat([x1[:, :-1, :], x_glb], dim=1)
    print('拼接 6 patch token + 新全局 token:', tuple(y_cat.shape))   # [B, 7, 64]

    y = layer.dropout(layer.activation(layer.conv1(y_cat.transpose(-1, 1))))
    y = layer.dropout(layer.conv2(y).transpose(-1, 1))
    layer_out = layer.norm3(y_cat + y)
    print('EncoderLayer 输出:', tuple(layer_out.shape))

    enc_out = model.encoder(en_embed_manual, ex_embed)               # 整个 Encoder（含最终 LayerNorm）
    print('Encoder 整体输出:', tuple(enc_out.shape))                  # [B, 7, 64]
    print('token 记账: 内生 7（6 patch + 1 glb）+ 外生 10（6 值列 + 4 时间列）= 17 个 token')

with torch.no_grad():
    enc_r = torch.reshape(enc_out, (-1, 1, enc_out.shape[-2], enc_out.shape[-1]))
    print('reshape 引入 n_vars 维:', tuple(enc_r.shape))            # [B, 1, 7, 64]
    enc_p = enc_r.permute(0, 1, 3, 2)
    print('permute 到 [B, nvars, d_model, patch_num+1]:', tuple(enc_p.shape))  # [B, 1, 64, 7]

with torch.no_grad():
    flat = model.head.flatten(enc_p)
    print('flatten(d_model × (patch_num+1)):', tuple(flat.shape))   # [B, 1, 448]
    dec = model.head.linear(flat)
    print('Linear(448→96):', tuple(dec.shape))                      # [B, 1, 96]
    dec_out = dec.permute(0, 2, 1)
    print('dec_out:', tuple(dec_out.shape))                         # [B, 96, 1]

with torch.no_grad():
    dec_out_denorm = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, pred_len, 1))
    dec_out_denorm = dec_out_denorm + (means[:, 0, -1:].unsqueeze(1).repeat(1, pred_len, 1))
    print('还原用的 stdev[:, 0, -1]:', tuple(stdev[:, 0, -1:].shape), '=', stdev[:, 0, -1].flatten().tolist())
    print('反归一化后 dec_out:', tuple(dec_out_denorm.shape))

with torch.no_grad():
    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    print('model 输出:', tuple(outputs.shape))                       # MS: [4, 96, 1]
    f_dim = -1 if configs.features == 'MS' else 0
    outputs_ms = outputs[:, -pred_len:, f_dim:]
    batch_y_ms = batch_y[:, -pred_len:, f_dim:]
    print('MS 切片后:', tuple(outputs_ms.shape))
    print('MSE loss:', nn.functional.mse_loss(outputs_ms, batch_y_ms).item())

model.train()                                   # 重新打开 dropout
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
print('step 2 loss:', train_step(), '← 同一批数据连训两步，loss 应下降')

configs_m = argparse.Namespace(**{**vars(configs), 'features': 'M'})
model_m = TimeXer(configs_m).float().eval()
with torch.no_grad():
    out_m = model_m(batch_x, batch_x_mark, dec_inp, batch_y_mark)
print('M 模式 n_vars:', model_m.n_vars)
print('M 模式输出:', tuple(out_m.shape))          # [4, 96, 7]
print('M 模式 token 记账: 每变元 7 内生 token（共 7×7）+ 外生 11 token（7 值列 + 4 时间列）')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import operator


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        if d_model < 1 or max_len < 1:
            raise ValueError('d_model and max_len must be positive')

        # Compute the positional encodings once in log space.
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                             * -(math.log(10000.0) / d_model))

        angles = position * div_term
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        # support odd d_model: there is one more even column than odd columns
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, :pe[:, 1::2].shape[1]])

        # keep the buffer persistent ('pe' key) for checkpoint compatibility
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f'PositionalEmbedding expects a 3-D input [B, L, D], got {tuple(x.shape)}')
        length = x.size(1)
        if length > self.pe.size(1):
            raise ValueError(
                f'sequence length {length} exceeds the positional encoding limit {self.pe.size(1)}')
        # cast to the input dtype so AMP does not silently up-cast the output
        return self.pe[:, :length].to(dtype=x.dtype)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # all supported torch versions (>= 1.5) use padding=1 here
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        position = torch.arange(0, c_in, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                             * -(math.log(10000.0) / d_model))

        angles = position * div_term
        w = torch.zeros(c_in, d_model, dtype=torch.float32)
        # support odd d_model: there is one more even column than odd columns
        w[:, 0::2] = torch.sin(angles)
        w[:, 1::2] = torch.cos(angles[:, :w[:, 1::2].shape[1]])

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Categorical embedding of integer time-stamp features (embed != 'timeF').

    The columns read from ``x_mark`` are fixed by ``freq``:
      d, b: [month, day, weekday]                          (3 fields)
      h, s: [month, day, weekday, hour]                    (4 fields)
      t:    [month, day, weekday, hour, minute]            (5 fields)
    Expected value ranges: month 1-12, day 1-31, weekday 0-6, hour 0-23 and
    minute 0-3 (``minute_size=4`` buckets of 15 minutes, NOT raw 0-59 minutes).
    Continuous time features must be fed to ``TimeFeatureEmbedding`` instead.
    """

    _FIELDS_PER_FREQ = {'d': 3, 'b': 3, 'h': 4, 's': 4, 't': 5}

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        if freq not in self._FIELDS_PER_FREQ:
            raise ValueError(
                f'TemporalEmbedding does not support freq={freq!r}; '
                f"supported freqs are {sorted(self._FIELDS_PER_FREQ)} "
                '(continuous time features need embed_type="timeF")')
        self.freq = freq

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        # hour_embed is created for every freq so old checkpoints keep the
        # same key set (it is simply not read for daily/business-day freqs,
        # whose marks have no hour column)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f'x_mark must have shape [B, L, K], got {tuple(x.shape)}')
        required = self._FIELDS_PER_FREQ[self.freq]
        if x.size(-1) < required:
            raise ValueError(
                f"freq={self.freq!r} requires at least {required} integer time "
                f'fields (month/day/weekday[/hour][/minute]), got {x.size(-1)} '
                'columns; continuous time features must use '
                'TimeFeatureEmbedding (embed_type="timeF")')

        x = x.long()
        result = (self.month_embed(x[:, :, 0])
                  + self.day_embed(x[:, :, 1])
                  + self.weekday_embed(x[:, :, 2]))
        if self.freq in ('h', 's', 't'):
            result = result + self.hour_embed(x[:, :, 3])
        if self.freq == 't':
            result = result + self.minute_embed(x[:, :, 4])
        return result


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # project first, then add the position embedding: under AMP autocast the
        # Linear may return fp16, and the position encoding should match the
        # projected dtype rather than the raw input dtype
        x = self.value_embedding(x)
        if x_mark is None:
            x = x + self.position_embedding(x)
        else:
            x = x + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        # `seq_len` (the lookback window length) is the input dimension of the value
        # embedding: each variate's whole series is projected onto one token.
        # `embed_type` / `freq` are kept for interface compatibility but unused here.
        # operator.index() rejects non-integers (e.g. 60.9) with a TypeError
        # instead of silently truncating them like int() would
        self.seq_len = operator.index(seq_len)
        if self.seq_len < 1:
            raise ValueError('seq_len must be a positive integer')
        self.value_embedding = nn.Linear(self.seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        if x.ndim != 3:
            raise ValueError(
                f'DataEmbedding_inverted expects x of shape [B, L, F], got {tuple(x.shape)}')
        if x.size(1) != self.seq_len:
            raise ValueError(
                f'window length {x.size(1)} does not match the expected seq_len {self.seq_len}')
        if x_mark is not None and (x_mark.ndim != 3 or x_mark.shape[:2] != x.shape[:2]):
            raise ValueError('x_mark must match x on the batch and time dimensions')

        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding: project first so the position embedding matches the
        # projected dtype (see DataEmbedding.forward)
        x = self.value_embedding(x)
        x = x + self.position_embedding(x)
        return self.dropout(x), n_vars

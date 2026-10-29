import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from einops import rearrange, repeat


def _resolve_attention_mask(attn_mask, mask_flag, batch_size, query_len, key_len, device):
    """Resolve the mask to apply to attention scores ([B, H, L, S], True = masked).

    An explicitly passed ``attn_mask`` always takes effect, regardless of ``mask_flag``.
    ``mask_flag`` only controls whether a triangular causal mask is auto-generated when
    no explicit mask is given (self-attention with L == S only).
    """
    if attn_mask is not None:
        mask = attn_mask.mask if hasattr(attn_mask, 'mask') else attn_mask
        if not torch.is_tensor(mask):
            raise TypeError(
                'attention mask must be a torch.Tensor '
                '(or an object exposing a bool `.mask` tensor)')
        if mask.dtype != torch.bool:
            raise TypeError('attention mask must be a bool tensor where True means masked')
        return mask.to(device=device)

    if mask_flag:
        if query_len != key_len:
            raise ValueError(
                'Auto triangular causal mask is only defined for self-attention (L == S); '
                'pass an explicit mask for cross-attention.')
        return TriangularCausalMask(batch_size, query_len, device=device).mask

    return None


def _apply_attention_mask(scores, mask, check_rows_fully_masked=True):
    """Validate ``mask``, broadcast it to ``scores`` and return the masked scores.

    ``scores``: [B, H, L, S]. Supported mask shapes (True = masked):
      [L, S]       shared by all samples and heads
      [B, L, S]    per-sample mask (the batch dim may also be 1)
      [B, H, L, S] full mask; each dim may be a broadcastable 1
    A key padding mask [B, S] must be converted by the caller to [B, 1, 1, S]:
    a 2-D mask is *never* interpreted as [B, S], because when B == L it would be
    indistinguishable from [L, S].

    This function is NOT in-place: callers must use the returned tensor.
    """
    if not torch.is_tensor(mask):
        raise TypeError('attention mask must be a torch.Tensor')

    if mask.dtype != torch.bool:
        raise TypeError('attention mask must be bool; True means masked')

    B, H, L, S = scores.shape

    if mask.ndim == 2:
        if tuple(mask.shape) != (L, S):
            raise ValueError(
                f'2-D mask must be [L, S] = [{L}, {S}], got {tuple(mask.shape)}; '
                'convert a key padding mask to [B, 1, 1, S]')
        mask = mask[None, None, :, :]

    elif mask.ndim == 3:
        if mask.shape[0] not in (1, B):
            raise ValueError(
                f'3-D mask must be [B, L, S] with batch size 1 or {B}, '
                f'got batch size {mask.shape[0]}')
        if tuple(mask.shape[1:]) != (L, S):
            raise ValueError(
                f'3-D mask must be [B, L, S] = [{B}, {L}, {S}], got {tuple(mask.shape)}')
        # per-SAMPLE mask: insert the head axis between batch and query axes.
        # (plain broadcast_to would right-align [B, L, S] as [1, B, L, S] and
        # silently apply sample masks to heads when B == H)
        mask = mask[:, None, :, :]

    elif mask.ndim != 4:
        raise ValueError('attention mask must have 2, 3 or 4 dimensions')

    mask = mask.to(device=scores.device)

    try:
        mask = torch.broadcast_to(mask, scores.shape)
    except RuntimeError as exc:
        raise ValueError(
            f'mask shape {tuple(mask.shape)} cannot broadcast to '
            f'scores shape {tuple(scores.shape)}') from exc

    if check_rows_fully_masked and bool(mask.all(dim=-1).any()):
        raise ValueError(
            'at least one query has all of its keys masked; '
            'softmax over a fully masked row is undefined')

    return scores.masked_fill(mask, float('-inf'))


def _check_qkv(queries, keys, values):
    """Validate attention inputs and derive their dimensions.

    queries: [B, L, H, E]; keys: [B, S, H, E]; values: [B, S, H, D].
    The key length S is taken from ``keys``: it is what defines the score
    matrix, while values only provide the V vectors.

    Returns (B, L, H, E, S).
    """
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError(
            'queries, keys and values must be 4-D tensors '
            '([B, L, H, E] / [B, S, H, E] / [B, S, H, D]); got shapes '
            f'{tuple(queries.shape)}, {tuple(keys.shape)}, {tuple(values.shape)}')

    B, L, H, E = queries.shape
    S = keys.size(1)

    if keys.size(0) != B or values.size(0) != B:
        raise ValueError(
            'queries, keys and values must share the batch size; got '
            f'{B}, {keys.size(0)}, {values.size(0)}')
    if keys.size(2) != H or values.size(2) != H:
        raise ValueError(
            'queries, keys and values must share the number of heads; got '
            f'{H}, {keys.size(2)}, {values.size(2)}')
    if keys.size(1) != values.size(1):
        raise ValueError(
            f'keys and values must share the key length; got {keys.size(1)}, {values.size(1)}')
    if keys.size(3) != E:
        raise ValueError(
            f'queries and keys must share the per-head dimension; got {E}, {keys.size(3)}')
    if min(B, L, H, E, S, values.size(3)) < 1:
        raise ValueError('batch, sequence, head and per-head dimensions must be positive')

    return B, L, H, E, S


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E, S = _check_qkv(queries, keys, values)
        scale = self.scale if self.scale is not None else 1. / sqrt(E)

        if tau is not None:
            # require the unambiguous [B, 1]: a [B] tau would right-align to the
            # head axis and silently corrupt results when B == H
            if not torch.is_tensor(tau) or tuple(tau.shape) != (B, 1):
                got = tuple(tau.shape) if torch.is_tensor(tau) else type(tau).__name__
                raise ValueError(f'tau must have shape [B, 1] = [{B}, 1], got {got}')
            tau = tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        else:
            tau = 1.0
        if delta is not None:
            if not torch.is_tensor(delta) or tuple(delta.shape) != (B, S):
                got = tuple(delta.shape) if torch.is_tensor(delta) else type(delta).__name__
                raise ValueError(f'delta must have shape [B, S] = [{B}, {S}], got {got}')
            delta = delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S
        else:
            delta = 0.0

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors.
        # Scale BEFORE masking: masking first would turn -inf entries into NaN
        # for scale == 0 and into +inf for negative scales.
        scores = (torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta) * scale

        mask = _resolve_attention_mask(attn_mask, self.mask_flag, B, L, S, queries.device)
        if mask is not None:
            # the auto causal mask can never fully mask a row; only check explicit masks
            scores = _apply_attention_mask(
                scores, mask, check_rows_fully_masked=attn_mask is not None)

        A = self.dropout(torch.softmax(scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E, S = _check_qkv(queries, keys, values)
        scale = self.scale if self.scale is not None else 1. / sqrt(E)

        # Scale BEFORE masking: masking first would turn -inf entries into NaN
        # for scale == 0 and into +inf for negative scales.
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale

        mask = _resolve_attention_mask(attn_mask, self.mask_flag, B, L, S, queries.device)
        if mask is not None:
            # the auto causal mask can never fully mask a row; only check explicit masks
            scores = _apply_attention_mask(
                scores, mask, check_rows_fully_masked=attn_mask is not None)

        A = self.dropout(torch.softmax(scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        # keep all index tensors on the data device
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(
            L_Q, device=Q.device).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B, device=Q.device)[:, None, None],
                   torch.arange(H, device=Q.device)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = self.dropout(torch.softmax(scores, dim=-1))  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B, device=context_in.device)[:, None, None],
        torch.arange(H, device=context_in.device)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            # Non-top queries are approximated by a uniform attention over all
            # keys; the placeholder map carries the full [B, H, L_Q, L_V] shape
            # (which also covers cross-attention where L_Q != L_V). It is an
            # approximation, not an exact explanation of the returned context.
            attns = attn.new_full((B, H, L_Q, L_V), 1.0 / L_V)
            attns[torch.arange(B, device=attn.device)[:, None, None],
            torch.arange(H, device=attn.device)[None, :, None],
            index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        if attn_mask is not None:
            # ProbAttention's sampling cannot honor arbitrary masks: both the
            # top-query path (masked_fill_ in _update_context) and the initial
            # context of the remaining queries (V.mean / cumsum) would have to
            # exclude masked keys. Reject explicitly instead of silently
            # ignoring (mask_flag=False) or overriding (mask_flag=True) the mask.
            raise NotImplementedError(
                'ProbAttention does not support explicit attention masks; '
                'use FullAttention instead')

        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        # keep at least one sampled key / top query (ceil(log(1)) == 0)
        U_part = max(U_part, 1)
        u = max(u, 1)

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale if self.scale is not None else 1. / sqrt(D)
        scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        # AttentionLayer expects [B, L, H, D]; context is [B, H, L, D]
        return context.transpose(1, 2).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        if d_model < 1 or n_heads < 1:
            raise ValueError('d_model and n_heads must be positive')
        # explicit None checks: `d_keys or ...` would silently treat a
        # deliberate d_keys=0 as "not provided"
        if d_keys is None:
            d_keys = d_model // n_heads
        if d_values is None:
            d_values = d_model // n_heads
        if d_keys < 1 or d_values < 1:
            raise ValueError('per-head dimensions must be positive')

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        if queries.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
            raise ValueError(
                'queries, keys and values must be 3-D tensors '
                '([B, L, D] / [B, S, D] / [B, S, D]); got shapes '
                f'{tuple(queries.shape)}, {tuple(keys.shape)}, {tuple(values.shape)}')
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        if keys.size(0) != B or values.size(0) != B:
            raise ValueError(
                'queries, keys and values must share the batch size; got '
                f'{B}, {keys.size(0)}, {values.size(0)}')
        if tuple(keys.shape[:2]) != tuple(values.shape[:2]):
            raise ValueError(
                'keys and values must share the batch and key dimensions; got '
                f'{tuple(keys.shape[:2])}, {tuple(values.shape[:2])}')

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.reshape(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        try:
            # lazy import: models that do not use ReformerLayer (e.g. TimeXer)
            # must not require `reformer-pytorch` to be installed
            from reformer_pytorch import LSHSelfAttention
        except ImportError as exc:
            raise ImportError(
                'ReformerLayer requires the `reformer-pytorch` package '
                '(install it with: pip install reformer-pytorch).') from exc
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series, keeping device and dtype of the input
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, queries.new_zeros(B, fill_len, C)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        if attn_mask is not None:
            # LSHSelfAttention has no mask interface, and zero-padding does NOT
            # make padded tokens stop attending; refuse instead of ignoring
            raise NotImplementedError(
                'ReformerLayer does not support attention masks; pass attn_mask=None')
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=False), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=False), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=False), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if attn_mask is not None:
            # the mask is not plumbed through the two-stage attention; refuse
            # instead of silently ignoring it
            raise NotImplementedError(
                'TwoStageAttentionLayer does not support attention masks; pass attn_mask=None')
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out

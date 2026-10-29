"""Regression tests for layers/Embed.py and layers/SelfAttention_Family.py.

Covers the mask-semantics review checklist:
  - mask batch/head broadcast ambiguity ([B, L, S] must be per-SAMPLE, never
    right-aligned to the head axis even when B == H)
  - scale applied BEFORE masking (scale == 0 must not produce NaN)
  - fully-masked query rows are rejected
  - [B, 1, 1, S] key padding masks
  - TemporalEmbedding daily/business-day freqs must not read the hour column
  - ProbAttention explicitly rejects attn_mask
  - Q/K/V structural validation and AttentionLayer argument validation
  - TimeXer: eval determinism (single vs. mixed batch), ex_mask effectiveness,
    checkpoint save/load regression

Run standalone:  python tests/test_embed_and_attention.py
Run with pytest: pytest tests/test_embed_and_attention.py
"""
import math
import os
import sys
import tempfile
import traceback
from types import SimpleNamespace

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from layers.Embed import (  # noqa: E402
    DataEmbedding,
    DataEmbedding_inverted,
    PatchEmbedding,
    PositionalEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
)
from layers.SelfAttention_Family import (  # noqa: E402
    AttentionLayer,
    DSAttention,
    FullAttention,
    ProbAttention,
    TwoStageAttentionLayer,
    _apply_attention_mask,
    _resolve_attention_mask,
)
from utils.masking import TriangularCausalMask  # noqa: E402


def _seed():
    torch.manual_seed(0)


def _randn(*shape, dtype=torch.float32):
    return torch.randn(*shape, dtype=dtype)


def manual_attention(q, k, v, mask=None, scale=None, tau=None, delta=None):
    """Reference implementation matching FullAttention / DSAttention semantics.

    q: [B, L, H, E]; k: [B, S, H, E]; v: [B, S, H, D].
    mask: None, [L, S], [B, L, S] or 4-D broadcastable (True = masked).
    tau: None or [B, 1]; delta: None or [B, S].
    """
    B, L, H, E = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(E)
    scores = torch.einsum('blhe,bshe->bhls', q, k)
    if tau is not None:
        scores = scores * (tau.unsqueeze(1).unsqueeze(1) if torch.is_tensor(tau) else tau)
    if delta is not None:
        scores = scores + (delta.unsqueeze(1).unsqueeze(1) if torch.is_tensor(delta) else delta)
    scores = scores * scale
    if mask is not None:
        m = mask
        if m.ndim == 2:
            m = m[None, None]
        elif m.ndim == 3:
            m = m[:, None]
        scores = scores.masked_fill(m, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum('bhls,bshd->blhd', attn, v), attn


# ---------------------------------------------------------------------------
# PositionalEmbedding
# ---------------------------------------------------------------------------

def test_positional_embedding_odd_and_even_d_model():
    for d_model in (1, 63, 64):
        pe = PositionalEmbedding(d_model=d_model, max_len=32)
        x = _randn(2, 10, 3)
        out = pe(x)
        assert out.shape == (1, 10, d_model)
        assert torch.isfinite(out).all()
    # buffer key kept for checkpoint compatibility
    assert 'pe' in PositionalEmbedding(d_model=8).state_dict()
    # length limit
    pe = PositionalEmbedding(d_model=8, max_len=16)
    try:
        pe(_randn(2, 17, 3))
        raise AssertionError('expected ValueError for over-long sequence')
    except ValueError:
        pass
    # ndim check
    try:
        pe(_randn(2, 8))
        raise AssertionError('expected ValueError for 2-D input')
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# FullAttention: mask semantics
# ---------------------------------------------------------------------------

def _qkv(B, L, S, H, E, D=None):
    D = D if D is not None else E
    return (_randn(B, L, H, E), _randn(B, S, H, E), _randn(B, S, H, D))


def test_full_attention_matches_manual_formula():
    _seed()
    attn = FullAttention(mask_flag=False, attention_dropout=0.0, output_attention=True).eval()
    q, k, v = _qkv(B=3, L=4, S=7, H=2, E=8)
    out, A = attn(q, k, v, None)
    ref_out, ref_A = manual_attention(q, k, v)
    assert out.shape == (3, 4, 2, 8)
    assert A.shape == (3, 2, 4, 7)
    assert torch.allclose(out, ref_out, atol=1e-6)
    assert torch.allclose(A, ref_A, atol=1e-6)


def test_full_attention_single_batch_single_head():
    _seed()
    attn = FullAttention(mask_flag=False, attention_dropout=0.0).eval()
    q, k, v = _qkv(B=1, L=4, S=7, H=1, E=8)
    out, _ = attn(q, k, v, None)
    ref_out, _ = manual_attention(q, k, v)
    assert out.shape == (1, 4, 1, 8)
    assert torch.allclose(out, ref_out, atol=1e-6)


def test_full_attention_batch_equals_heads_mask_regression():
    """[B, L, S] masks must be applied per SAMPLE, never mis-aligned to heads.

    With B == H == 2 the old right-aligned broadcast silently applied sample 0's
    mask to head 0 of both samples; this test fails on that behavior.
    """
    _seed()
    B = H = 2
    L, S = 3, 5
    attn = FullAttention(mask_flag=False, attention_dropout=0.0, output_attention=True).eval()
    q, k, v = _qkv(B=B, L=L, S=S, H=H, E=8)

    mask = torch.zeros(B, L, S, dtype=torch.bool)
    mask[0, :, 3:] = True   # sample 0: keys 3, 4 masked
    mask[1, :, :2] = True   # sample 1: keys 0, 1 masked

    out, A = attn(q, k, v, mask)
    ref_out, ref_A = manual_attention(q, k, v, mask=mask)
    assert torch.allclose(out, ref_out, atol=1e-6), 'per-sample [B, L, S] mask mis-applied'
    assert torch.allclose(A, ref_A, atol=1e-6)
    # masked keys receive exactly zero weight in every sample and head
    for b in range(B):
        assert torch.all(A[b, :, :, mask[b].all(dim=0)] == 0)
    # each row sums to 1 over the unmasked keys
    assert torch.allclose(A.sum(dim=-1), torch.ones(B, H, L), atol=1e-6)


def test_full_attention_key_padding_mask_4d():
    _seed()
    B, H, L, S = 3, 2, 4, 6
    attn = FullAttention(mask_flag=False, attention_dropout=0.0, output_attention=True).eval()
    q, k, v = _qkv(B=B, L=L, S=S, H=H, E=8)

    pad = torch.zeros(B, 1, 1, S, dtype=torch.bool)
    pad[0, ..., -2:] = True
    pad[1, ..., -1:] = True
    out, A = attn(q, k, v, pad)
    ref_out, ref_A = manual_attention(q, k, v, mask=pad)
    assert torch.allclose(out, ref_out, atol=1e-6)
    assert torch.all(A[pad.broadcast_to(A.shape)] == 0)
    assert torch.allclose(A.sum(dim=-1), torch.ones(B, H, L), atol=1e-6)

    # masked values must not influence the output at all
    v_big = v.clone()
    v_big[0, -2:] = 1e6
    v_big[1, -1:] = -1e6
    out_big, _ = attn(q, k, v_big, pad)
    assert torch.allclose(out, out_big, atol=1e-5), 'masked values leaked into the output'


def test_full_attention_shared_2d_mask():
    _seed()
    B, H, L, S = 3, 2, 4, 6
    attn = FullAttention(mask_flag=False, attention_dropout=0.0).eval()
    q, k, v = _qkv(B=B, L=L, S=S, H=H, E=8)
    mask = torch.zeros(L, S, dtype=torch.bool)
    mask[:, -1] = True
    out, _ = attn(q, k, v, mask)
    ref_out, _ = manual_attention(q, k, v, mask=mask)
    assert torch.allclose(out, ref_out, atol=1e-6)


def test_full_attention_fully_masked_row_raises():
    _seed()
    attn = FullAttention(mask_flag=False, attention_dropout=0.0).eval()
    q, k, v = _qkv(B=2, L=3, S=4, H=2, E=8)
    mask = torch.zeros(2, 1, 1, 4, dtype=torch.bool)
    mask[0] = True  # sample 0 has every key masked
    try:
        attn(q, k, v, mask)
        raise AssertionError('expected ValueError for a fully masked row')
    except ValueError:
        pass
    # a partially masked mask must still work for the other samples
    mask[0, ..., 0] = False
    out, _ = attn(q, k, v, mask)
    assert torch.isfinite(out).all()


def test_full_attention_scale_zero_with_mask():
    _seed()
    attn = FullAttention(mask_flag=False, scale=0.0, attention_dropout=0.0,
                         output_attention=True).eval()
    B, H, L, S = 2, 2, 3, 5
    q, k, v = _qkv(B=B, L=L, S=S, H=H, E=8)
    mask = torch.zeros(B, 1, 1, S, dtype=torch.bool)
    mask[0, ..., 3:] = True
    out, A = attn(q, k, v, mask)
    assert torch.isfinite(out).all() and torch.isfinite(A).all(), 'scale=0 + mask produced NaN/inf'
    # scale == 0 -> uniform attention over the unmasked keys
    assert torch.allclose(A[0, :, :, :3], torch.full((H, L, 3), 1.0 / 3), atol=1e-7)
    assert torch.allclose(A[1], torch.full((H, L, S), 1.0 / S), atol=1e-7)
    # masked values are still excluded
    v_big = v.clone()
    v_big[0, 3:] = 1e6
    out_big, _ = attn(q, k, v_big, mask)
    assert torch.allclose(out, out_big, atol=1e-5)


def test_full_attention_auto_causal_mask():
    _seed()
    attn = FullAttention(mask_flag=True, attention_dropout=0.0).eval()
    q, k, v = _qkv(B=2, L=5, S=5, H=2, E=8)
    out, _ = attn(q, k, v, None)
    causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    ref_out, _ = manual_attention(q, k, v, mask=causal)
    assert torch.allclose(out, ref_out, atol=1e-6)
    # auto causal is only defined for self-attention
    q2, k2, v2 = _qkv(B=2, L=5, S=7, H=2, E=8)
    try:
        attn(q2, k2, v2, None)
        raise AssertionError('expected ValueError for auto causal with L != S')
    except ValueError:
        pass


def test_full_attention_triangular_mask_object():
    _seed()
    attn = FullAttention(mask_flag=False, attention_dropout=0.0).eval()
    q, k, v = _qkv(B=2, L=5, S=5, H=2, E=8)
    out, _ = attn(q, k, v, TriangularCausalMask(2, 5))
    causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    ref_out, _ = manual_attention(q, k, v, mask=causal)
    assert torch.allclose(out, ref_out, atol=1e-6)


def test_full_attention_mask_shape_validation():
    _seed()
    attn = FullAttention(mask_flag=False, attention_dropout=0.0).eval()
    q, k, v = _qkv(B=3, L=4, S=6, H=2, E=8)

    def expect_error(mask, exc_type):
        try:
            attn(q, k, v, mask)
            raise AssertionError(f'expected {exc_type.__name__} for mask {tuple(mask.shape)}')
        except exc_type:
            pass

    expect_error(torch.zeros(4, 6, dtype=torch.bool).new_zeros(6), ValueError)          # 1-D
    expect_error(torch.zeros(3, 6, dtype=torch.bool), ValueError)                       # [B, S] padding mask
    expect_error(torch.zeros(4, 7, dtype=torch.bool), ValueError)                       # wrong S
    expect_error(torch.zeros(3, 5, 6, dtype=torch.bool), ValueError)                    # wrong L
    expect_error(torch.zeros(2, 4, 6, dtype=torch.bool), ValueError)                    # batch=H, not 1 or B
    expect_error(torch.zeros(3, 2, 4, 7, dtype=torch.bool), ValueError)                  # wrong S, 4-D
    expect_error(torch.zeros(3, 2, 4, 6), TypeError)                                     # not bool
    expect_error(torch.zeros(3, 2, 4, 6, 1, dtype=torch.bool), ValueError)               # 5-D
    try:
        attn(q, k, v, torch.zeros(4, 6))  # wrong dtype, given as raw tensor
        raise AssertionError('expected TypeError for non-bool mask')
    except TypeError:
        pass
    # [1, L, S] broadcasts over the batch
    out, _ = attn(q, k, v, torch.zeros(1, 4, 6, dtype=torch.bool))
    ref_out, _ = manual_attention(q, k, v)
    assert torch.allclose(out, ref_out, atol=1e-6)


def test_apply_attention_mask_rejects_non_tensor():
    scores = _randn(2, 2, 3, 4)
    import numpy as np
    for bad in (np.zeros((3, 4), dtype=bool), [[True] * 4] * 3, 'mask'):
        try:
            _apply_attention_mask(scores, bad)
            raise AssertionError('expected TypeError for non-tensor mask')
        except TypeError:
            pass
    try:
        _resolve_attention_mask(True, False, 2, 3, 4, torch.device('cpu'))
        raise AssertionError('expected TypeError for a bool passed as mask')
    except TypeError:
        pass


def test_full_attention_qkv_validation():
    _seed()
    attn = FullAttention(mask_flag=False, attention_dropout=0.0).eval()
    q, k, v = _qkv(B=2, L=3, S=4, H=2, E=8)

    cases = [
        (q[:1], k, v),                    # batch mismatch
        (q, k[:, :3], v),                 # key/value length mismatch
        (q, k[:, :, :1], v),              # head mismatch
    ]
    for qq, kk, vv in cases:
        try:
            attn(qq, kk, vv, None)
            raise AssertionError('expected ValueError for inconsistent Q/K/V')
        except ValueError:
            pass
    # queries and keys must share the per-head dimension
    try:
        attn(q, _randn(2, 4, 2, 4), v, None)
        raise AssertionError('expected ValueError for per-head dim mismatch')
    except ValueError:
        pass
    # values MAY have a different per-head dim than queries/keys
    out, _ = attn(q, k, v[..., :4], None)
    assert out.shape == (2, 3, 2, 4)
    # 2-D queries
    try:
        attn(q[0], k, v, None)
        raise AssertionError('expected ValueError for 2-D queries')
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# DSAttention
# ---------------------------------------------------------------------------

def test_ds_attention_tau_delta_and_mask():
    _seed()
    attn = DSAttention(mask_flag=False, attention_dropout=0.0, output_attention=True).eval()
    B, L, S, H, E = 3, 4, 6, 2, 8
    q, k, v = _qkv(B=B, L=L, S=S, H=H, E=E)
    tau = torch.rand(B, 1) + 0.5
    delta = _randn(B, S)

    out, A = attn(q, k, v, None, tau=tau, delta=delta)
    ref_out, ref_A = manual_attention(q, k, v, tau=tau, delta=delta)
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.allclose(A, ref_A, atol=1e-5)

    # mask + tau/delta together
    mask = torch.zeros(B, 1, 1, S, dtype=torch.bool)
    mask[:, ..., -1] = True
    out, A = attn(q, k, v, mask, tau=tau, delta=delta)
    ref_out, ref_A = manual_attention(q, k, v, mask=mask, tau=tau, delta=delta)
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.all(A[..., -1] == 0)

    # scale == 0 stays finite with a mask applied
    attn0 = DSAttention(mask_flag=False, scale=0.0, attention_dropout=0.0).eval()
    out0, _ = attn0(q, k, v, mask)
    assert torch.isfinite(out0).all()

    # tau=1.0 / delta=0.0 defaults reduce to plain attention
    out_plain, _ = attn(q, k, v, None)
    ref_plain, _ = manual_attention(q, k, v)
    assert torch.allclose(out_plain, ref_plain, atol=1e-6)


def test_ds_attention_tau_delta_shape_validation():
    _seed()
    attn = DSAttention(mask_flag=False, attention_dropout=0.0).eval()
    B, L, S, H, E = 3, 4, 6, 2, 8
    q, k, v = _qkv(B=B, L=L, S=S, H=H, E=E)

    for tau in (torch.rand(B), torch.rand(B, 2), torch.rand(1), 2.0):
        try:
            attn(q, k, v, None, tau=tau)
            raise AssertionError('expected ValueError for a bad tau shape')
        except ValueError:
            pass
    for delta in (_randn(B), _randn(B, S + 1), _randn(B, 1, S), 1.0):
        try:
            attn(q, k, v, None, delta=delta)
            raise AssertionError('expected ValueError for a bad delta shape')
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# ProbAttention
# ---------------------------------------------------------------------------

def test_prob_attention_rejects_explicit_mask():
    _seed()
    for mask_flag in (False, True):
        attn = ProbAttention(mask_flag=mask_flag, attention_dropout=0.0).eval()
        q, k, v = _qkv(B=2, L=4, S=4, H=2, E=8)
        mask = torch.zeros(4, 4, dtype=torch.bool)
        try:
            attn(q, k, v, mask)
            raise AssertionError('expected NotImplementedError for an explicit mask')
        except NotImplementedError:
            pass


def test_prob_attention_shapes_and_singletons():
    _seed()
    attn = ProbAttention(mask_flag=False, attention_dropout=0.0,
                        output_attention=True).eval()
    # self-attention
    q, k, v = _qkv(B=2, L=8, S=8, H=2, E=16)
    out, attns = attn(q, k, v, None)
    assert out.shape == (2, 8, 2, 16)
    assert attns.shape == (2, 2, 8, 8)
    assert torch.isfinite(out).all()
    # cross-attention with L_Q != L_V: the attention map must be [B, H, L_Q, L_V]
    q, k, v = _qkv(B=2, L=4, S=8, H=2, E=16)
    out, attns = attn(q, k, v, None)
    assert out.shape == (2, 4, 2, 16)
    assert attns.shape == (2, 2, 4, 8), (
        f'expected [B, H, L_Q, L_V] = [2, 2, 4, 8], got {tuple(attns.shape)}')
    # singletons: B == H == 1 and length 1 (sampling count must stay >= 1)
    q, k, v = _qkv(B=1, L=1, S=1, H=1, E=4)
    out, attns = attn(q, k, v, None)
    assert out.shape == (1, 1, 1, 4)
    q, k, v = _qkv(B=1, L=2, S=2, H=1, E=4)
    out, _ = attn(q, k, v, None)
    assert out.shape == (1, 2, 1, 4)


# ---------------------------------------------------------------------------
# AttentionLayer
# ---------------------------------------------------------------------------

def test_attention_layer_validation():
    # (args, kwargs) pairs that must all be rejected with a ValueError
    bad_configs = [
        ((FullAttention(), 4, 8), {}),                 # n_heads > d_model -> per-head dim 0
        ((FullAttention(), 16, 0), {}),                # n_heads == 0
        ((FullAttention(), 0, 4), {}),                 # d_model == 0
        ((FullAttention(), 16, 4), {'d_keys': 0}),     # explicit d_keys=0 must NOT
        ((FullAttention(), 16, 4), {'d_values': 0.0}),  # be treated as "not provided"
    ]
    for args, kwargs in bad_configs:
        try:
            AttentionLayer(*args, **kwargs)
            raise AssertionError(f'expected ValueError for {args[1:]} {kwargs}')
        except ValueError:
            pass
    # None defaults derive d_model // n_heads
    layer = AttentionLayer(FullAttention(), d_model=16, n_heads=4)
    assert layer.query_projection.out_features == 16
    layer = AttentionLayer(FullAttention(), d_model=16, n_heads=4, d_keys=8, d_values=4)
    assert layer.query_projection.out_features == 32
    assert layer.value_projection.out_features == 16


def test_attention_layer_forward_matches_manual():
    _seed()
    layer = AttentionLayer(FullAttention(mask_flag=False, attention_dropout=0.0), 32, 4).eval()
    q = _randn(2, 5, 32)
    kv = _randn(2, 7, 32)
    out, attn = layer(q, kv, kv, None)
    assert out.shape == (2, 5, 32)

    qh = layer.query_projection(q).view(2, 5, 4, -1)
    kh = layer.key_projection(kv).view(2, 7, 4, -1)
    vh = layer.value_projection(kv).view(2, 7, 4, -1)
    inner, _ = FullAttention(mask_flag=False, attention_dropout=0.0).eval()(qh, kh, vh, None)
    ref = layer.out_projection(inner.reshape(2, 5, -1))
    assert torch.allclose(out, ref, atol=1e-6)

    # structural checks
    for qq, kk, vv in ((q[:1], kv, kv), (q, kv[:, :6], kv), (q[0], kv, kv)):
        try:
            layer(qq, kk, vv, None)
            raise AssertionError('expected ValueError for inconsistent inputs')
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Reformer / TwoStage: unsupported masks must be refused
# ---------------------------------------------------------------------------

def test_reformer_layer_behavior():
    try:
        from layers.SelfAttention_Family import ReformerLayer
        layer = ReformerLayer(None, 32, 4, bucket_size=2, n_hashes=2)
    except ImportError:
        return  # reformer-pytorch not installed: the lazy import already guards this
    x = _randn(2, 8, 32)
    out, _ = layer(x, x, x, None, None, None)
    assert out.shape == (2, 8, 32)
    try:
        layer(x, x, x, torch.zeros(8, 8, dtype=torch.bool), None, None)
        raise AssertionError('expected NotImplementedError for a Reformer mask')
    except NotImplementedError:
        pass


def test_two_stage_attention_layer_rejects_mask():
    _seed()
    configs = SimpleNamespace(dropout=0.1, factor=3)
    layer = TwoStageAttentionLayer(configs, seg_num=6, factor=3, d_model=32, n_heads=4).eval()
    x = _randn(2, 3, 6, 32)
    out = layer(x)
    assert out.shape == (2, 3, 6, 32)
    try:
        layer(x, attn_mask=torch.zeros(6, 6, dtype=torch.bool))
        raise AssertionError('expected NotImplementedError for a TwoStage mask')
    except NotImplementedError:
        pass


# ---------------------------------------------------------------------------
# TemporalEmbedding
# ---------------------------------------------------------------------------

def test_temporal_embedding_daily_freq_ignores_hour_column():
    _seed()
    emb = TemporalEmbedding(d_model=16, embed_type='fixed', freq='d')

    # 3-column marks [month, day, weekday] must work
    marks3 = torch.stack([
        torch.randint(1, 13, (2, 6)),
        torch.randint(1, 32, (2, 6)),
        torch.randint(0, 7, (2, 6)),
    ], dim=-1)
    out3 = emb(marks3)
    expected = (emb.month_embed(marks3[:, :, 0].long())
                + emb.day_embed(marks3[:, :, 1].long())
                + emb.weekday_embed(marks3[:, :, 2].long()))
    assert torch.allclose(out3, expected, atol=1e-6)

    # a 4th column (hour) must NOT be read for daily freq
    marks4 = torch.cat([marks3, torch.randint(0, 24, (2, 6, 1))], dim=-1)
    out4 = emb(marks4)
    assert torch.allclose(out3, out4, atol=1e-6), 'daily freq read the hour column'

    # too few columns -> clear error
    try:
        emb(marks3[:, :, :2])
        raise AssertionError('expected ValueError for 2-column marks with freq=d')
    except ValueError:
        pass
    # 2-D input -> clear error
    try:
        emb(marks3[0])
        raise AssertionError('expected ValueError for 2-D x_mark')
    except ValueError:
        pass


def test_temporal_embedding_hourly_and_minutely():
    _seed()
    hour_emb = TemporalEmbedding(d_model=16, embed_type='fixed', freq='h')
    marks = torch.stack([
        torch.randint(1, 13, (2, 6)),
        torch.randint(1, 32, (2, 6)),
        torch.randint(0, 7, (2, 6)),
        torch.randint(0, 24, (2, 6)),
    ], dim=-1)
    out = hour_emb(marks)
    expected = (hour_emb.month_embed(marks[:, :, 0].long())
                + hour_emb.day_embed(marks[:, :, 1].long())
                + hour_emb.weekday_embed(marks[:, :, 2].long())
                + hour_emb.hour_embed(marks[:, :, 3].long()))
    assert torch.allclose(out, expected, atol=1e-6)
    try:
        hour_emb(marks[:, :, :3])
        raise AssertionError('expected ValueError for 3-column marks with freq=h')
    except ValueError:
        pass

    minute_emb = TemporalEmbedding(d_model=16, embed_type='fixed', freq='t')
    marks5 = torch.cat([marks, torch.randint(0, 4, (2, 6, 1))], dim=-1)  # minute bucket 0-3
    out = minute_emb(marks5)
    assert torch.allclose(
        out, expected + minute_emb.minute_embed(marks5[:, :, 4].long()), atol=1e-6)
    assert any(key.startswith('minute_embed.') for key in minute_emb.state_dict())
    # freq='s' keeps the legacy 4-field layout
    sec_emb = TemporalEmbedding(d_model=16, embed_type='fixed', freq='s')
    out = sec_emb(marks)
    assert torch.allclose(out, expected, atol=1e-6)


def test_temporal_embedding_rejects_unsupported_freq():
    for freq in ('w', 'm', 'a', 'x'):
        try:
            TemporalEmbedding(d_model=16, embed_type='fixed', freq=freq)
            raise AssertionError(f'expected ValueError for freq={freq!r}')
        except ValueError:
            pass
    # checkpoint key set: hour_embed exists for every supported freq
    for freq in ('d', 'b', 'h', 's', 't'):
        emb = TemporalEmbedding(d_model=16, embed_type='fixed', freq=freq)
        keys = set(emb.state_dict())
        assert 'hour_embed.emb.weight' in keys, f'freq={freq} lost the hour_embed keys'


# ---------------------------------------------------------------------------
# DataEmbedding_inverted
# ---------------------------------------------------------------------------

def test_data_embedding_inverted_seq_len_validation():
    try:
        DataEmbedding_inverted(96.9, 32)
        raise AssertionError('expected TypeError for a non-integer seq_len')
    except TypeError:
        pass
    for bad in (0, -1):
        try:
            DataEmbedding_inverted(bad, 32)
            raise AssertionError('expected ValueError for seq_len <= 0')
        except ValueError:
            pass


def test_data_embedding_inverted_forward():
    _seed()
    emb = DataEmbedding_inverted(96, 32, dropout=0.0).eval()
    assert emb.seq_len == 96
    x = _randn(2, 96, 5)
    out = emb(x)
    assert out.shape == (2, 5, 32)
    ref = torch.nn.functional.linear(x.permute(0, 2, 1), emb.value_embedding.weight,
                                     emb.value_embedding.bias)
    assert torch.allclose(out, ref, atol=1e-6)

    # x_mark concatenation adds one token per time-feature column
    x_mark = _randn(2, 96, 4)
    out = emb(x, x_mark)
    assert out.shape == (2, 9, 32)
    ref = torch.nn.functional.linear(
        torch.cat([x, x_mark], dim=2).permute(0, 2, 1),
        emb.value_embedding.weight, emb.value_embedding.bias)
    assert torch.allclose(out, ref, atol=1e-6)

    # window length mismatch -> clear error
    try:
        emb(_randn(2, 48, 5))
        raise AssertionError('expected ValueError for a wrong window length')
    except ValueError:
        pass
    try:
        emb(_randn(2, 96))
        raise AssertionError('expected ValueError for 2-D x')
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# DataEmbedding / PatchEmbedding smoke (position embedding ordering)
# ---------------------------------------------------------------------------

def test_data_embedding_forward():
    _seed()
    emb = DataEmbedding(c_in=5, d_model=32, embed_type='fixed', freq='h', dropout=0.0).eval()
    x = _randn(2, 96, 5)
    marks = torch.stack([
        torch.randint(1, 13, (2, 96)),
        torch.randint(1, 32, (2, 96)),
        torch.randint(0, 7, (2, 96)),
        torch.randint(0, 24, (2, 96)),
    ], dim=-1)
    out = emb(x, marks)
    assert out.shape == (2, 96, 32)
    assert torch.isfinite(out).all()

    expected = (emb.value_embedding(x)
                + emb.temporal_embedding(marks)
                + emb.position_embedding(emb.value_embedding(x)))
    assert torch.allclose(out, expected, atol=1e-6)

    out = emb(x, None)
    assert out.shape == (2, 96, 32)
    assert torch.allclose(out, emb.value_embedding(x)
                          + emb.position_embedding(emb.value_embedding(x)), atol=1e-6)


def test_patch_embedding_forward():
    _seed()
    emb = PatchEmbedding(d_model=32, patch_len=16, stride=8, padding=4, dropout=0.0).eval()
    x = _randn(2, 3, 96)
    out, n_vars = emb(x)
    assert n_vars == 3
    assert out.shape[0] == 6 and out.shape[-1] == 32
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Pyraformer-style 4-D RegularMask path
# ---------------------------------------------------------------------------

def test_pyraformer_style_regular_mask():
    _seed()
    from layers.Pyraformer_EncDec import RegularMask, get_mask
    mask2d, all_size = get_mask(48, [4, 2], 3)  # [66, 66], True = masked
    B, H, E = 2, 4, 16
    L = mask2d.shape[0]
    q, k, v = _qkv(B=B, L=L, S=L, H=H, E=E)
    mask4d = RegularMask(mask2d.repeat(B, 1, 1)).mask  # [B, 1, L, S]
    attn = FullAttention(mask_flag=True, attention_dropout=0.0, output_attention=True).eval()
    out, A = attn(q, k, v, mask4d)
    assert out.shape == (B, L, H, E)
    assert torch.isfinite(out).all()
    full = mask4d.broadcast_to(A.shape)
    assert torch.all(A[full] == 0), 'Pyraformer pattern mask was not applied'
    assert torch.allclose(A.sum(-1), torch.ones(B, H, L), atol=1e-6)


# ---------------------------------------------------------------------------
# TimeXer end-to-end
# ---------------------------------------------------------------------------

def _timexer_configs(**overrides):
    cfg = dict(
        task_name='long_term_forecast', features='MS', seq_len=96, pred_len=24,
        use_norm=True, patch_len=16, enc_in=7, d_model=64, n_heads=8, e_layers=2,
        d_ff=128, factor=3, dropout=0.1, activation='gelu', embed='timeF', freq='h',
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _timexer_inputs(B, n_ex_vars=6):
    x_enc = _randn(B, 96, 7)
    x_dec = _randn(B, 72, 7)
    return x_enc, x_dec


def test_timexer_eval_consistency_and_checkpoint_roundtrip():
    from models.TimeXer import Model as TimeXer

    model = TimeXer(_timexer_configs()).eval()
    x_enc, x_dec = _timexer_inputs(1)
    single = model(x_enc, None, x_dec, None)

    x_enc2, x_dec2 = _timexer_inputs(2)
    # mixed batch: a duplicated sample must produce the same output in eval mode
    dup = model(torch.cat([x_enc, x_enc]), None, torch.cat([x_dec, x_dec]), None)
    assert torch.allclose(dup[0], single, atol=1e-6)
    assert torch.allclose(dup[1], single, atol=1e-6)
    # a genuinely different sample must (generically) differ
    other = model(x_enc2[:1], None, x_dec2[:1], None)
    assert not torch.allclose(other, single, atol=1e-4)

    # checkpoint save/load roundtrip keeps eval outputs identical
    with tempfile.NamedTemporaryFile(suffix='.pth') as tmp:
        torch.save(model.state_dict(), tmp.name)
        model2 = TimeXer(_timexer_configs()).eval()
        model2.load_state_dict(torch.load(tmp.name))
        reloaded = model2(x_enc, None, x_dec, None)
    assert torch.allclose(reloaded, single, atol=1e-6)


def test_timexer_ex_mask_effectiveness():
    from models.TimeXer import Model as TimeXer

    # use_norm=False so that changing the masked exogenous column cannot leak
    # through the normalization statistics
    model = TimeXer(_timexer_configs(use_norm=False)).eval()
    x_enc, x_dec = _timexer_inputs(2)
    ex_mask = torch.zeros(2, 6, dtype=torch.bool)
    ex_mask[:, 0] = True  # first exogenous variate is missing

    out_masked = model(x_enc, None, x_dec, None, ex_mask=ex_mask)
    assert out_masked.shape == (2, 24, 1)

    # the values of a masked token must not matter
    x_huge = x_enc.clone()
    x_huge[:, :, 0] = 1e6
    out_huge = model(x_huge, None, x_dec, None, ex_mask=ex_mask)
    assert torch.allclose(out_masked, out_huge, atol=1e-5), \
        'masked exogenous values leaked into the forecast'

    # masking must actually change the output (vs. no mask)
    out_free = model(x_enc, None, x_dec, None)
    assert not torch.allclose(out_masked, out_free, atol=1e-5), \
        'ex_mask had no effect'

    # masking every exogenous token is rejected
    try:
        model(x_enc, None, x_dec, None, ex_mask=torch.ones(2, 6, dtype=torch.bool))
        raise AssertionError('expected ValueError when all exogenous tokens are masked')
    except ValueError:
        pass

    # wrong ex_mask shape is rejected
    try:
        model(x_enc, None, x_dec, None, ex_mask=torch.zeros(2, 5, dtype=torch.bool))
        raise AssertionError('expected ValueError for a wrong ex_mask shape')
    except ValueError:
        pass


def test_timexer_rejects_imputation_mask_argument():
    from models.TimeXer import Model as TimeXer

    model = TimeXer(_timexer_configs()).eval()
    x_enc, x_dec = _timexer_inputs(1)
    try:
        model(x_enc, None, x_dec, None, mask=torch.ones(1, 96, 7))
        raise AssertionError('expected NotImplementedError for the imputation-style mask')
    except NotImplementedError:
        pass


def test_timexer_multivariate_features_smoke():
    from models.TimeXer import Model as TimeXer

    model = TimeXer(_timexer_configs(features='M', use_norm=False)).eval()
    x_enc, x_dec = _timexer_inputs(2)
    out = model(x_enc, None, x_dec, None)
    assert out.shape == (2, 24, 7)
    ex_mask = torch.zeros(2, 7, dtype=torch.bool)
    ex_mask[0, -1] = True
    out_m = model(x_enc, None, x_dec, None, ex_mask=ex_mask)
    assert out_m.shape == (2, 24, 7)
    assert not torch.allclose(out_m[0], out[0], atol=1e-5)
    assert torch.allclose(out_m[1], out[1], atol=1e-5)


# ---------------------------------------------------------------------------
# GPU smoke tests (skipped automatically when CUDA is unavailable)
# ---------------------------------------------------------------------------

def _cuda_available():
    return torch.cuda.is_available()


def test_gpu_full_attention_mask_matches_cpu():
    if not _cuda_available():
        return
    _seed()
    B, H, L, S, E = 3, 2, 4, 6, 8
    q, k, v = _qkv(B=B, L=L, S=S, H=H, E=E)
    mask = torch.zeros(B, 1, 1, S, dtype=torch.bool)
    mask[:, ..., -1] = True
    attn = FullAttention(mask_flag=False, attention_dropout=0.0).eval()
    cpu_out, _ = attn(q, k, v, mask)
    gpu_out, _ = attn(q.cuda(), k.cuda(), v.cuda(), mask.cuda())
    assert torch.allclose(cpu_out, gpu_out.cpu(), atol=1e-5)


def test_gpu_informer_forward():
    if not _cuda_available():
        return
    from models.Informer import Model as Informer

    _seed()
    cfg = SimpleNamespace(
        task_name='long_term_forecast', pred_len=24, label_len=48, seq_len=96,
        enc_in=7, dec_in=7, c_out=7, d_model=64, n_heads=8, e_layers=2, d_layers=1,
        d_ff=128, factor=3, dropout=0.1, activation='gelu', distil=True,
        num_class=10, embed='timeF', freq='h',
    )
    model = Informer(cfg).cuda().eval()
    out = model(_randn(2, 96, 7).cuda(), _randn(2, 96, 4).cuda(),
                _randn(2, 72, 7).cuda(), _randn(2, 72, 4).cuda())
    assert out.shape == (2, 24, 7)
    assert torch.isfinite(out).all()


def test_gpu_nonstationary_forward():
    if not _cuda_available():
        return
    from models.Nonstationary_Transformer import Model as Nonstationary

    _seed()
    cfg = SimpleNamespace(
        task_name='long_term_forecast', pred_len=24, label_len=48, seq_len=96,
        enc_in=7, dec_in=7, c_out=7, d_model=64, n_heads=8, e_layers=2, d_layers=1,
        d_ff=128, factor=3, dropout=0.1, activation='gelu', p_hidden_dims=[128, 128],
        p_hidden_layers=2, embed='timeF', freq='h',
    )
    model = Nonstationary(cfg).cuda().eval()
    out = model(_randn(2, 96, 7).cuda(), _randn(2, 96, 4).cuda(),
                _randn(2, 72, 7).cuda(), _randn(2, 72, 4).cuda())
    assert out.shape == (2, 24, 7)
    assert torch.isfinite(out).all()


def test_gpu_pyraformer_mask_and_timexer():
    if not _cuda_available():
        return
    from layers.Pyraformer_EncDec import RegularMask, get_mask

    _seed()
    mask2d, _ = get_mask(48, [4, 2], 3)
    B, H, E = 2, 4, 16
    L = mask2d.shape[0]
    q, k, v = _qkv(B=B, L=L, S=L, H=H, E=E)
    mask = RegularMask(mask2d.repeat(B, 1, 1))
    attn = FullAttention(mask_flag=True, attention_dropout=0.0).eval()
    out, _ = attn(q.cuda(), k.cuda(), v.cuda(), mask.mask.cuda())
    assert torch.isfinite(out).all()

    from models.TimeXer import Model as TimeXer
    model = TimeXer(_timexer_configs(use_norm=False)).cuda().eval()
    x_enc, x_dec = _timexer_inputs(2)
    ex_mask = torch.zeros(2, 6, dtype=torch.bool)
    ex_mask[:, 0] = True
    out = model(x_enc.cuda(), None, x_dec.cuda(), None, ex_mask=ex_mask.cuda())
    assert out.shape == (2, 24, 1)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _run_all():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f'PASS  {name}')
        except Exception:
            failed.append(name)
            print(f'FAIL  {name}')
            traceback.print_exc()
    print(f'\n{len(tests) - len(failed)}/{len(tests)} tests passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_run_all())

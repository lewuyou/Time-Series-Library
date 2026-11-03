# -*- coding: utf-8 -*-
"""TimesNet 适配器：构造、前向形状、参数校验、checkpoint 存取。"""

import argparse

import pytest
import torch

from stock.adapters import get_adapter, adapter_for_checkpoint
from stock.adapters.timesnet_model import Model as TimesNetModel


def _args(**overrides):
    base = dict(
        d_model=16, e_layers=1, n_heads=2, patch_len=5,
        dropout=0.1, ex_mask=False,
        d_ff=None, top_k=5, num_kernels=6, use_norm=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _schema():
    # 与真实协议一致：末列是目标列（TARGET）
    return [
        "px_close_ret", "px_volume_log", "baidu_search",
        "stock_return",
    ]


def _adapter():
    return get_adapter("TimesNet")


def test_registered_with_checkpoint_format():
    adapter = _adapter()
    assert adapter.name == "TimesNet"
    assert adapter.checkpoint_format == "timesnet_mv_v1"


def test_build_model_is_upstream_timesnet():
    config = _adapter().model_config(_args(), 60, _schema())
    model = _adapter().build_model(config)
    assert type(model) is TimesNetModel
    assert model.configs.enc_in == len(_schema())
    assert model.configs.seq_len == 60
    assert model.configs.pred_len == 1
    assert model.configs.top_k == 5
    assert model.configs.d_ff == 32  # 缺省 d_ff = d_model*2
    # 默认协议：不做窗口内归一化（特征保持导出期尺度）
    assert model.use_norm is False


def test_forward_shape_and_finiteness():
    adapter = _adapter()
    config = adapter.model_config(_args(), 60, _schema())
    model = adapter.build_model(config)
    model.eval()
    x = torch.randn(3, 60, len(_schema()))
    with torch.no_grad():
        y = adapter.forward(model, x)
    assert y.shape == (3, 1, 1)
    assert torch.isfinite(y).all()


def test_config_json_reproducible():
    import json
    adapter = _adapter()
    config = adapter.model_config(_args(d_ff=64), 60, _schema())
    clone = json.loads(json.dumps(config))
    assert clone == config
    assert type(adapter.build_model(config)) is type(
        adapter.build_model(clone)
    )


def test_validate_args_rejects_invalid():
    adapter = _adapter()
    # ex_mask 协议显式拒绝（不静默退化）
    with pytest.raises(ValueError, match="ex-mask"):
        adapter.validate_args(_args(ex_mask=True), 60)
    # FFT 可用频段：seq_len=60 → (60+1)//2=30
    with pytest.raises(ValueError, match="频段"):
        adapter.validate_args(_args(top_k=31), 60)
    with pytest.raises(ValueError, match="正数"):
        adapter.validate_args(_args(top_k=0), 60)
    with pytest.raises(ValueError, match="正数"):
        adapter.validate_args(_args(num_kernels=0), 60)
    with pytest.raises(ValueError, match="d-ff"):
        adapter.validate_args(_args(d_ff=0), 60)
    # 边界内合法；seq_len 不必被 top_k 整除（TimesBlock 零填充对齐）
    adapter.validate_args(_args(top_k=30), 60)
    adapter.validate_args(_args(top_k=3), 60)


def test_checkpoint_save_load_predict_consistent(tmp_path):
    adapter = _adapter()
    config = adapter.model_config(_args(), 60, _schema())
    model = adapter.build_model(config)
    model.eval()
    torch.manual_seed(7)
    x = torch.randn(2, 60, len(_schema()))
    with torch.no_grad():
        before = adapter.forward(model, x)

    path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "metadata": {
                "format": adapter.checkpoint_format,
                "model": adapter.name,
                "model_config": config,
            },
            "state_dict": model.state_dict(),
        },
        path,
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)
    # 按 checkpoint 登记反查适配器（predict / predict_daily 的路径）
    loaded_adapter = get_adapter(saved["metadata"]["model"])
    assert adapter_for_checkpoint(
        saved["metadata"]["format"]
    ) is loaded_adapter
    loaded = loaded_adapter.build_model(saved["metadata"]["model_config"])
    loaded.load_state_dict(saved["state_dict"])
    loaded.eval()
    with torch.no_grad():
        after = loaded_adapter.forward(loaded, x)
    assert torch.allclose(before, after, atol=0, rtol=0)


def test_ex_mask_never_registered_in_config():
    adapter = _adapter()
    config = adapter.model_config(_args(), 60, _schema())
    # predictor 按 config["ex_mask"] 决定是否计算掩码；
    # TimesNet 协议不登记该键 → 推理端不会构造掩码。
    assert "ex_mask" not in config

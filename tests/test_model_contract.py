# -*- coding: utf-8 -*-
"""模型适配器契约：配置可复现、输出形状、参数校验、注册表。"""

import argparse

import pytest
import torch

from stock.adapters import get_adapter, list_adapters


def _args(**overrides):
    base = dict(
        d_model=8, e_layers=1, n_heads=2, patch_len=5,
        dropout=0.1, ex_mask=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _schema():
    return [
        "px_close_ret", "px_volume_log", "baidu_search",
        "stock_return",
    ]


def test_timexer_is_registered_and_config_reproducible():
    adapter = get_adapter("TimeXer")
    assert adapter.name == "TimeXer"
    assert "TimeXer" in list_adapters()
    config = adapter.model_config(_args(), 10, _schema())
    assert config["enc_in"] == len(_schema())
    assert config["seq_len"] == 10
    assert config["pred_len"] == 1
    # config 可 JSON 序列化并复现模型（checkpoint 契约的前提）
    import json
    clone = json.loads(json.dumps(config))
    assert clone == config
    model_a = adapter.build_model(config)
    model_b = adapter.build_model(clone)
    assert type(model_a) is type(model_b)


def test_forward_output_contract():
    adapter = get_adapter("TimeXer")
    config = adapter.model_config(_args(), 10, _schema())
    model = adapter.build_model(config)
    x = torch.randn(2, 10, len(_schema()))
    with torch.no_grad():
        y = adapter.forward(model, x)
    assert y.shape == (2, 1, 1)
    assert torch.isfinite(y).all()


def test_forward_with_exogenous_mask():
    adapter = get_adapter("TimeXer")
    config = adapter.model_config(_args(ex_mask=True), 10, _schema())
    model = adapter.build_model(config)
    x = torch.randn(3, 10, len(_schema()))
    ex_mask = torch.zeros(3, len(_schema()) - 1, dtype=torch.bool)
    with torch.no_grad():
        y = adapter.forward(model, x, ex_mask=ex_mask)
    assert y.shape == (3, 1, 1)


def test_unknown_adapter_is_rejected():
    with pytest.raises(ValueError, match="未知模型适配器"):
        get_adapter("NotAModel")


def test_adapter_validates_model_specific_args_early():
    adapter = get_adapter("TimeXer")
    with pytest.raises(ValueError, match="整除"):
        adapter.validate_args(_args(patch_len=4), 10)
    with pytest.raises(ValueError, match="整除|n-heads"):
        adapter.validate_args(_args(d_model=7, n_heads=2), 10)
    adapter.validate_args(_args(), 10)
    assert "d_model" in adapter.cli_params
    assert "batch_size" not in adapter.cli_params

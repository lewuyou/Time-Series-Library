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
        # TimesNet 专属（TimeXer 适配器不读取，多给无害）
        d_ff=None, top_k=5, num_kernels=6, use_norm=False,
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


# ---------- 双模型通用契约（新增模型时把名字加进参数表） ----------

MODEL_NAMES = ["TimeXer", "TimesNet"]


def test_both_models_registered_and_case_insensitive_lookup():
    assert set(list_adapters()) == {"TimeXer", "TimesNet"}
    for name in MODEL_NAMES:
        assert get_adapter(name).name == name
        assert get_adapter(name.lower()).name == name


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_config_reproducible_and_json_serializable(name):
    adapter = get_adapter(name)
    config = adapter.model_config(_args(), 10, _schema())
    assert config["enc_in"] == len(_schema())
    assert config["seq_len"] == 10
    assert config["pred_len"] == 1
    import json
    clone = json.loads(json.dumps(config))
    assert clone == config
    model_a = adapter.build_model(config)
    model_b = adapter.build_model(clone)
    assert type(model_a) is type(model_b)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_output_contract(name):
    adapter = get_adapter(name)
    config = adapter.model_config(_args(), 10, _schema())
    model = adapter.build_model(config)
    x = torch.randn(2, 10, len(_schema()))
    with torch.no_grad():
        y = adapter.forward(model, x)
    assert y.shape == (2, 1, 1)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_cli_params_declare_model_specific_only(name):
    adapter = get_adapter(name)
    assert "batch_size" not in adapter.cli_params
    assert "lr" not in adapter.cli_params
    assert "seed" not in adapter.cli_params
    for param in adapter.cli_params:
        assert isinstance(param, str) and param


def test_timesnet_semantics_documented_in_config():
    """TimesNet：全特征通道、目标末列、无 token 掩码、use_norm 关。"""
    adapter = get_adapter("TimesNet")
    config = adapter.model_config(_args(), 10, _schema())
    assert config["enc_in"] == len(_schema())  # 全特征通道
    assert config["c_out"] == 1
    assert config["pred_len"] == 1
    assert config["top_k"] == 5
    assert config["num_kernels"] == 6
    assert config["use_norm"] is False
    assert config["d_ff"] == _args().d_model * 2  # 缺省 d_ff=d_model*2
    assert "ex_mask" not in config  # 不登记掩码协议
    assert "top_k" in adapter.cli_params
    assert "use_norm" in adapter.cli_params


def test_timesnet_validates_fft_band_and_rejects_ex_mask():
    adapter = get_adapter("TimesNet")
    # seq_len=10 → 可用频段 (10+1)//2=5；top_k=6 必然选中 0 频除零
    with pytest.raises(ValueError, match="频段"):
        adapter.validate_args(_args(top_k=6), 10)
    adapter.validate_args(_args(top_k=5), 10)
    with pytest.raises(ValueError, match="正数"):
        adapter.validate_args(_args(top_k=0), 10)
    with pytest.raises(ValueError, match="正数"):
        adapter.validate_args(_args(num_kernels=0), 10)
    # ex_mask 协议显式拒绝，不静默退化
    with pytest.raises(ValueError, match="ex-mask"):
        adapter.validate_args(_args(ex_mask=True), 10)


def test_timesnet_forward_rejects_non_null_mask_defensively():
    adapter = get_adapter("TimesNet")
    config = adapter.model_config(_args(), 10, _schema())
    model = adapter.build_model(config)
    x = torch.randn(2, 10, len(_schema()))
    mask = torch.zeros(2, len(_schema()) - 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="ex_mask"):
        adapter.forward(model, x, ex_mask=mask)


def test_timesnet_forward_with_use_norm_still_targets_last_channel():
    adapter = get_adapter("TimesNet")
    config = adapter.model_config(_args(use_norm=True), 10, _schema())
    assert config["use_norm"] is True
    model = adapter.build_model(config)
    x = torch.randn(2, 10, len(_schema()))
    with torch.no_grad():
        y = adapter.forward(model, x)
    assert y.shape == (2, 1, 1)  # 反归一化广播扩成 [B,1,C] 后切末列
    assert torch.isfinite(y).all()

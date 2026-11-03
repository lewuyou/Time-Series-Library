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
        # 日历协议开关（P6）：TimeXer 默认 time_marks；契约测试里
        # head 默认关闭，保持既有的"裸 forward 无需 head_calendar"契约；
        # head 开启协议见 test_timexer_head_calendar_protocol。
        calendar_mode="time_marks", head_calendar=False,
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
    with pytest.raises(ValueError, match="calendar_mode"):
        adapter.validate_args(_args(calendar_mode="both"), 10)
    adapter.validate_args(_args(), 10)
    assert "d_model" in adapter.cli_params
    assert "calendar_mode" in adapter.cli_params
    assert "head_calendar" in adapter.cli_params
    assert "batch_size" not in adapter.cli_params


def test_timexer_head_calendar_protocol():
    """head_calendar 开启（默认生产行为）：格式 v2、forward 契约、参数量。"""
    adapter = get_adapter("TimeXer")
    cfg = adapter.model_config(_args(head_calendar=True), 10, _schema())
    assert cfg["head_calendar_dim"] == 5
    assert cfg["calendar_mode"] == "time_marks"
    assert adapter.checkpoint_format_for(cfg) == "global_timexer_typed_v2"
    model = adapter.build_model(cfg)
    x = torch.randn(2, 10, len(_schema()))
    # head 开启时缺 head_calendar 必须显式报错
    with pytest.raises(ValueError, match="head_calendar"):
        adapter.forward(model, x)
    head = torch.zeros(2, 5)
    head[:, 1] = 1.0
    with torch.no_grad():
        y = adapter.forward(model, x, head_calendar=head)
    assert y.shape == (2, 1, 1) and torch.isfinite(y).all()
    # 与同配置 head 关闭的参数量差 = 5（head Linear 多 5 个输入权重）
    cfg_off = dict(cfg, head_calendar_dim=0)
    model_off = adapter.build_model(cfg_off)
    assert adapter.checkpoint_format_for(cfg_off) == "global_timexer_typed_v1"
    diff = sum(p.numel() for p in model.parameters()) - sum(
        p.numel() for p in model_off.parameters()
    )
    assert diff == 5


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
    # TimesNet 只支持 columns 日历模式（time_marks 是 TimeXer 专属通道）
    cols = dict(calendar_mode="columns")
    # seq_len=10 → 可用频段 (10+1)//2=5；top_k=6 必然选中 0 频除零
    with pytest.raises(ValueError, match="频段"):
        adapter.validate_args(_args(top_k=6, **cols), 10)
    adapter.validate_args(_args(top_k=5, **cols), 10)
    with pytest.raises(ValueError, match="正数"):
        adapter.validate_args(_args(top_k=0, **cols), 10)
    with pytest.raises(ValueError, match="正数"):
        adapter.validate_args(_args(num_kernels=0, **cols), 10)
    # ex_mask 协议显式拒绝，不静默退化
    with pytest.raises(ValueError, match="ex-mask"):
        adapter.validate_args(_args(ex_mask=True, **cols), 10)
    # time_marks / head_calendar 显式拒绝
    with pytest.raises(ValueError, match="columns"):
        adapter.validate_args(_args(calendar_mode="time_marks"), 10)
    with pytest.raises(ValueError, match="head-calendar"):
        adapter.validate_args(_args(head_calendar=True, **cols), 10)


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

# -*- coding: utf-8 -*-
"""checkpoint 重载与特征 schema 契约校验。"""

import argparse
from pathlib import Path

import pytest
import torch

from stock.adapters import get_adapter, adapter_for_checkpoint
from stock.data.schema import feature_schema_doc
from stock.inference.predictor import verify_feature_schema


def _adapter_and_config(model_name="TimeXer"):
    adapter = get_adapter(model_name)
    args = argparse.Namespace(
        d_model=8, e_layers=1, n_heads=2, patch_len=5,
        dropout=0.1, ex_mask=False,
        # 日历协议（P6）：本文件主 roundtrip 保持 v1（head 关闭）
        calendar_mode="time_marks", head_calendar=False,
        d_ff=None, top_k=5, num_kernels=6, use_norm=False,
    )
    schema = ["px_close_ret", "px_volume_log", "stock_return"]
    config = adapter.model_config(args, 10, schema)
    return adapter, config, schema


def _saved(schema, model_name="TimeXer"):
    adapter = get_adapter(model_name)
    return {
        "format": adapter.checkpoint_format,
        "model": adapter.name,
        "schema": schema,
        "feature_schema": feature_schema_doc(schema),
        "preprocessing": {},
    }


def _prep(schema):
    return {
        "symbol": "000001",
        "feature_columns": list(schema),
        "feature_schema": {
            key: value
            for key, value in feature_schema_doc(schema).items()
            if key != "columns"
        },
        "statistics": {},
    }


@pytest.mark.parametrize("model_name", ["TimeXer", "TimesNet"])
def test_checkpoint_roundtrip_preserves_output(model_name):
    adapter, config, schema = _adapter_and_config(model_name)
    model = adapter.build_model(config)
    model.eval()
    x = torch.randn(2, 10, len(schema))
    with torch.no_grad():
        before = adapter.forward(model, x)

    path = Path(f"/tmp/tsl_test_checkpoint_{model_name}.pth")
    torch.save({"metadata": {"format": adapter.checkpoint_format},
                "state_dict": model.state_dict()}, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    loaded_adapter = adapter_for_checkpoint(
        saved["metadata"]["format"]
    )
    assert loaded_adapter.name == model_name
    loaded = loaded_adapter.build_model(config)
    loaded.load_state_dict(saved["state_dict"])
    loaded.eval()
    with torch.no_grad():
        after = loaded_adapter.forward(loaded, x)
    assert torch.allclose(before, after, atol=0, rtol=0)
    path.unlink()


def test_checkpoint_roundtrip_v2_head_calendar():
    """head 开启（生产默认）走 v2 格式号：反查适配器并保输出一致。"""
    adapter = get_adapter("TimeXer")
    args = argparse.Namespace(
        d_model=8, e_layers=1, n_heads=2, patch_len=5,
        dropout=0.1, ex_mask=False,
        calendar_mode="time_marks", head_calendar=True,
    )
    schema = ["px_close_ret", "px_volume_log", "stock_return"]
    config = adapter.model_config(args, 10, schema)
    fmt = adapter.checkpoint_format_for(config)
    assert fmt == "global_timexer_typed_v2"

    model = adapter.build_model(config)
    model.eval()
    x = torch.randn(2, 10, len(schema))
    head = torch.zeros(2, 5)
    head[:, 3] = 1.0
    with torch.no_grad():
        before = adapter.forward(model, x, head_calendar=head)

    path = Path("/tmp/tsl_test_checkpoint_v2.pth")
    torch.save({"metadata": {"format": fmt},
                "state_dict": model.state_dict()}, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    loaded_adapter = adapter_for_checkpoint(saved["metadata"]["format"])
    assert loaded_adapter.name == "TimeXer"
    loaded = loaded_adapter.build_model(config)
    loaded.load_state_dict(saved["state_dict"])
    loaded.eval()
    with torch.no_grad():
        after = loaded_adapter.forward(loaded, x, head_calendar=head)
    assert torch.allclose(before, after, atol=0, rtol=0)
    path.unlink()


def test_feature_schema_hash_is_stable_and_order_sensitive():
    schema = ["b", "a", "stock_return"]
    doc = feature_schema_doc(schema)
    assert doc == feature_schema_doc(list(schema))
    assert [c["name"] for c in doc["columns"]] == schema
    assert [c["order"] for c in doc["columns"]] == [0, 1, 2]
    assert doc["hash"] != feature_schema_doc(["a", "b", "stock_return"])["hash"]


def test_strict_mismatch_raises():
    schema = ["px_close_ret", "px_volume_log", "stock_return"]
    saved = _saved(schema)
    prep = _prep(schema)
    verify_feature_schema(saved, prep)

    saved["feature_schema"]["hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        verify_feature_schema(saved, prep)


def test_extra_prep_columns_warn_but_do_not_break_legacy(capsys):
    schema = ["px_close_ret", "px_volume_log", "stock_return"]
    saved = _saved(schema)
    prep = _prep(schema)
    prep["feature_columns"] = prep["feature_columns"] + ["extra_col"]
    prep["feature_schema"] = {
        key: value
        for key, value in feature_schema_doc(
            prep["feature_columns"]
        ).items()
        if key != "columns"
    }
    verify_feature_schema(saved, prep)
    captured = capsys.readouterr()
    assert "schema 之外的列" in captured.out
    assert "extra_col" in captured.out


def test_legacy_artifacts_warn_but_do_not_silently_claim_match(capsys):
    schema = ["px_close_ret", "px_volume_log", "stock_return"]
    saved = _saved(schema)
    prep = _prep(schema)
    del saved["feature_schema"]
    del prep["feature_schema"]
    verify_feature_schema(saved, prep)
    captured = capsys.readouterr()
    assert "缺少 feature_schema" in captured.out
    assert "重算 hash" in captured.out

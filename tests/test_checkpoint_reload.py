# -*- coding: utf-8 -*-
"""checkpoint 重载与特征 schema 契约校验。"""

import argparse
from pathlib import Path

import pytest
import torch

from stock.adapters import get_adapter, adapter_for_checkpoint
from stock.data.schema import feature_schema_doc
from stock.inference.predictor import verify_feature_schema


def _adapter_and_config():
    adapter = get_adapter("TimeXer")
    args = argparse.Namespace(
        d_model=8, e_layers=1, n_heads=2, patch_len=5,
        dropout=0.1, ex_mask=False,
    )
    schema = ["px_close_ret", "px_volume_log", "stock_return"]
    config = adapter.model_config(args, 10, schema)
    return adapter, config, schema


def _saved(schema):
    return {
        "format": "global_timexer_typed_v1",
        "model": "TimeXer",
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


def test_checkpoint_roundtrip_preserves_output():
    adapter, config, schema = _adapter_and_config()
    model = adapter.build_model(config)
    model.eval()
    x = torch.randn(2, 10, len(schema))
    with torch.no_grad():
        before = adapter.forward(model, x)

    path = Path("/tmp/tsl_test_checkpoint.pth")
    torch.save({"metadata": {"format": adapter.checkpoint_format},
                "state_dict": model.state_dict()}, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    loaded_adapter = adapter_for_checkpoint(
        saved["metadata"]["format"]
    )
    loaded = loaded_adapter.build_model(config)
    loaded.load_state_dict(saved["state_dict"])
    loaded.eval()
    with torch.no_grad():
        after = loaded_adapter.forward(loaded, x)
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

    saved["feature_schema"] = feature_schema_doc(schema)
    prep["feature_columns"] = prep["feature_columns"] + ["extra_col"]
    prep["feature_schema"] = {
        key: value
        for key, value in feature_schema_doc(
            prep["feature_columns"]
        ).items()
        if key != "columns"
    }
    with pytest.raises(ValueError, match="之外"):
        verify_feature_schema(saved, prep)


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

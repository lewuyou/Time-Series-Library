# -*- coding: utf-8 -*-
"""设备策略：训练默认 GPU，仅显式 --device cpu 才用 CPU；
请求 cuda 而不可用时必须报错，禁止静默降级。

历史教训：wf_orchestrate 曾硬编码 device="cpu"，叠加 runner auto
的静默回退，使整套 walk-forward 精评在 CPU 上运行而无人察觉。
"""

import pytest
import torch

from stock.training import runner


def test_parser_default_device_is_cuda():
    args = runner.build_train_parser().parse_args([])
    assert args.device == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="无 CUDA 环境")
def test_resolve_device_cuda_and_auto():
    assert runner.resolve_device("cuda").type == "cuda"
    assert runner.resolve_device("auto").type == "cuda"


def test_resolve_device_explicit_cpu():
    assert runner.resolve_device("cpu").type == "cpu"


def test_resolve_device_no_silent_fallback(monkeypatch):
    """cuda 不可用时：请求 cuda/auto 必须报错，不得静默降级 CPU。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        runner.resolve_device("cuda")
    with pytest.raises(RuntimeError):
        runner.resolve_device("auto")
    # 显式 cpu 不受影响
    assert runner.resolve_device("cpu").type == "cpu"


def test_resolve_device_rejects_unknown():
    with pytest.raises(ValueError):
        runner.resolve_device("tpu")

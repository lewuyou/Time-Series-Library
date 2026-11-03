import pytest
import torch

from stock.training.trainer import per_sample_loss


def test_per_sample_loss_mse():
    diff = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
    loss = per_sample_loss(diff, "mse")
    assert torch.allclose(loss, torch.tensor([1.0, 0.25]))


def test_per_sample_loss_huber_matches_manual():
    diff = torch.tensor([[0.5], [2.0]])
    loss = per_sample_loss(diff, "huber", huber_delta=1.0)
    # |d|<=delta: 0.5*d^2=0.125；|d|>delta: delta*(|d|-0.5*delta)=1.5
    assert torch.allclose(loss, torch.tensor([0.125, 1.5]))


def test_per_sample_loss_huber_downweights_outlier():
    diff = torch.tensor([[10.0]])
    huber = per_sample_loss(diff, "huber", huber_delta=1.0)
    mse = per_sample_loss(diff, "mse")
    assert huber < mse


def test_per_sample_loss_rejects_unknown():
    with pytest.raises(ValueError):
        per_sample_loss(torch.zeros(2, 1), "l1")

"""训练产物登记辅助：运行环境、首个预测时刻。"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def prediction_time(spec):
    """首个测试预测时刻：test_start 的 decision_hour:minute（Asia/Shanghai）。"""
    ts = pd.Timestamp(spec["test_start"])
    ts = ts + pd.Timedelta(
        hours=int(spec.get("decision_hour", 8)),
        minutes=int(spec.get("decision_minute", 30)),
    )
    return ts.tz_localize("Asia/Shanghai")


def environment_info(device):
    """训练产物登记运行环境（版本/设备），便于复现性审计。

    运行环境的墙钟性能不影响预测口径，只用于追溯训练条件。
    """
    import platform

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        import duckdb

        info["duckdb"] = duckdb.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        if commit:
            info["git_commit"] = commit
    except Exception:
        pass
    return info

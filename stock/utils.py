"""通用小工具（全仓唯一实现，替代原先散落在各脚本的重复版本）。

- read_json / write_json：原子写（先写 .tmp 再替换），indent=2、
  ensure_ascii=False、allow_nan=False，与旧 dataset.json_write /
  train_global_timexer.write_json 输出格式一致；
- sha256_file：与旧 dataset.sha256_file 相同实现；
- date_text：任意时间值 → 'YYYY-MM-DD'（walk-forward 折划分用）；
- now_text：当前上海时区时间戳（训练产物登记用）。
"""

import hashlib
import json
from pathlib import Path

import pandas as pd


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    """原子写 JSON：写 .tmp 后替换，避免进程中断留下半个文件。"""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def date_text(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def now_text():
    """当前时间戳（Asia/Shanghai，tz-aware）。"""
    return pd.Timestamp.now(tz="Asia/Shanghai")

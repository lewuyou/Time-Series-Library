"""模型适配器：连接股票数据契约与 models/ 模型库。

不同时序模型的输入输出协议各不相同（是否需要 x_mark/decoder 输入、
是否支持外生变量与缺失掩码、输出形状等）。适配器把这些差异封装在
各自模块中，对训练器/推理器暴露统一接口：

    adapter.model_config(args, seq_len, schema)  → 模型构造配置
    adapter.build_model(config)                  → nn.Module
    adapter.forward(model, x, ex_mask)           → [B, 1, 1] 预测
    adapter.checkpoint_format                    → checkpoint 登记格式号

统一训练入口 train_stock.py 通过 registry 按名字取适配器；
新增模型 = 新增一个适配器模块并注册，不复制训练脚本。
"""

from stock.adapters.base import ModelAdapter
from stock.adapters.registry import (
    adapter_for_checkpoint,
    get_adapter,
    list_adapters,
    register_adapter,
)

# 导入即注册内置适配器（TimeXer 等）；新增适配器在此追加一行。
from stock.adapters import timexer  # noqa: F401
from stock.adapters import timesnet  # noqa: F401

__all__ = [
    "ModelAdapter",
    "adapter_for_checkpoint",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]

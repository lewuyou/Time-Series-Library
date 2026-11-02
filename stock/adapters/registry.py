"""适配器注册表：按模型名 / checkpoint 格式号取适配器。"""

from stock.adapters.base import ModelAdapter

_REGISTRY = {}
_FORMAT_INDEX = {}


def register_adapter(adapter):
    """登记适配器实例（同名 / 同格式号重复登记视为代码错误）。"""
    if not adapter.name or not adapter.checkpoint_format:
        raise ValueError("适配器必须定义 name 与 checkpoint_format")
    if adapter.name in _REGISTRY:
        raise ValueError(f"适配器重名：{adapter.name}")
    if adapter.checkpoint_format in _FORMAT_INDEX:
        raise ValueError(
            f"checkpoint 格式号冲突：{adapter.checkpoint_format}"
        )
    _REGISTRY[adapter.name] = adapter
    _FORMAT_INDEX[adapter.checkpoint_format] = adapter
    return adapter


def get_adapter(name):
    """按模型名取适配器（train_stock.py --model）。"""
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "<无>"
        raise ValueError(
            f"未知模型适配器：{name}（可用：{available}）"
        ) from None


def adapter_for_checkpoint(fmt, default=None):
    """按 checkpoint 登记的 format 字段反查适配器。

    旧 checkpoint 无 format 字段时回退 default（None 则报错）。
    """
    adapter = _FORMAT_INDEX.get(fmt)
    if adapter is not None:
        return adapter
    if default is not None and fmt is None:
        return get_adapter(default)
    known = ", ".join(sorted(_FORMAT_INDEX)) or "<无>"
    raise ValueError(
        f"不支持的 checkpoint 格式：{fmt}（已知：{known}）"
    )


def list_adapters():
    return sorted(_REGISTRY)

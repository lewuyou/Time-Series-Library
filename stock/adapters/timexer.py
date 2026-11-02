"""TimeXer 适配器：原 train_global_timexer.py 的模型构造原样迁出。

- TypedExogenousEmbedding：官方外生 embedding 上叠加变量身份编码；
- build_model：构造 TimeXer 并替换 ex_embedding；
- checkpoint 格式号保持 "global_timexer_typed_v1"，与既有产物互认。
"""

import argparse

import torch
from torch import nn

from models.TimeXer import Model as TimeXer

from stock.adapters.base import ModelAdapter
from stock.adapters.registry import register_adapter


class TypedExogenousEmbedding(nn.Module):
    """在官方外生 embedding 上增加变量身份编码。"""

    def __init__(self, original, feature_count, d_model):
        super().__init__()
        self.original = original
        self.feature_id = nn.Parameter(
            torch.empty(1, feature_count, d_model)
        )
        nn.init.normal_(self.feature_id, mean=0.0, std=0.02)

    def forward(self, x, x_mark=None):
        if x_mark is not None:
            raise ValueError("本实现不使用额外时间标记 token")
        embedded = self.original(x, None)
        if embedded.shape[1] != self.feature_id.shape[1]:
            raise ValueError("外生 token 数与 schema 不一致")
        return embedded + self.feature_id


def build_model(config):
    model = TimeXer(argparse.Namespace(**config)).float()
    model.ex_embedding = TypedExogenousEmbedding(
        model.ex_embedding,
        config["enc_in"] - 1,
        config["d_model"],
    )
    return model


class TimeXerAdapter(ModelAdapter):
    name = "TimeXer"
    checkpoint_format = "global_timexer_typed_v1"
    # 模型专属参数：调度器按此声明组装命令，禁止无条件透传。
    cli_params = (
        "d_model", "e_layers", "n_heads", "patch_len", "dropout", "ex_mask",
    )

    def validate_args(self, args, seq_len):
        if args.patch_len < 1 or seq_len % args.patch_len:
            raise ValueError(
                "patch-len 必须为正数且整除 seq-len（TimeXer 内生 patch 从"
                "窗口起点切，非整除会丢弃最新的尾部目标历史）"
            )
        if args.d_model < 1 or args.n_heads < 1:
            raise ValueError("d-model / n-heads 必须为正数")
        if args.d_model % args.n_heads:
            raise ValueError("d-model 必须能被 n-heads 整除")

    def model_config(self, args, seq_len, schema):
        return {
            "task_name": "long_term_forecast",
            "features": "MS",
            "seq_len": seq_len,
            "pred_len": 1,
            "label_len": 0,
            "enc_in": len(schema),
            "c_out": 1,
            "d_model": args.d_model,
            "d_ff": args.d_model * 2,
            "e_layers": args.e_layers,
            "n_heads": args.n_heads,
            "factor": 1,
            "dropout": args.dropout,
            "embed": "fixed",
            "freq": "d",
            "activation": "gelu",
            # 保留已导出的尺度与 0/1 缺失标记，不再做窗口内归一化。
            "use_norm": 0,
            "patch_len": args.patch_len,
            # 外生 token 掩码协议开关（训练/验证/测试/推理必须一致）：
            # True = 屏蔽窗口内整段无观测的外生值 token。登记进
            # checkpoint，predict / predict_daily 读取后自动跟随。
            "ex_mask": bool(args.ex_mask),
        }

    def build_model(self, config):
        return build_model(config)

    def forward(self, model, x, ex_mask=None):
        return model(x, None, None, None, ex_mask=ex_mask)


register_adapter(TimeXerAdapter())

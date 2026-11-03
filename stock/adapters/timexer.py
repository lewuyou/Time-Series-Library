"""TimeXer 适配器：原 train_global_timexer.py 的模型构造原样迁出。

- TypedExogenousEmbedding：官方外生 embedding 上叠加变量身份编码；
  x_mark（时间标记通道）拼接在值列之后，每个外生 token 拥有独立的
  身份编码（feature_id），列顺序由 schema hash 固定；
- build_model：构造 TimeXer 并替换 ex_embedding；
- checkpoint 格式号：默认 "global_timexer_typed_v1"；
  预测头拼接目标日星期 one-hot（head_calendar_dim > 0，默认开启）时
  用 "global_timexer_typed_v2"（head Linear 输入维变化，不可复用旧号）。
"""

import argparse

import torch
from torch import nn

from models.TimeXer import Model as TimeXer

from stock.adapters.base import ModelAdapter
from stock.adapters.registry import register_adapter
from stock.data.schema import is_calendar_col, is_mark_col

#: calendar_mode 合法取值（训练/验证/测试/推理必须一致，登记进
#: checkpoint，推理端按 checkpoint 复现，不接受命令行覆盖）。
#: time_marks（默认，官方通道）：3 个时间标记列移入 x_mark；
#: columns：全部日历列留在输入列；none：关闭全部日历信息。
CALENDAR_MODES = ("time_marks", "columns", "none")

#: 预测头日历向量维数 = 目标日星期 one-hot（周一~周五）。
HEAD_CALENDAR_DIM = 5


class TypedExogenousEmbedding(nn.Module):
    """在官方外生 embedding 上增加变量身份编码。

    feature_count = x_enc 外生 token 数 + 时间标记 token 数
    （model_config 的 calendar_mark_dim）。token 顺序恒为
    "x_enc 外生列（schema 顺序）在前、x_mark 列（schema 顺序）在后"。
    每个 token 的身份由 feature_id 显式编码（官方实现没有这一层，
    官方 x_mark 各列共享同一投影）；因此"交换周一/周二列输出不变"
    的担忧在本仓库不成立（见 stock_timexer/CALENDAR_DESIGN.md）。
    """

    def __init__(self, original, feature_count, d_model):
        super().__init__()
        self.original = original
        self.feature_id = nn.Parameter(
            torch.empty(1, feature_count, d_model)
        )
        nn.init.normal_(self.feature_id, mean=0.0, std=0.02)

    def forward(self, x, x_mark=None):
        embedded = self.original(x, x_mark)
        if embedded.shape[1] != self.feature_id.shape[1]:
            raise ValueError("外生 token 数与 schema 不一致")
        return embedded + self.feature_id


def build_model(config):
    model = TimeXer(argparse.Namespace(**config)).float()
    mark_dim = int(config.get("calendar_mark_dim", 0) or 0)
    model.ex_embedding = TypedExogenousEmbedding(
        model.ex_embedding,
        config["enc_in"] - 1 + mark_dim,
        config["d_model"],
    )
    return model


class TimeXerAdapter(ModelAdapter):
    name = "TimeXer"
    checkpoint_format = "global_timexer_typed_v1"
    # 预测头拼接目标日日历向量（head_calendar_dim > 0）的 checkpoint
    # 使用 v2 格式号，仍由本适配器管辖。
    extra_formats = ("global_timexer_typed_v2",)
    # 模型专属参数：调度器按此声明组装命令，禁止无条件透传。
    cli_params = (
        "d_model", "e_layers", "n_heads", "patch_len", "dropout", "ex_mask",
        "calendar_mode", "head_calendar",
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
        mode = getattr(args, "calendar_mode", None) or "time_marks"
        if mode not in CALENDAR_MODES:
            raise ValueError(f"不支持的 calendar_mode：{mode}")

    def model_config(self, args, seq_len, schema):
        mode = getattr(args, "calendar_mode", None) or "time_marks"
        head_dim = (
            HEAD_CALENDAR_DIM if bool(getattr(args, "head_calendar", True))
            else 0
        )
        # 日历列由 load_catalog 固定排在外生列末尾（紧邻 TARGET）。
        ex_columns = schema[:-1]
        calendar_columns = [c for c in ex_columns if is_calendar_col(c)]
        mark_columns = [c for c in calendar_columns if is_mark_col(c)]
        if mode == "time_marks":
            # 时间标记列移入 x_mark（官方通道），旗标列仍留在 x_enc。
            enc_in = len(schema) - len(mark_columns)
            mark_dim = len(mark_columns)
        elif mode == "none":
            # 关闭全部日历信息（A/B 基线）。
            enc_in = len(schema) - len(calendar_columns)
            mark_dim = 0
        else:  # columns
            enc_in = len(schema)
            mark_dim = 0
        # head_calendar（目标日星期 one-hot）由数据集按目标日期直接
        # 推导，与日历列开关无关：三种 calendar_mode 均可独立开/关，
        # A/B 时可单独关掉任一方。

        config = {
            "task_name": "long_term_forecast",
            "features": "MS",
            "seq_len": seq_len,
            "pred_len": 1,
            "label_len": 0,
            "enc_in": enc_in,
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
            # 目标日日历通道协议（与 stock.data.panel.PanelWindows 的
            # calendar_mode 一致）：time_marks（默认，官方通道）= 时间
            # 标记列移入 x_mark；columns = 全部日历列留在输入列；
            # none = 关闭全部日历信息。登记进 checkpoint，推理端复现。
            "calendar_mode": mode,
            "calendar_columns": calendar_columns,
            "calendar_mark_columns": (
                mark_columns if mode == "time_marks" else []
            ),
            "calendar_mark_dim": mark_dim,
            # 预测头日历拼接（默认开启）：目标日星期 one-hot（5 维）
            # 拼到 FlattenHead 展平表示后再 Linear。恒写入（含 0）。
            "head_calendar_dim": head_dim,
        }
        return config

    def checkpoint_format_for(self, model_config):
        if int(model_config.get("head_calendar_dim", 0) or 0) > 0:
            return "global_timexer_typed_v2"
        return self.checkpoint_format

    def build_model(self, config):
        return build_model(config)

    def forward(self, model, x, ex_mask=None, x_mark=None,
                head_calendar=None):
        mark = x_mark if (x_mark is not None and x_mark.shape[-1] > 0) else None
        head_dim = int(getattr(model, "head_calendar_dim", 0) or 0)
        if head_dim > 0:
            # 目标日星期 one-hot 由数据集/推理器按目标日期直接推导
            # （与日历列开关无关），调用方必须提供。
            if head_calendar is None:
                raise ValueError(
                    "head_calendar_dim > 0 需要 head_calendar"
                    "（目标日星期 one-hot）"
                )
            if head_calendar.shape[-1] != head_dim:
                raise ValueError(
                    f"head_calendar 维数 {head_calendar.shape[-1]} "
                    f"!= head_calendar_dim {head_dim}"
                )
        else:
            head_calendar = None
        return model(
            x, mark, None, None, ex_mask=ex_mask,
            head_calendar=head_calendar,
        )


register_adapter(TimeXerAdapter())

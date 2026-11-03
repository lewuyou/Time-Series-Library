"""TimesNet 适配器：多变量通道级接入（无内生/外生 token 区分）。

输入语义（与 TimeXer 的本质差异，训练/推理/回测必须一致理解）：
- TimesNet 没有 TimeXer 那种内生 patch token / 外生 value token 的
  结构。它把整个窗口张量 x [B, L, n_schema] 当作同质的多变量序列，
  全特征通道一起读入（DataEmbedding 按通道线性嵌入），包括：
    * 目标列（schema 末列，TARGET）；
    * 全部外生值列；
    * 全部 __missing 0/1 缺失标记列与年龄列。
- 目标通道＝schema 末列：模型输出经 projection 得 c_out=1 通道；
  use_norm=1 时反归一化按通道统计广播会把输出扩成 [B, 1, C]，
  适配器固定切末列 → [B, 1, 1]（与上游 MS 模式 f_dim=-1 的取法
  一致）；use_norm=0 时输出本就是 [B, 1, 1]，切片为无操作。
- use_norm：默认 0（与 TimeXer 协议一致——特征已按导出期统计量
  标准化，且 0/1 缺失标记通道经窗口内均值/方差归一化会被破坏）。
  可用 --use-norm 开启（拷贝的 timesnet_model.py 已加开关），但
  开启时缺失标记语义被污染，只建议作消融对照，不作默认协议。
- ex_mask：不支持。TimesNet 没有 token 级掩码机制，缺失信息只能
  靠 __missing 标记通道本身参与训练。--ex-mask 开启会在
  validate_args 显式拒绝（不得静默退化）；forward 收到非空掩码
  也直接报错（纵深防御）。
- "MS 单目标 + 外生" 这种 TimeXer 式区分在本数据协议中不存在
  （数据集恒输出全 schema 通道），因此无需按模式拒绝：TimesNet
  的合法输入只有"全特征通道"一种，其余差异见上。

超参含义：
- enc_in = len(schema)：输入通道数（含目标列与缺失标记列）；
- seq_len：窗口长度（数据集给定的 seq_len，非 CLI 参数）；
- pred_len = 1（固定：次日收益预测）；
- top_k：FFT 周期检测保留的主周期个数（每层 TimesBlock 取前
  top_k 个振幅最大的频率做 2D 建模）；约束见 validate_args；
- num_kernels：Inception 卷积核尺寸集合大小（固定 6 即上游默认）；
- d_model：通道嵌入后的隐层宽度；d_ff：TimesBlock 内 Inception
  卷积的中间宽度（默认 d_model*2）；e_layers：TimesBlock 层数。

checkpoint 格式号 "timesnet_mv_v1"（multivariate 全通道协议）。
"""

import argparse

import torch

from stock.adapters.timesnet_model import Model as TimesNet

from stock.adapters.base import ModelAdapter
from stock.adapters.registry import register_adapter


class TimesNetAdapter(ModelAdapter):
    name = "TimesNet"
    checkpoint_format = "timesnet_mv_v1"
    # 模型专属参数：调度器按此声明组装命令，禁止无条件透传。
    cli_params = (
        "d_model", "d_ff", "e_layers", "top_k", "num_kernels",
        "dropout", "use_norm",
    )

    def validate_args(self, args, seq_len):
        if getattr(args, "ex_mask", False):
            raise ValueError(
                "TimesNet 不支持 --ex-mask 外生 token 掩码协议（无 "
                "token 结构，缺失信息只能靠 __missing 标记通道本身"
                "参与训练）；请勿对 TimesNet 开启该开关"
            )
        top_k = getattr(args, "top_k", 5)
        num_kernels = getattr(args, "num_kernels", 6)
        d_ff = getattr(args, "d_ff", None)
        if top_k < 1 or num_kernels < 1:
            raise ValueError("top-k / num-kernels 必须为正数")
        if d_ff is not None and d_ff < 1:
            raise ValueError("d-ff 必须为正数")
        # FFT 周期检测的可用频段：predict_linear 先把时间维对齐到
        # T = seq_len + pred_len(=1)，rfft 有 T//2+1 个频率 bin，
        # bin 0（直流分量）被置零；top_k 超过 T//2 时 topk 必然
        # 选中 0 频导致 period = T // 0 除零，必须在启动前拒绝。
        usable = (seq_len + 1) // 2
        if top_k > usable:
            raise ValueError(
                f"top-k={top_k} 超过 FFT 可用频段上限 (seq_len+1)//2"
                f"={usable}（seq_len={seq_len}），0 频被置零后不足 "
                f"top_k 个候选周期"
            )
        # seq_len 无需被 top_k 整除：TimesBlock 对每个检测周期做
        # 零填充对齐，非整除只影响填充量，不影响形状正确性。

    def model_config(self, args, seq_len, schema):
        return {
            "task_name": "long_term_forecast",
            "features": "MS",
            "seq_len": seq_len,
            "label_len": 0,
            "pred_len": 1,
            "enc_in": len(schema),
            "c_out": 1,
            "d_model": args.d_model,
            "d_ff": getattr(args, "d_ff", None) or args.d_model * 2,
            "e_layers": args.e_layers,
            "top_k": getattr(args, "top_k", 5),
            "num_kernels": getattr(args, "num_kernels", 6),
            "embed": "fixed",
            "freq": "d",
            "dropout": args.dropout,
            # 与 TimeXer 一致：特征已按导出期尺度保存，不再做窗口内
            # 归一化（--use-norm 可开启，仅作消融对照，见模块 docstring）。
            "use_norm": bool(getattr(args, "use_norm", False)),
        }

    def build_model(self, config):
        return TimesNet(argparse.Namespace(**config)).float()

    def forward(self, model, x, ex_mask=None):
        if ex_mask is not None:
            raise ValueError(
                "TimesNet 不支持 ex_mask（训练时 --ex-mask 已在 "
                "validate_args 拒绝；checkpoint 协议也不会登记掩码）"
            )
        out = model(x, None, None, None)
        # use_norm=1 时反归一化广播会把输出扩成 [B, 1, C]；目标固定
        # 为 schema 末列，切末列统一成 [B, 1, 1]（use_norm=0 时无操作）。
        return out[..., -1:]


register_adapter(TimesNetAdapter())

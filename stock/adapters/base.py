"""适配器基类：定义训练器/推理器依赖的统一接口。"""

import abc


class ModelAdapter(abc.ABC):
    """一个适配器对应 models/ 中的一种模型在股票任务上的接入方式。

    约定：
    - model_config 产出的 config 同时用于构造模型与登记进 checkpoint
      （build_model(config) 必须可从 checkpoint 里的 config 复现模型）；
    - forward 输出 [B, pred_len, 1]，pred_len 当前为 1；
    - checkpoint_format 用于 predict / predict_daily 反查适配器，
      格式号一旦发布不可变更语义。
    """

    #: 适配器名（train_stock.py --model 参数值）
    name = ""
    #: checkpoint 登记格式号（"format" 字段）
    checkpoint_format = ""
    #: 该模型专属的 CLI 参数名；walk_forward / train_stock 只透传这些。
    #: 通用参数（batch-size/lr/device 等）不属于任何模型。
    cli_params = ()

    def validate_args(self, args, seq_len):
        """在启动昂贵流程前校验模型专属参数（默认无约束）。"""
        return None

    @abc.abstractmethod
    def model_config(self, args, seq_len, schema):
        """由 CLI 参数 + 数据 schema 构造模型配置 dict。"""

    @abc.abstractmethod
    def build_model(self, config):
        """由 config 构造模型（可从 checkpoint 复现）。"""

    @abc.abstractmethod
    def forward(self, model, x, ex_mask=None, x_mark=None,
                head_calendar=None):
        """标准前向：x [B, L, n_schema] → 预测 [B, 1, 1]。

        ex_mask 为外生 token 有效性掩码（数据集恒产出，是否传入模型
        由训练协议决定）；不支持掩码的模型应在能力说明中注明并在
        忽略时显式登记，不得静默假装支持。
        x_mark 为可选的时间标记通道 [B, L, mark_dim]（目标日语义）；
        不支持该通道的模型必须在非 None 输入时显式报错，不得静默忽略。
        head_calendar 为目标日星期 one-hot [B, 5]（数据集恒产出）；
        仅当模型配置 head_calendar_dim > 0 时使用，不支持预测头日历
        拼接的模型必须在配置开启时显式报错。
        """

    def checkpoint_format_for(self, model_config):
        """给定 model_config 的 checkpoint 格式号。

        默认返回静态 checkpoint_format；格式号随配置变化的适配器
        （如预测头扩展）覆写本方法。
        """
        return self.checkpoint_format

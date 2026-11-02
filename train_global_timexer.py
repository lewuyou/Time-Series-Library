# -*- coding: utf-8 -*-
"""兼容转发：原实现已拆分至 stock/ 包（行为不变）。

迁移映射：
    训练/预测/refit 主流程   -> stock/training/runner.py（新入口 train_stock.py）
    训练循环 / 测试集装载    -> stock/training/trainer.py
    特征列布局与张量对齐     -> stock/data/schema.py（单一事实源）
    json / 时间 / 哈希工具   -> stock/utils.py
    TimeXer 构造（可替换）   -> stock/adapters/timexer.py
    评估与测试报告           -> stock/evaluation/metrics.py

推荐新用法（与本脚本完全等价，--model 可选其他已注册模型）：

    python train_stock.py --mode train --data-root ... --out-dir ...

本文件保留旧命令行用法与 `import train_global_timexer` 的符号兼容。
"""

from stock.adapters.timexer import build_model
from stock.data.panel import PanelWindows, load_catalog
from stock.data.schema import (
    TARGET,
    align_features,
    base_name,
    compute_ex_mask,
    exogenous_mask_layout,
)
from stock.evaluation.metrics import metrics, save_test_report
from stock.training.artifacts import environment_info, prediction_time
from stock.training.runner import (
    build_train_parser,
    main,
    train,
    train_refit,
)
from stock.training.trainer import run_epoch, test_predictions
from stock.utils import now_text, read_json, write_json

if __name__ == "__main__":
    main()

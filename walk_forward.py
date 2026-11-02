# -*- coding: utf-8 -*-
"""兼容转发：walk-forward 滚动评估已迁移至 stock/evaluation/walk_forward.py。

新用法（推荐）：

    python -m stock.evaluation.walk_forward --db ... --run-root ...

迁移说明：内部训练子进程改调 train_stock.py（--model TimeXer，
与原 train_global_timexer.py 训练协议一致）；评估报告复用
stock/evaluation/metrics.save_test_report；折边界与两阶段 refit
协议、产物格式均不变。

旧命令 `python walk_forward.py ...` 继续可用（本文件转发）。
"""

from stock.evaluation.walk_forward import main
from stock.utils import date_text  # noqa: F401  旧模块级名字兼容

if __name__ == "__main__":
    main()

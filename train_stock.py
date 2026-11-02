# -*- coding: utf-8 -*-
"""统一股票模型训练入口（原 train_global_timexer.py 的替代）。

用法与原脚本一致，新增 --model 选择模型（经适配器注册表）：

    python train_stock.py --mode train \
        --data-root dataset/global_001 \
        --out-dir checkpoints/timexer_001 \
        --epochs 30 --patience 5 --batch-size 256 --workers 0 \
        --d-model 64 --e-layers 2 --patch-len 10 --device auto

    python train_stock.py --mode train --model TimeXer ...   # 等价（默认）

完整训练协议（早停 / 两阶段 refit / 产物登记）见
stock/training/runner.py 与 stock/training/trainer.py。
"""

from stock.training.runner import main

if __name__ == "__main__":
    main()

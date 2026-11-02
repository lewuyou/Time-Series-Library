# -*- coding: utf-8 -*-
"""兼容转发：每日选股预测已迁移至 stock/production/daily_pipeline.py。

新用法（推荐）：

    python -m stock.production.daily_pipeline \
        --checkpoint ... --data-root ... --db ... --top 10

旧命令 `python predict_daily.py ...` 继续可用（本文件转发）。
"""

from stock.production.daily_pipeline import main

if __name__ == "__main__":
    main()

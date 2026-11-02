# -*- coding: utf-8 -*-
"""兼容转发：逐日命中率评估已迁移至 stock/evaluation/daily_hit.py。

新用法（推荐）：

    python -m stock.evaluation.daily_hit --run-dir checkpoints/global_001

旧命令 `python analyze_daily_hit.py --run-dir ...` 继续可用。
"""

from stock.evaluation.daily_hit import main

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""兼容转发：股票池粗筛已迁移至 stock/selection/universe.py。

新用法（推荐）：

    python -m stock.selection.universe --db ... --out ... --top-mv 3000

旧命令 `python select_universe.py ...` 继续可用（walk-forward 的
--pit-universe 子进程调用即走本文件）。
"""

from stock.selection.universe import main

if __name__ == "__main__":
    raise SystemExit(main())

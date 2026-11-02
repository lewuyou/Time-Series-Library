# -*- coding: utf-8 -*-
"""兼容转发：批量导出已迁移至 stock/data/export.py。

新用法（推荐）：

    python -m stock.data.export --db ... --universe-csv ... --data-root ...

旧命令 `python export_global.py ...` 继续可用（walk-forward 每折的
子进程调用即走本文件）。

迁移说明：切分边界解析（resolve_split）、进程池初始化与单股导出
复检（init_worker / export_one）逐字保留在 stock/data/export.py；
export_config.json 的 export_script_sha256 登记实现模块自身的哈希，
断点续跑的口径一致性检查不受本 shim 影响。
"""

from stock.data.export import (
    export_one,
    init_worker,
    main,
    parse_args,
    resolve_split,
)

if __name__ == "__main__":
    main()

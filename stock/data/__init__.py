"""股票数据层。

- schema.py：特征列布局单一事实源（生成侧 + 解析侧共用，纯 Python）。
- panel.py：训练侧面板数据（导出目录装载、全局 schema、窗口 Dataset）。
- export.py：批量导出调度（股票池 → 逐股 dataset.prepare_one → 导出后
  复检；原根目录 export_global.py 已删除，入口为 `python -m stock.data.export`）。

根目录 dataset.py（数据引擎：DuckDB → PIT 特征 → 导出）暂保留在原位，
后续分批迁入本包；其列生成已改为引用 schema.py 的常量与命名函数。
"""

"""股票预测业务层（stock 包）。

在 Time-Series-Library 原生框架旁新增的股票业务包：
- stock.data        数据契约与训练侧面板数据（列布局单一事实源在 schema.py）
- stock.adapters    模型适配器（连接 stock 数据契约与 models/ 模型库）
- stock.training    统一训练器（train / refit）
- stock.inference   单股 checkpoint 预测与批量打分
- stock.selection   选股规则（分位 + 阈值混合信号）
- stock.evaluation  指标、测试报告、walk-forward、命中率分析
- stock.production  每日预测生产管线

统一入口（根目录已不再保留转发脚本）：
- 导出：`python -m stock.data.export`
- 训练：`python -m stock.training.runner`
- 选股池：`python -m stock.selection.universe`
- 每日预测：`python -m stock.production.daily_pipeline`
"""

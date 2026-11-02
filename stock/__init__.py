"""股票预测业务层（stock 包）。

在 Time-Series-Library 原生框架旁新增的股票业务包：
- stock.data        数据契约与训练侧面板数据（列布局单一事实源在 schema.py）
- stock.adapters    模型适配器（连接 stock 数据契约与 models/ 模型库）
- stock.training    统一训练器（train / refit）
- stock.inference   单股 checkpoint 预测与批量打分
- stock.selection   选股规则（分位 + 阈值混合信号）
- stock.evaluation  指标、测试报告、walk-forward、命中率分析
- stock.production  每日预测生产管线

统一入口：train_stock.py（--model 选择适配器，默认 TimeXer）。
旧的根目录入口脚本保留为兼容转发（train_global_timexer.py 等）。
"""

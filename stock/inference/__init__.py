"""推理层：单股 checkpoint 预测与全池批量打分。"""

from stock.inference.predictor import predict, score

__all__ = ["predict", "score"]

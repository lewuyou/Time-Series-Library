#!/usr/bin/env python
"""汇总最后 30 个交易日每只股票"预测第二天正收益"的准确率。

输入：run_universe.py 产出的训练目录（每只股票子目录下
test_predictions.csv，由 train_timexer.py 生成；test 即最后 N 个
交易日目标日）。

指标口径（y > 0 记为"正收益"）：
  acc_direction  方向准确率：(pred>0) == (true>0) 的比例；
  precision_pos  预测正收益的准确率：pred>0 的日子中 true>0 的比例；
  recall_pos     实际正收益日中被预测为正的比例；
  base_rate_pos  实际正收益日占比（"无脑全预测正"的准确率）；
  acc_baseline_* 各基准（0 / 训练均值 / 持续性）的同口径方向准确率。

输出（--out，默认 <train-root>/eval_last30/）：
  per_stock.csv  逐股票明细（按 acc_direction 降序）；
  summary.json   汇总统计与分布。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-root", default="checkpoints/universe_001")
    p.add_argument("--out", default=None, help="默认 <train-root>/eval_last30")
    p.add_argument("--min-days", type=int, default=20,
                   help="test 预测天数下限，不足则剔除并单独计数")
    return p.parse_args()


def sign_metrics(y_true, y_pred):
    """y>0 为正收益的同口径指标。"""
    pos_p, pos_t = y_pred > 0, y_true > 0
    n_pred_pos = int(pos_p.sum())
    n_true_pos = int(pos_t.sum())
    return {
        "n_days": len(y_true),
        "n_pred_pos": n_pred_pos,
        "n_true_pos": n_true_pos,
        "acc_direction": float((pos_p == pos_t).mean()),
        "precision_pos": (
            float(pos_t[pos_p].mean()) if n_pred_pos else None),
        "recall_pos": (
            float(pos_p[pos_t].mean()) if n_true_pos else None),
        "base_rate_pos": float(pos_t.mean()),
    }


def main():
    args = parse_args()
    train_root = Path(args.train_root)
    out_dir = Path(args.out) if args.out else train_root / "eval_last30"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = {}
    results_path = train_root / "run_results.jsonl"
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("name"):
                names[str(rec["symbol"])] = str(rec["name"])

    rows, skipped_short = [], []
    for d in sorted(train_root.iterdir()):
        pred_path = d / "test_predictions.csv"
        if not d.is_dir() or not pred_path.exists():
            continue
        df = pd.read_csv(pred_path)
        if len(df) < args.min_days:
            skipped_short.append({"symbol": d.name, "n_days": len(df)})
            continue
        y_true = df["y_true_simple"].to_numpy(dtype=float)
        row = {"symbol": d.name, "name": names.get(d.name),
               "first_date": str(df["target_date"].iloc[0]),
               "last_date": str(df["target_date"].iloc[-1])}
        row.update(sign_metrics(y_true, df["y_pred_simple"].to_numpy(float)))
        for b in ["baseline_zero", "baseline_train_mean",
                  "baseline_persistence"]:
            m = sign_metrics(y_true, df[b].to_numpy(dtype=float))
            row[f"acc_{b}"] = m["acc_direction"]
            row[f"precision_{b}"] = m["precision_pos"]
        rows.append(row)

    if not rows:
        print(f"[ERROR] {train_root} 下没有可汇总的 test_predictions.csv")
        return 1

    per = pd.DataFrame(rows).sort_values(
        ["acc_direction", "precision_pos"], ascending=False)
    per.insert(0, "rank", range(1, len(per) + 1))
    per.to_csv(out_dir / "per_stock.csv", index=False,
               encoding="utf-8-sig")

    def col(c):
        return per[c].dropna().to_numpy(dtype=float)

    acc = col("acc_direction")
    prec = col("precision_pos")
    buckets = {
        "acc<0.45": int((acc < 0.45).sum()),
        "0.45-0.50": int(((acc >= 0.45) & (acc < 0.50)).sum()),
        "0.50-0.55": int(((acc >= 0.50) & (acc < 0.55)).sum()),
        "0.55-0.60": int(((acc >= 0.55) & (acc < 0.60)).sum()),
        "acc>=0.60": int((acc >= 0.60).sum()),
    }
    summary = {
        "n_stocks": len(per),
        "n_skipped_short": skipped_short,
        "test_window": {
            "first": str(per["first_date"].min()),
            "last": str(per["last_date"].max()),
        },
        "acc_direction": {
            "mean": float(acc.mean()), "median": float(np.median(acc)),
            "std": float(acc.std()), "min": float(acc.min()),
            "max": float(acc.max()), "distribution": buckets,
            "gt_0.5_count": int((acc > 0.5).sum()),
        },
        "precision_pos": {
            "mean": float(prec.mean()), "median": float(np.median(prec)),
            "gt_base_rate_count": int(
                (per["precision_pos"] > per["base_rate_pos"]).sum()),
        },
        "baseline_mean_acc": {
            b.replace("acc_", ""): float(col(b).mean())
            for b in ["acc_baseline_zero", "acc_baseline_train_mean",
                      "acc_baseline_persistence"]
        },
        "beat_baseline": {
            "vs_zero": int((per["acc_direction"]
                            > per["acc_baseline_zero"]).sum()),
            "vs_train_mean": int((per["acc_direction"]
                                  > per["acc_baseline_train_mean"]).sum()),
            "vs_persistence": int((per["acc_direction"]
                                   > per["acc_baseline_persistence"]).sum()),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"[EVAL] 股票数={summary['n_stocks']} "
          f"窗口={summary['test_window']['first']}.."
          f"{summary['test_window']['last']} "
          f"剔除(天数不足)={len(skipped_short)}")
    a = summary["acc_direction"]
    print(f"  方向准确率: mean={a['mean']:.4f} median={a['median']:.4f} "
          f">0.5 占比={a['gt_0.5_count']}/{summary['n_stocks']}")
    print(f"  预测正收益准确率(precision): mean="
          f"{summary['precision_pos']['mean']:.4f} 跑赢全正基准="
          f"{summary['precision_pos']['gt_base_rate_count']}")
    print(f"  基准方向准确率: {summary['baseline_mean_acc']}")
    print(f"  跑赢基准股票数: {summary['beat_baseline']}")
    print(f"  分布: {buckets}")
    print(f"  明细 -> {out_dir / 'per_stock.csv'}")
    print(f"  汇总 -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

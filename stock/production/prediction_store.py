"""预测落库：SQLite 为唯一事实源，插入即冻结。

历史实现只把每日结果写成 `predictions/predict_<target_date>.csv`，
同日重跑会直接覆盖同名文件：既没有 run 概念，也无法回答"这条预测
是哪次、用哪个模型/数据集、当时是否 stale"。本模块用两张表解决：

- `prediction_runs`：一次运行的元信息（as_of/target_date、model_id、
  dataset_id、checkpoint 指纹、发布结论与阻断原因）；
- `predictions`：逐 `(run_id, symbol, target_date)` 预测记录；主键即
  唯一键，`BEFORE UPDATE/DELETE` 触发器保证插入后不可改写。重跑必须
  新建 run_id，历史结论永远可回溯。

CSV 不再是事实源：`export_view` 从库里读出并导出为只读视图。
"""

import sqlite3
from pathlib import Path

import pandas as pd

from stock.inference.contracts import PredictionStatus, STATUS_VALUES
from stock.utils import now_text

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prediction_runs (
    run_id                TEXT PRIMARY KEY,
    created_at            TEXT NOT NULL,
    as_of_date            TEXT NOT NULL,
    target_date           TEXT NOT NULL,
    model_id              TEXT NOT NULL,
    dataset_id            TEXT NOT NULL,
    checkpoint            TEXT NOT NULL,
    checkpoint_sha256     TEXT NOT NULL,
    feature_schema_hash   TEXT,
    quantile_pct          REAL NOT NULL,
    floor_pct             REAL NOT NULL,
    publish_status        TEXT NOT NULL,
    blockers              TEXT,
    notes                 TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id             TEXT NOT NULL,
    symbol             TEXT NOT NULL,
    target_date        TEXT NOT NULL,
    as_of_date         TEXT NOT NULL,
    model_id           TEXT NOT NULL,
    dataset_id         TEXT NOT NULL,
    y_pred             REAL,
    pred_return_pct    REAL,
    status             TEXT NOT NULL,
    rank               INTEGER,
    quantile_pass      INTEGER,
    floor_pass         INTEGER,
    signal             INTEGER,
    input_date         TEXT,
    detail             TEXT,
    inserted_at        TEXT NOT NULL,
    PRIMARY KEY (run_id, symbol, target_date),
    FOREIGN KEY (run_id) REFERENCES prediction_runs(run_id)
);

CREATE TRIGGER IF NOT EXISTS predictions_frozen_update
BEFORE UPDATE ON predictions
BEGIN
    SELECT RAISE(ABORT, 'predictions 已冻结：重跑请新建 run_id，不要原地更新');
END;

CREATE TRIGGER IF NOT EXISTS predictions_frozen_delete
BEFORE DELETE ON predictions
BEGIN
    SELECT RAISE(ABORT, 'predictions 已冻结：不允许删除历史预测记录');
END;
"""

PUBLISH_STATUSES = ("pending", "published", "blocked")


class FrozenPredictionError(RuntimeError):
    """重复写入已冻结的 (run_id, symbol, target_date)。"""


def connect(db_path):
    """打开/初始化预测库（WAL + 外键 + 冻结触发器）。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA_SQL)
    con.commit()
    return con


def make_run_id(con, target_date, checkpoint_sha256):
    """生成可读且库内唯一的 run_id：日期 + 秒级时间 + checkpoint 前缀。"""
    stamp = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%dT%H%M%S")
    base = f"run-{str(target_date).replace('-', '')}-{stamp}-{checkpoint_sha256[:8]}"
    candidate, n = base, 1
    while con.execute(
        "SELECT 1 FROM prediction_runs WHERE run_id = ?", (candidate,)
    ).fetchone():
        n += 1
        candidate = f"{base}-{n}"
    return candidate


def register_run(con, *, run_id, as_of_date, target_date, model_id,
                 dataset_id, checkpoint, checkpoint_sha256,
                 feature_schema_hash=None, quantile_pct, floor_pct,
                 publish_status="pending", blockers=None, notes=None):
    """登记一次运行（元信息可变；逐条预测记录不可变）。"""
    if publish_status not in PUBLISH_STATUSES:
        raise ValueError(f"未知发布状态：{publish_status}")
    con.execute(
        """INSERT INTO prediction_runs (
            run_id, created_at, as_of_date, target_date, model_id,
            dataset_id, checkpoint, checkpoint_sha256, feature_schema_hash,
            quantile_pct, floor_pct, publish_status, blockers, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id, str(now_text()), str(as_of_date), str(target_date),
            str(model_id), str(dataset_id), str(checkpoint),
            str(checkpoint_sha256), feature_schema_hash,
            float(quantile_pct), float(floor_pct), publish_status,
            None if blockers is None else "\n".join(map(str, blockers)),
            notes,
        ),
    )
    con.commit()
    return run_id


def _scalar(value):
    """pandas 标量 → sqlite 可存值（NaN/NA → NULL）。"""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    return value.item() if hasattr(value, "item") else value


def insert_predictions(con, run_id, frame):
    """写入某次运行的全部记录（失败/stale 也落库），插入即冻结。

    重复的 (run_id, symbol, target_date) 会触发主键冲突并显式报错，
    不会静默覆盖。
    """
    if frame is None or frame.empty:
        raise ValueError("预测记录为空，不能落库")
    bad = sorted(set(frame["status"].astype(str)) - STATUS_VALUES)
    if bad:
        raise ValueError(f"未知预测状态：{bad}")
    run = con.execute(
        "SELECT as_of_date, target_date FROM prediction_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if run is None:
        raise ValueError(f"run_id 未登记：{run_id}")
    stamp = str(now_text())
    rows = []
    for record in frame.itertuples(index=False):
        rows.append((
            run_id,
            str(record.symbol),
            str(record.target_date),
            str(record.as_of_date),
            str(record.model_id),
            str(record.dataset_id),
            _scalar(record.y_pred),
            _scalar(record.pred_return_pct),
            str(record.status),
            _scalar(getattr(record, "rank", None)),
            _scalar(getattr(record, "quantile_pass", None)),
            _scalar(getattr(record, "floor_pass", None)),
            _scalar(getattr(record, "signal", None)),
            _scalar(getattr(record, "input_date", None)),
            _scalar(getattr(record, "detail", None)),
            stamp,
        ))
    try:
        with con:
            con.executemany(
                """INSERT INTO predictions (
                    run_id, symbol, target_date, as_of_date, model_id,
                    dataset_id, y_pred, pred_return_pct, status, rank,
                    quantile_pass, floor_pass, signal, input_date, detail,
                    inserted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
    except sqlite3.IntegrityError as exc:
        raise FrozenPredictionError(
            f"run_id={run_id} 存在重复 (symbol, target_date) 或记录已冻结：{exc}"
        ) from exc
    return len(rows)


def update_publish_status(con, run_id, publish_status, blockers=None):
    """更新运行级发布结论（预测明细本身仍不可变）。"""
    if publish_status not in PUBLISH_STATUSES:
        raise ValueError(f"未知发布状态：{publish_status}")
    con.execute(
        "UPDATE prediction_runs SET publish_status = ?, blockers = ? "
        "WHERE run_id = ?",
        (
            publish_status,
            None if blockers is None else "\n".join(map(str, blockers)),
            run_id,
        ),
    )
    con.commit()


def read_run(con, run_id):
    row = con.execute(
        "SELECT * FROM prediction_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"run_id 不存在：{run_id}")
    columns = [d[0] for d in con.execute(
        "SELECT * FROM prediction_runs LIMIT 1"
    ).description]
    return dict(zip(columns, row))


def read_predictions(con, run_id):
    frame = pd.read_sql_query(
        "SELECT * FROM predictions WHERE run_id = ? "
        "ORDER BY target_date, rank, symbol",
        con, params=(run_id,),
    )
    return frame


def list_runs(con, target_date=None, limit=20):
    sql = "SELECT * FROM prediction_runs"
    params = []
    if target_date is not None:
        sql += " WHERE target_date = ?"
        params.append(str(target_date))
    sql += " ORDER BY created_at DESC, run_id DESC LIMIT ?"
    params.append(int(limit))
    cursor = con.execute(sql, params)
    columns = [d[0] for d in cursor.description]
    return pd.DataFrame(cursor.fetchall(), columns=columns)


def run_ids_for_target(con, target_date):
    """同一 target_date 的全部 run_id（按创建顺序）。"""
    return [
        r[0] for r in con.execute(
            "SELECT run_id FROM prediction_runs WHERE target_date = ? "
            "ORDER BY created_at, run_id", (str(target_date),),
        )
    ]


VIEW_NOTE = "# 视图，以库为准（authoritative: SQLite prediction store）"

VIEW_COLUMNS = [
    "run_id", "rank", "symbol", "name", "industry_l2", "as_of_date",
    "target_date", "pred_return_pct", "quantile_pass", "floor_pass",
    "signal", "status",
]


def export_view(con, run_id, path, universe=None):
    """把库内某次运行导出为 CSV 视图（第一行为显式"以库为准"标注）。"""
    frame = read_predictions(con, run_id)
    if universe is not None and len(universe):
        frame = frame.merge(universe, on="symbol", how="left")
    if "name" not in frame:
        frame["name"] = None
        frame["industry_l2"] = None
    columns = [c for c in VIEW_COLUMNS if c in frame]
    text = frame.reindex(columns=columns).to_csv(index=False)
    path = Path(path)
    path.write_text(
        f"{VIEW_NOTE}  run_id={run_id}\n{text}", encoding="utf-8-sig"
    )
    return path


def db_path_for(root):
    return Path(root) / "predictions.db"

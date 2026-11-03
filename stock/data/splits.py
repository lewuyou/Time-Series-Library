"""自动倒推切分边界（resolve_split，自 stock/data/export.py 原样迁出）。

函数体逐字节未动（含 `ds.` 引用），以维持 tests/test_equiv.py 对
stock.data.export.resolve_split 的源码哈希冻结基线：export.py 改为
从本模块转发 import，inspect.getsource 指向此处，哈希不变。
`import dataset as ds` 不会形成加载环：根目录 dataset.py 用 PEP 562
模块级 __getattr__ 延迟转发 resolve_split，不在顶层导入本模块，
因此本模块加载时 dataset（或 __main__ 之外的 dataset 实例）已完整。
"""

import duckdb
import pandas as pd

import dataset as ds

def resolve_split(db, test_days, val_days, as_of=None, train_end=None,
                  val_end=None, train_start=None, split_layout="train_val_test"):
    """返回 (as_of, train_end, val_end, test_start)。

    与 dataset.py 同源的交易日历。两种布局：
      - train_val_test（原协议）：
        * 未提供显式边界：按库内最新行情日自动倒推（与历史单次评估一致）；
        * 显式边界：as_of/train_end/val_end 必须同时提供，均为交易日历内
          日期且满足 train_end < val_end < as_of <= 库内最新行情日；
          train_start 可选，限制最早训练目标日期（rolling），
          必须不晚于 train_end。
      - train_test（walk-forward 重拟合导出）：
        * 必须显式提供 as_of 与 train_end（train_end 即 fit_end，含原
          验证区间；生产式全量重拟合时 train_end 可等于 as_of），
          不得提供 val_end；
        * 满足 train_end <= as_of <= 库内最新行情日；
        * test_start = train_end 的下一交易日（测试覆盖率为评估信息，
          不决定训练资格，见 prepare_one）。
    """
    if split_layout not in ds.SPLIT_LAYOUTS:
        raise ValueError(f"不支持的 split_layout：{split_layout}")

    if split_layout == "train_test":
        if val_end is not None:
            raise ValueError("train_test 布局没有验证集，不能提供 --val-end")
        if as_of is None or train_end is None:
            raise ValueError(
                "train_test 布局必须显式提供 --as-of 与 --train-end"
                "（--train-end 即最终训练截止日 fit_end）"
            )
        as_of = pd.Timestamp(as_of).normalize()
        train_end = pd.Timestamp(train_end).normalize()
    else:
        explicit = [as_of, train_end, val_end]
        if any(x is not None for x in explicit):
            if not all(x is not None for x in explicit):
                raise ValueError(
                    "显式切分必须同时提供 --as-of、--train-end、--val-end"
                )
            as_of = pd.Timestamp(as_of).normalize()
            train_end = pd.Timestamp(train_end).normalize()
            val_end = pd.Timestamp(val_end).normalize()

    con = duckdb.connect(db, read_only=True)
    try:
        calendar, _ = ds.load_calendar(
            con, None, ds.DEFAULT_CALENDAR_START, 50
        )
        value = con.execute(
            "SELECT max(date) FROM akshare_tx WHERE adjust='hfq'"
        ).fetchone()[0]
        if value is None:
            raise ValueError("数据库没有后复权行情")
        latest = pd.Timestamp(value).normalize()

        if as_of is None:
            as_of = latest
            if as_of not in calendar:
                raise ValueError(f"as-of {as_of.date()} 不在交易日历内")
            upto = calendar[calendar <= as_of]
            # 全局日期的粗略下限检查；单股样本是否达标由 prepare_one() 判断。
            need = test_days + val_days + 300
            if len(upto) < need:
                raise ValueError(f"交易日不足：{len(upto)} < {need}")
            val_end = upto[-(test_days + 1)]
            train_end = upto[-(test_days + val_days + 1)]
        elif as_of > latest:
            raise ValueError(
                f"as-of {as_of.date()} 晚于库内最新行情日 {latest.date()}；"
                "walk-forward 历史折的 as-of 不能超过库内数据"
            )

        for name, boundary in (
            ("as-of", as_of),
            ("train-end", train_end),
            ("val-end", val_end),
        ):
            if boundary is None:
                # train_test 布局没有验证集：val_end=None 合法且必须；
                # 其余边界缺失才是配置错误。
                if split_layout == "train_test" and name != "val-end":
                    raise ValueError(f"{name} 未提供")
                continue
            if boundary not in calendar:
                raise ValueError(f"{name}={boundary.date()} 不在交易日历")

        if split_layout == "train_test":
            if not train_end <= as_of:
                raise ValueError("必须满足 train-end <= as-of")
        elif not train_end < val_end < as_of:
            raise ValueError("必须满足 train-end < val-end < as-of")

        if calendar.get_loc(as_of) + 1 >= len(calendar):
            raise ValueError("交易日历缺少 as-of 之后的下一交易日")

        if train_start is not None:
            start = pd.Timestamp(train_start).normalize()
            if start not in calendar:
                raise ValueError(
                    f"train-start={start.date()} 不在交易日历"
                )
            if start > train_end:
                raise ValueError("train-start 不能晚于 train-end")

        boundary_after_train = val_end if split_layout == "train_val_test" \
            else train_end
        test_start = calendar[calendar.get_loc(boundary_after_train) + 1]
        return as_of, train_end, val_end, test_start
    finally:
        con.close()



"""수집 상태 점검: 하루에 몇 번 수집됐고 어디가 비었는지 보여준다.

사용법:  python src/health.py
"""
from __future__ import annotations

import sys

import pandas as pd

from data import STEPS_PER_DAY, load_snapshots, to_grid

GAP = pd.Timedelta(minutes=30)  # 이보다 길게 비면 누락으로 본다


def daily_summary(grid: pd.DataFrame) -> pd.DataFrame:
    got = grid.notna().any(axis=1)
    by_day = got.groupby(grid.index.date)
    out = pd.DataFrame({"수집": by_day.sum(), "눈금": by_day.size()})
    out["수집률"] = (out["수집"] / out["눈금"]).round(3)
    out["대여소"] = grid.notna().sum(axis=1).groupby(grid.index.date).max()
    return out


def find_gaps(times: pd.Series) -> pd.DataFrame:
    t = pd.Series(sorted(times.unique()))
    d = t.diff()
    return pd.DataFrame({"부터": t.shift()[d > GAP], "까지": t[d > GAP], "길이": d[d > GAP]})


def main() -> int:
    snaps = load_snapshots()
    if snaps.empty:
        print("수집된 데이터가 없습니다.")
        return 1
    grid = to_grid(snaps)
    n = snaps["ts"].nunique()
    span = snaps["ts"].max() - snaps["ts"].min()
    print(f"수집 {n}회, {snaps['ts'].min()} ~ {snaps['ts'].max()} ({span})")
    print(f"15분 눈금 {len(grid)}개 중 {int(grid.notna().any(axis=1).sum())}개 채움 "
          f"(하루 {STEPS_PER_DAY}개가 만점)")
    print(daily_summary(grid).to_string())
    gaps = find_gaps(snaps["ts"])
    print(f"\n30분 넘게 빈 구간 {len(gaps)}곳")
    if len(gaps):
        print(gaps.sort_values("길이", ascending=False).head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

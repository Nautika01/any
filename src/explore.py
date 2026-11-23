"""탐색 분석. 쌓인 데이터로 시간대 패턴, 자주 비는 대여소, 예측 거리별 변화 크기를 요약해 출력한다.

사용법:  python3 src/explore.py
데이터가 늘어날수록 같은 표가 더 믿을 만해진다. 요일 패턴은 1주일분이 쌓인 뒤에야 볼 수 있다.
"""
from __future__ import annotations

import pandas as pd

from data import STEPS_PER_HOUR, load_grid, load_stations, load_weather


def hourly_profile(grid: pd.DataFrame, horizon: int = STEPS_PER_HOUR) -> pd.DataFrame:
    """시(0~23)별 요약. 변화는 그 시각에서 horizon 눈금 뒤까지의 절대 변화 평균(= 직전 값 유지 기준선의 오차)."""
    change = (grid.shift(-horizon) - grid).abs().mean(axis=1)
    out = pd.DataFrame({
        "눈금": grid.notna().any(axis=1).groupby(grid.index.hour).sum(),
        "총대수": grid.sum(axis=1, min_count=1).groupby(grid.index.hour).mean(),
        "빈 대여소 비율": (grid == 0).where(grid.notna()).mean(axis=1).groupby(grid.index.hour).mean(),
        "변화": change.groupby(grid.index.hour).mean(),
    })
    out.index.name = "시"
    return out


def empty_share(grid: pd.DataFrame) -> pd.Series:
    """대여소별로 관측된 눈금 중 자전거가 0대였던 비율."""
    return ((grid == 0).sum() / grid.notna().sum()).sort_values(ascending=False)


def change_by_horizon(grid: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """예측 거리별 절대 변화. 직전 값을 그대로 내놓을 때 얼마나 틀리는지와 같다."""
    rows = {}
    for h in horizons:
        d = (grid.shift(-h) - grid).abs().stack()
        rows[h * 15] = {"평균": d.mean(), "변화 없음": (d == 0).mean(), "p90": d.quantile(0.9), "p99": d.quantile(0.99)}
    out = pd.DataFrame(rows).T
    out.index.name = "분 뒤"
    return out


def net_flow(grid: pd.DataFrame, start: str, end: str) -> pd.Series:
    """start~end 사이 대여소별 순변화(끝 - 시작). 음수는 자전거가 빠져나간 곳."""
    window = grid.loc[start:end].dropna(how="all")
    if len(window) < 2:
        return pd.Series(dtype=float)
    return (window.iloc[-1] - window.iloc[0]).dropna().sort_values()


def main() -> int:
    grid, stations = load_grid(), load_stations()
    if grid.empty:
        print("수집된 데이터가 없습니다.")
        return 1
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    span = grid.index.max() - grid.index.min()
    print(f"기간 {grid.index.min()} ~ {grid.index.max()} ({span}), 대여소 {grid.shape[1]}곳, "
          f"채워진 눈금 {int(grid.notna().any(axis=1).sum())}/{len(grid)}")
    values = grid.stack()
    racks = stations["racks"].reindex(grid.columns)
    print(f"\n자전거 수: 평균 {values.mean():.1f}, 중앙값 {values.median():.0f}, 최대 {values.max():.0f}, "
          f"0대 {(values == 0).mean():.1%}, 거치대 수 초과 {(grid.gt(racks, axis=1)).stack().mean():.1%}")

    print("\n[예측 거리별 변화 = 직전 값 유지의 오차]")
    print(change_by_horizon(grid, [1, 4, 8]).round(3).to_string())

    print("\n[시간대별] 변화는 1시간 뒤 기준")
    print(hourly_profile(grid).round(3).to_string())

    share = empty_share(grid)
    print(f"\n[빈 대여소] 한 번도 안 빈 곳 {(share == 0).mean():.1%}, 절반 넘게 빈 곳 {(share > 0.5).mean():.1%}")
    top = share.head(15)
    print(pd.DataFrame({"빈 비율": top.round(2), "거치대": racks.reindex(top.index),
                        "이름": stations["name"].reindex(top.index)}).to_string())

    for day in sorted(set(grid.index.date)):
        flow = net_flow(grid, f"{day} 06:00", f"{day} 09:00")
        if len(flow):
            names = stations["name"]
            print(f"\n[{day} 06~09시 순변화] 합계 {flow.sum():+.0f}대, 줄어든 곳 {(flow < 0).sum()}, 늘어난 곳 {(flow > 0).sum()}")
            print("  많이 빠진 곳: " + ", ".join(f"{names[s]}({v:+.0f})" for s, v in flow.head(5).items()))
            print("  많이 들어온 곳: " + ", ".join(f"{names[s]}({v:+.0f})" for s, v in flow.tail(5)[::-1].items()))

    weather = load_weather().reindex(grid.index)
    hourly_change = (grid.shift(-STEPS_PER_HOUR) - grid).abs().mean(axis=1)
    print("\n[날씨] " + ", ".join(f"{c} {weather[c].min():.1f}~{weather[c].max():.1f}" for c in ["temp_c", "precip_mm"]))
    wet = weather["precip_mm"] > 0
    if wet.any() and (~wet).any():
        print(f"  비 온 눈금 {int(wet.sum())}개의 1시간 변화 평균 {hourly_change[wet].mean():.2f}, "
              f"안 온 눈금 {int((~wet).sum())}개 {hourly_change[~wet].mean():.2f} (시간대가 섞여 있어 비 효과로 읽으면 안 된다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

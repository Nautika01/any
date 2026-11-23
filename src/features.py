"""모델 학습용 표 만들기. 특성은 예측 시점까지 알 수 있는 값만 쓴다."""
from __future__ import annotations

import pandas as pd

from data import STEPS_PER_DAY, STEPS_PER_WEEK

# 예측 시점 기준 과거 시차(15분 단위). 0 = 예측 시점의 값.
LAGS = (0, 1, 2, 4, 8)


def make_dataset(grid: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """행 하나 = (목표 시각, 대여소). 목표 시각의 자전거 수를 horizon 전에 예측한다.

    특성
      lag_k        예측 시점에서 k 눈금 전의 값 (lag_0 = 예측 시점의 값)
      diff_1h      예측 시점 직전 1시간 동안의 변화량
      day_ago      목표 시각 하루 전 값 (horizon < 1일 일 때만 알 수 있다)
      week_ago     목표 시각 1주 전 값
      hour, dow    목표 시각의 시(분 포함 소수)와 요일 — 달력은 미리 알 수 있다
    """
    if horizon < 1:
        raise ValueError("horizon 은 1 이상이어야 한다")
    cols = {"target": grid}
    for k in LAGS:
        cols[f"lag_{k}"] = grid.shift(horizon + k)
    cols["diff_1h"] = grid.shift(horizon) - grid.shift(horizon + 4)
    if horizon < STEPS_PER_DAY:
        cols["day_ago"] = grid.shift(STEPS_PER_DAY)
    if horizon < STEPS_PER_WEEK:
        cols["week_ago"] = grid.shift(STEPS_PER_WEEK)
    long = pd.concat(
        {name: frame.stack(future_stack=True) for name, frame in cols.items()}, axis=1
    )
    long.index.names = ["ts", "station_id"]
    long = long.dropna(subset=["target", "lag_0"]).reset_index()
    long["hour"] = long["ts"].dt.hour + long["ts"].dt.minute / 60
    long["dow"] = long["ts"].dt.dayofweek
    return long

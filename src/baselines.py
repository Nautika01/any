"""기준선 예측. 모든 함수는 '목표 시각' 기준으로 정렬된 예측 표를 돌려준다.

즉 결과의 행 T 는 "T 시점의 자전거 수를 T - horizon 에 예측한 값"이다.
"""
from __future__ import annotations

import pandas as pd

from data import STEPS_PER_WEEK


def persistence(grid: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """직전 값 유지: 예측 시점의 값이 그대로 이어진다고 본다."""
    return grid.shift(horizon)


def last_week(grid: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """지난주 같은 요일·시각 값. 예측 거리가 1주보다 짧아야 쓸 수 있다."""
    if horizon > STEPS_PER_WEEK:
        raise ValueError("예측 거리가 1주를 넘으면 지난주 값을 아직 모른다")
    return grid.shift(STEPS_PER_WEEK)


BASELINES = {"직전 값 유지": persistence, "지난주 같은 시각": last_week}

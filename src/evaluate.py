"""시간 순서 분할과 평가. 모든 모델을 같은 검증 구간·같은 대여소에서 비교한다.

사용법:  python src/evaluate.py [검증 일수]
"""
from __future__ import annotations

import sys

import pandas as pd

from baselines import BASELINES
from data import STEPS_PER_DAY, STEPS_PER_HOUR, load_grid

HORIZONS = {"1시간": STEPS_PER_HOUR, "2시간": 2 * STEPS_PER_HOUR}


def split_point(grid: pd.DataFrame, val_days: int) -> pd.Timestamp:
    """검증 구간 시작 시각. 마지막 val_days 일이 검증, 그 앞이 학습."""
    start = grid.index.max() - pd.Timedelta(days=val_days) + pd.Timedelta(minutes=15)
    if start <= grid.index.min():
        raise ValueError(f"데이터가 {val_days}일보다 짧아 검증 구간을 나눌 수 없다")
    return start


def score_mask(grid: pd.DataFrame, preds: list[pd.DataFrame], val_start: pd.Timestamp) -> pd.DataFrame:
    """채점할 칸: 검증 구간이고, 실제 값과 모든 모델의 예측이 다 있는 칸만.

    모델마다 예측 못 하는 칸이 다르면(예: 지난주 값이 없는 첫 주) 쉬운 칸만
    채점한 모델이 유리해지므로, 공통으로 예측한 칸에서만 비교한다.
    """
    mask = grid.notna() & (grid.index >= val_start)[:, None]
    for p in preds:
        mask &= p.reindex_like(grid).notna()
    return mask


def mae(grid: pd.DataFrame, pred: pd.DataFrame, mask: pd.DataFrame) -> float:
    err = (grid - pred.reindex_like(grid)).abs().where(mask)
    return float(err.stack().mean())


def evaluate(grid: pd.DataFrame, models: dict, val_start: pd.Timestamp) -> pd.DataFrame:
    """models: 이름 -> (grid, horizon) 을 받아 목표 시각 기준 예측 표를 주는 함수."""
    rows = {}
    counts = {}
    for label, h in HORIZONS.items():
        preds = {name: fn(grid, h) for name, fn in models.items()}
        mask = score_mask(grid, list(preds.values()), val_start)
        counts[label] = int(mask.values.sum())
        for name, p in preds.items():
            rows.setdefault(name, {})[f"MAE {label}"] = mae(grid, p, mask)
    out = pd.DataFrame(rows).T
    out.attrs["counts"] = counts
    return out


def main() -> int:
    val_days = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    grid = load_grid()
    days = len(grid) / STEPS_PER_DAY
    print(f"데이터 {grid.index.min()} ~ {grid.index.max()} ({days:.1f}일, 대여소 {grid.shape[1]}곳)")
    try:
        val_start = split_point(grid, val_days)
    except ValueError as e:
        print(e)
        return 1
    result = evaluate(grid, BASELINES, val_start)
    print(f"검증 구간 {val_start} ~ {grid.index.max()}")
    print(result.round(3).to_string())
    print("채점한 칸 수:", result.attrs["counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

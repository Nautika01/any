import numpy as np
import pandas as pd
import pytest

import baselines
import data
import evaluate
import features
import health


def make_grid(days=10, stations=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-11-02", periods=days * 96, freq="15min", name="ts")
    vals = rng.integers(0, 20, size=(len(idx), stations)).astype(float)
    return pd.DataFrame(vals, index=idx, columns=[f"ST-{i}" for i in range(stations)])


def test_to_grid_snaps_late_runs_and_keeps_gaps():
    snaps = pd.DataFrame({
        "ts": pd.to_datetime(["2026-11-23T09:55", "2026-11-23T10:19", "2026-11-23T10:21", "2026-11-23T11:02"]),
        "station_id": ["A"] * 4,
        "bikes": [1, 2, 3, 4],
    })
    grid = data.to_grid(snaps)
    assert list(grid.index.strftime("%H:%M")) == ["10:00", "10:15", "10:30", "10:45", "11:00"]
    # 10:19 와 10:21 은 같은 눈금 -> 마지막 값, 빠진 눈금은 NaN 그대로
    assert grid["A"].tolist()[:2] == [1, 3]
    assert grid["A"].isna().tolist() == [False, False, True, True, False]


def test_baselines_are_aligned_to_target_time():
    grid = make_grid()
    t = grid.index[800]
    assert baselines.persistence(grid, 4).loc[t].equals(grid.loc[t - pd.Timedelta(hours=1)])
    assert baselines.last_week(grid, 4).loc[t].equals(grid.loc[t - pd.Timedelta(days=7)])


def changed_rows(fn, grid, horizon, cut):
    """cut 이후의 실제 값을 바꿨을 때 예측이 달라지는 목표 시각들."""
    tampered = grid.copy()
    tampered.loc[tampered.index > cut] += 1000
    a, b = fn(grid, horizon), fn(tampered, horizon)
    diff = (a.fillna(-1) != b.fillna(-1)).any(axis=1)
    return diff[diff].index


@pytest.mark.parametrize("fn", list(baselines.BASELINES.values()))
@pytest.mark.parametrize("horizon", [4, 8])
def test_baselines_do_not_see_the_future(fn, horizon):
    grid = make_grid()
    cut = grid.index[700]  # 이 시각까지만 안다고 가정
    limit = cut + horizon * pd.Timedelta(minutes=15)
    # cut 까지의 정보로 예측하는 목표 시각(<= cut + horizon)은 영향을 받으면 안 된다
    assert (changed_rows(fn, grid, horizon, cut) > limit).all()


@pytest.mark.parametrize("horizon", [4, 8])
def test_features_do_not_see_the_future(horizon):
    grid = make_grid()
    cut = grid.index[700]
    limit = cut + horizon * pd.Timedelta(minutes=15)
    tampered = grid.copy()
    tampered.loc[tampered.index > cut] += 1000
    a = features.make_dataset(grid, horizon).set_index(["ts", "station_id"]).drop(columns="target")
    b = features.make_dataset(tampered, horizon).set_index(["ts", "station_id"]).drop(columns="target")
    early = a.index.get_level_values("ts") <= limit
    pd.testing.assert_frame_equal(a[early], b.loc[a[early].index])


def test_features_values():
    grid = make_grid()
    ds = features.make_dataset(grid, 4)
    row = ds[(ds["ts"] == grid.index[900]) & (ds["station_id"] == "ST-1")].iloc[0]
    assert row["target"] == grid["ST-1"].iloc[900]
    assert row["lag_0"] == grid["ST-1"].iloc[896]
    assert row["lag_4"] == grid["ST-1"].iloc[892]
    assert row["diff_1h"] == grid["ST-1"].iloc[896] - grid["ST-1"].iloc[892]
    assert row["day_ago"] == grid["ST-1"].iloc[900 - 96]
    assert row["week_ago"] == grid["ST-1"].iloc[900 - 672]


def test_split_is_time_ordered_and_rejects_short_data():
    grid = make_grid(days=10)
    start = evaluate.split_point(grid, 2)
    assert (grid.index >= start).sum() == 2 * 96
    with pytest.raises(ValueError):
        evaluate.split_point(make_grid(days=1), 2)


def test_evaluate_scores_all_models_on_the_same_cells():
    grid = make_grid(days=8)  # 지난주 값은 마지막 하루에만 있다
    val_start = evaluate.split_point(grid, 2)
    result = evaluate.evaluate(grid, baselines.BASELINES, val_start)
    # 검증 2일 중 지난주 값이 있는 건 하루뿐 -> 두 모델 모두 그 하루로만 채점
    assert result.attrs["counts"]["1시간"] == 96 * grid.shape[1]
    assert result.notna().all().all()


def test_evaluate_known_answer():
    grid = make_grid(days=3)
    val_start = evaluate.split_point(grid, 1)
    perfect = lambda g, h: g.copy()
    off_by_two = lambda g, h: g + 2
    result = evaluate.evaluate(grid, {"완벽": perfect, "2대 차이": off_by_two}, val_start)
    assert result.loc["완벽"].tolist() == [0, 0]
    assert result.loc["2대 차이"].tolist() == [2, 2]


def test_health_finds_gaps():
    times = pd.Series(pd.to_datetime(["2026-11-23T09:55", "2026-11-23T10:10", "2026-11-23T11:30"]))
    gaps = health.find_gaps(times)
    assert len(gaps) == 1 and gaps["길이"].iloc[0] == pd.Timedelta(minutes=80)

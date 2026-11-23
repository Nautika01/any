import pandas as pd

from explore import change_by_horizon, empty_share, hourly_profile, net_flow


def small_grid() -> pd.DataFrame:
    ts = pd.date_range("2026-11-23 06:00", periods=8, freq="15min")
    return pd.DataFrame({"A": [4, 3, 2, 1, 0, 0, 0, 0], "B": [1, 1, 1, 1, 5, 5, None, 5]}, index=ts)


def test_empty_share_counts_only_observed_ticks():
    share = empty_share(small_grid())
    assert share["A"] == 0.5
    assert share["B"] == 0.0  # 결측 눈금은 빈 것으로 세지 않는다


def test_change_by_horizon_is_persistence_error():
    out = change_by_horizon(small_grid()[["A"]], [1, 4])
    assert out.loc[15, "평균"] == 4 / 7  # 7쌍 중 4번 1대씩 변함
    assert out.loc[60, "평균"] == (4 + 3 + 2 + 1) / 4


def test_hourly_profile_groups_by_hour_and_looks_forward():
    out = hourly_profile(small_grid()[["A"]], horizon=4)
    assert list(out.index) == [6, 7]
    assert out.loc[6, "변화"] == (4 + 3 + 2 + 1) / 4
    assert pd.isna(out.loc[7, "변화"])  # 1시간 뒤 값이 없으면 지어내지 않는다
    assert out.loc[7, "빈 대여소 비율"] == 1.0


def test_net_flow_is_end_minus_start():
    flow = net_flow(small_grid(), "2026-11-23 06:00", "2026-11-23 07:45")
    assert flow["A"] == -4 and flow["B"] == 4
    assert list(flow.index) == ["A", "B"]  # 많이 빠진 곳부터

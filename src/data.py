"""수집된 원본을 읽어 15분 격자(시각 × 대여소) 표로 만든다."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FREQ = "15min"
STEPS_PER_HOUR = 4
STEPS_PER_DAY = 96
STEPS_PER_WEEK = 7 * STEPS_PER_DAY


def load_snapshots(data_dir: Path = DATA) -> pd.DataFrame:
    """data/raw 의 일별 파일을 모두 읽어 (ts, station_id, bikes) 긴 표로 돌려준다."""
    files = sorted((data_dir / "raw").glob("*.csv.gz"))
    if not files:
        return pd.DataFrame({"ts": pd.to_datetime([]), "station_id": [], "bikes": []})
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def to_grid(snapshots: pd.DataFrame) -> pd.DataFrame:
    """긴 표를 15분 격자 넓은 표로 바꾼다. 행은 시각, 열은 대여소, 값은 자전거 수.

    GitHub 예약 실행은 늦어지기 일쑤라 수집 시각이 들쭉날쭉하다. 가장 가까운
    15분 눈금에 붙이고, 한 눈금에 여러 번 수집됐으면 마지막 값을 쓴다.
    수집이 빠진 눈금은 채우지 않고 NaN으로 둔다(없는 값을 지어내지 않는다).
    """
    if snapshots.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="ts"))
    df = snapshots.sort_values("ts").assign(ts=lambda d: d["ts"].dt.round(FREQ))
    grid = df.pivot_table(index="ts", columns="station_id", values="bikes", aggfunc="last")
    full = pd.date_range(grid.index.min(), grid.index.max(), freq=FREQ, name="ts")
    return grid.reindex(full)


def load_grid(data_dir: Path = DATA) -> pd.DataFrame:
    return to_grid(load_snapshots(data_dir))


def load_stations(data_dir: Path = DATA) -> pd.DataFrame:
    return pd.read_csv(data_dir / "stations.csv").set_index("station_id")


def load_weather(data_dir: Path = DATA) -> pd.DataFrame:
    """수집 시각의 날씨를 15분 격자에 맞춘 표."""
    files = sorted((data_dir / "weather").glob("*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"]).dt.round(FREQ)
    return df.drop_duplicates("ts", keep="last").set_index("ts").sort_index()

import csv
import gzip
import io
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import collect


def row(i, bikes):
    return {
        "rackTotCnt": "15", "stationName": f"{i}. 대여소", "parkingBikeTotCnt": str(bikes),
        "shared": "50", "stationLatitude": "37.5", "stationLongitude": "126.9",
        "stationId": f"ST-{i}",
    }


def page(rows):
    return {"rentBikeStatus": {"list_total_count": len(rows),
                               "RESULT": {"CODE": "INFO-000", "MESSAGE": "정상 처리되었습니다."},
                               "row": rows}}


def test_parse_page():
    assert collect.parse_page(page([row(1, 3)]))[0]["stationId"] == "ST-1"
    assert collect.parse_page({"RESULT": {"CODE": "INFO-200", "MESSAGE": "없음"}}) == []
    with pytest.raises(RuntimeError):
        collect.parse_page({"RESULT": {"CODE": "INFO-100", "MESSAGE": "인증키 오류"}})


def test_fetch_pages_until_short(monkeypatch):
    pages = [page([row(i, 1) for i in range(1000)]), page([row(i, 1) for i in range(1000, 1700)])]
    monkeypatch.setattr(collect, "get_json", lambda url: pages.pop(0))
    assert len(collect.fetch_stations("k")) == 1700


def test_one_file_per_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(collect, "DATA", tmp_path)
    a = collect.write_snapshot([row(1, 3), row(2, 0)], datetime(2026, 11, 23, 9, 0))
    b = collect.write_snapshot([row(1, 5)], datetime(2026, 11, 23, 9, 15))
    assert a.relative_to(tmp_path).as_posix() == "raw/2026-11-23/0900.csv.gz"
    assert b.relative_to(tmp_path).as_posix() == "raw/2026-11-23/0915.csv.gz"
    got = list(csv.reader(io.StringIO(gzip.decompress(a.read_bytes()).decode())))
    assert got == [["ts", "station_id", "bikes"],
                   ["2026-11-23T09:00", "ST-1", "3"], ["2026-11-23T09:00", "ST-2", "0"]]


def test_snapshots_load_back_through_data_module(monkeypatch, tmp_path):
    import data
    monkeypatch.setattr(collect, "DATA", tmp_path)
    collect.write_snapshot([row(1, 3), row(2, 0)], datetime(2026, 11, 23, 9, 0))
    collect.write_snapshot([row(1, 5), row(2, 1)], datetime(2026, 11, 24, 0, 15))
    grid = data.load_grid(tmp_path)
    assert grid.shape == (62, 2)  # 09:00 ~ 다음 날 00:15, 15분 간격
    assert grid["ST-1"].iloc[[0, -1]].tolist() == [3, 5]


def test_weather_row_takes_next_two_hours_forecast():
    payload = {
        "current": {"temperature_2m": 6.0, "relative_humidity_2m": 78, "precipitation": 0.1, "wind_speed_10m": 7.7},
        "hourly": {"time": ["10:00", "11:00", "12:00"], "temperature_2m": [5.7, 7.0, 9.4],
                   "precipitation": [0.1, 0.5, 0.2], "precipitation_probability": [10, 60, 30]},
    }
    got = collect.weather_row(payload, datetime(2026, 11, 23, 10, 15))
    assert dict(zip(collect.WEATHER_FIELDS, got)) == {
        "ts": "2026-11-23T10:15", "temp_c": 6.0, "humidity": 78, "precip_mm": 0.1, "wind_kmh": 7.7,
        "fc_temp_1h": 7.0, "fc_temp_2h": 9.4, "fc_precip_1h": 0.5, "fc_precip_2h": 0.2,
        "fc_pop_1h": 60, "fc_pop_2h": 30,
    }


def test_stations_merge(monkeypatch, tmp_path):
    monkeypatch.setattr(collect, "DATA", tmp_path)
    collect.write_stations([row(2, 0)])
    collect.write_stations([row(1, 0)])
    lines = (tmp_path / "stations.csv").read_text(encoding="utf-8").splitlines()
    assert [l.split(",")[0] for l in lines] == ["station_id", "ST-1", "ST-2"]

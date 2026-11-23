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


def test_snapshots_append(monkeypatch, tmp_path):
    monkeypatch.setattr(collect, "DATA", tmp_path)
    collect.write_snapshot([row(1, 3), row(2, 0)], datetime(2026, 11, 23, 9, 0))
    p = collect.write_snapshot([row(1, 5)], datetime(2026, 11, 23, 9, 15))
    got = list(csv.reader(io.StringIO(gzip.decompress(p.read_bytes()).decode())))
    assert got == [["ts", "station_id", "bikes"],
                   ["2026-11-23T09:00", "ST-1", "3"], ["2026-11-23T09:00", "ST-2", "0"],
                   ["2026-11-23T09:15", "ST-1", "5"]]


def test_stations_merge(monkeypatch, tmp_path):
    monkeypatch.setattr(collect, "DATA", tmp_path)
    collect.write_stations([row(2, 0)])
    collect.write_stations([row(1, 0)])
    lines = (tmp_path / "stations.csv").read_text(encoding="utf-8").splitlines()
    assert [l.split(",")[0] for l in lines] == ["station_id", "ST-1", "ST-2"]

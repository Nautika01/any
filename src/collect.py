"""서울 따릉이 대여소 실시간 현황 + 날씨·예보를 한 번 수집해 data/ 아래에 쓴다.

사용법:  SEOUL_API_KEY=... python src/collect.py
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BIKE_URL = "http://openapi.seoul.go.kr:8088/{key}/json/bikeList/{start}/{end}/"
WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast?latitude=37.5665&longitude=126.9780"
    "&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
    "&hourly=temperature_2m,precipitation,precipitation_probability&forecast_hours=3"
    "&timezone=Asia%2FSeoul"
)
PAGE = 1000  # API가 한 번에 허용하는 최대 행 수
MAX_PAGES = 10


def get_json(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 3)
    raise AssertionError("unreachable")


def parse_page(payload: dict) -> list[dict]:
    """bikeList 응답 한 쪽에서 대여소 행을 꺼낸다. 범위를 벗어난 쪽은 빈 목록."""
    if "rentBikeStatus" not in payload:
        code = payload.get("RESULT", {}).get("CODE", "")
        if code == "INFO-200":  # 해당하는 데이터 없음 = 마지막 쪽을 지남
            return []
        raise RuntimeError(f"따릉이 API 오류: {payload}")
    body = payload["rentBikeStatus"]
    code = body.get("RESULT", {}).get("CODE", "")
    if code != "INFO-000":
        raise RuntimeError(f"따릉이 API 오류: {body.get('RESULT')}")
    return body.get("row", [])


def fetch_stations(key: str) -> list[dict]:
    rows: list[dict] = []
    for page in range(MAX_PAGES):
        start = page * PAGE + 1
        got = parse_page(get_json(BIKE_URL.format(key=key, start=start, end=start + PAGE - 1)))
        rows.extend(got)
        if len(got) < PAGE:
            break
    return rows


def write_snapshot(rows: list[dict], now: datetime) -> Path:
    """시각·대여소·자전거 수를 수집 1회당 파일 하나로 쓴다.

    하루치를 한 파일에 덧붙이면 커밋마다 파일 전체가 git에 새로 쌓여
    저장소가 빠르게 불어난다. 한 번 쓴 파일은 다시 건드리지 않는다.
    """
    path = DATA / "raw" / f"{now:%Y-%m-%d}" / f"{now:%H%M}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["ts", "station_id", "bikes"])
    ts = now.strftime("%Y-%m-%dT%H:%M")
    for r in rows:
        w.writerow([ts, r["stationId"], int(r["parkingBikeTotCnt"])])
    path.write_bytes(gzip.compress(buf.getvalue().encode(), mtime=0))
    return path


def write_stations(rows: list[dict]) -> None:
    """대여소 이름·좌표·거치대 수. 기존 목록에 새 대여소를 합쳐 다시 쓴다."""
    path = DATA / "stations.csv"
    fields = ["station_id", "name", "lat", "lon", "racks"]
    known: dict[str, dict] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            known = {r["station_id"]: r for r in csv.DictReader(f)}
    for r in rows:
        known[r["stationId"]] = {
            "station_id": r["stationId"],
            "name": r["stationName"],
            "lat": r["stationLatitude"],
            "lon": r["stationLongitude"],
            "racks": r["rackTotCnt"],
        }
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(known[k] for k in sorted(known))


WEATHER_FIELDS = [
    "ts", "temp_c", "humidity", "precip_mm", "wind_kmh",
    "fc_temp_1h", "fc_temp_2h", "fc_precip_1h", "fc_precip_2h", "fc_pop_1h", "fc_pop_2h",
]


def weather_row(payload: dict, now: datetime) -> list:
    """현재 날씨와, 지금 시점에 나와 있는 1·2시간 뒤 예보.

    예보는 나중에 되돌려 받을 수 없으므로 수집 시점에 함께 남긴다. 예측 모델은
    목표 시각의 실제 날씨를 미리 알 수 없으니 이 예보 값을 특성으로 써야 한다.
    hourly[0] 이 현재 시각이 속한 정시, [1]·[2] 가 다음 두 정시다.
    """
    cur, h = payload["current"], payload["hourly"]
    return [
        now.strftime("%Y-%m-%dT%H:%M"),
        cur["temperature_2m"], cur["relative_humidity_2m"], cur["precipitation"], cur["wind_speed_10m"],
        h["temperature_2m"][1], h["temperature_2m"][2],
        h["precipitation"][1], h["precipitation"][2],
        h["precipitation_probability"][1], h["precipitation_probability"][2],
    ]


def write_weather(now: datetime) -> None:
    row = weather_row(get_json(WEATHER_URL), now)
    path = DATA / "weather" / f"{now:%Y-%m}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        if new:
            w.writerow(WEATHER_FIELDS)
        w.writerow(row)


def main() -> int:
    key = os.environ.get("SEOUL_API_KEY")
    if not key:
        print("SEOUL_API_KEY 환경변수가 없습니다.", file=sys.stderr)
        return 2
    now = datetime.now(KST)
    rows = fetch_stations(key)
    if len(rows) < 100:  # 정상이면 2,700곳 안팎. 샘플 키는 5행만 준다.
        print(f"대여소가 {len(rows)}곳뿐입니다. 키를 확인하세요.", file=sys.stderr)
        return 1
    path = write_snapshot(rows, now)
    write_stations(rows)
    try:
        write_weather(now)
    except Exception as e:  # 날씨가 실패해도 자전거 기록은 남긴다
        print(f"날씨 수집 실패: {e}", file=sys.stderr)
    print(f"{now:%Y-%m-%d %H:%M} 대여소 {len(rows)}곳 -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

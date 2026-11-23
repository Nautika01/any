# 따릉이 잔여 대수 예측

서울 공공자전거(따릉이) 대여소별 잔여 대수를 15분마다 수집해, 1~2시간 뒤 잔여 대수를 예측하는 프로젝트.

## 진행 상황

- [x] 수집기: 대여소 실시간 현황 + 서울 날씨 (`src/collect.py`)
- [x] 자동 수집: GitHub Actions 15분 주기 (`.github/workflows/collect.yml`)
- [ ] 탐색 분석: 시간대·요일·날씨별 패턴
- [ ] 예측 모델: 기준선(직전 값, 지난주 같은 시각) 대비 성능 비교
- [ ] 예측 정확도 모니터링 화면

## 데이터

| 파일 | 내용 |
|---|---|
| `data/raw/YYYY-MM-DD.csv.gz` | `ts, station_id, bikes` — 수집 시각(KST), 대여소 ID, 거치된 자전거 수 |
| `data/stations.csv` | 대여소 이름, 위도·경도, 거치대 수 |
| `data/weather/YYYY-MM.csv` | 수집 시각의 기온, 습도, 강수량, 풍속 (서울시청 기준) |

출처: [서울 열린데이터광장 — 서울시 공공자전거 실시간 대여정보](https://data.seoul.go.kr/dataList/OA-15493/A/1/datasetView.do), [Open-Meteo](https://open-meteo.com/).

## 실행

```bash
SEOUL_API_KEY=발급받은키 python src/collect.py
```

```bash
python -m pytest tests
```

자동 수집을 켜려면 저장소 Settings → Secrets and variables → Actions 에 `SEOUL_API_KEY` 를 등록한다.

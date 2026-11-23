# Kaggle Playground Series S6E11 — 카드 사기 탐지

거래 정보로 사기 여부(`is_fraud`)의 확률을 예측한다. 평가 지표는 ROC AUC, 마감 2026-11-30.
[대회 페이지](https://www.kaggle.com/competitions/playground-series-s6e11)

## 데이터

- 학습 699,960건, 테스트 299,984건. 결측치와 중복 행 없음.
- 사기 비율 1.67% (11,689건) — 심한 불균형.
- 수치형 13개, 범주형 5개, 참/거짓 6개.
- 사기율이 크게 갈리는 열: `ip_country_mismatch`(16.5% vs 1.4%), `billing_shipping_mismatch`(13.0%),
  `is_foreign_transaction`(12.6%), `auth_method`(인증 없음 6.4% vs 생체 0.5%),
  `merchant_category`(기프트카드 7.3%, 암호화폐 6.5%, 식료품 1.1%), `cvv_retry_count`(상관 0.21).
- `channel`, `device_type`, `card_type`, 시각·요일은 사기율 차이가 거의 없다.

## 실행

`data/` 에 대회 파일 3개(train.csv, test.csv, sample_submission.csv)를 둔다. 저장소에는 올리지 않는다.

```bash
python train.py e11_leaves8_reg --model lgbm --set num_leaves=8 learning_rate=0.03 min_child_samples=200 reg_lambda=10 colsample_bytree=0.5
python blend.py e14_blend e09_leaves4 e11_leaves8_reg e13_catboost
```

## 규칙

- 검증은 층화 5겹 교차검증(seed 42)으로 고정한다. 모든 실험을 같은 분할로 비교한다.
- 실험마다 `experiments.md`에 CV 점수와 (제출했다면) 리더보드 점수를 적는다. 나빠진 실험도 남긴다.
- 겹별 점수의 표준편차가 약 0.003이다. 이보다 작은 CV 차이는 개선으로 치지 않는다.

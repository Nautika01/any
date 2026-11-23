"""Playground S6E11 — 값·자릿수 목표 인코딩 + GLM + GLM에서 출발하는 얕은 나무.

합성 데이터라서 같은 값이 여러 번 반복되고 값마다 사기율이 다르며, 소수 자릿수에도 신호가 있다.
나무는 구간으로만 나누므로 이 신호를 못 본다. 그래서 값과 자릿수 키를 "그 키를 가진 행의 사기율"로 바꾼다.
방법의 출처는 공개 노트북(goodpjw2008, Alex Chilton, megayak, davutkursun)이고 코드는 새로 썼다.

누수 방지: 사기율은 정답을 쓰므로 반드시 겹 안에서 센다.
  - 검증 겹과 테스트 행: 나머지 학습 겹 전체에서 센다.
  - 학습 행(겹 g): 검증 겹과 자기 겹 g를 뺀 나머지에서 센다. 어떤 행도 자기 정답을 보지 않는다.

사용법:  python train_te.py <실험이름> [--stage glm|trees] [--no-digits] [--shuffle-check]
결과:    out/<실험이름>_oof.npy, out/<실험이름>_submission.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train import CATS, FLAGS, FOLDS, HERE, SEED, TARGET, load

DECIMALS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km",
            "account_balance_usd", "velocity_score", "merchant_risk_score"]  # 소수 둘째 자리까지
INTS = ["txn_count_last_24h", "card_age_months", "customer_age", "cvv_retry_count",
        "time_of_day_hour", "day_of_week", "prior_disputes"]
M = 20  # 평활: 기준 사기율을 가진 가상 행 20개


def make_keys(df: pd.DataFrame, digits: bool = True) -> dict[str, np.ndarray]:
    """열마다 묶음 키를 만든다. 반환값은 키 이름 → 0부터 시작하는 정수 코드(학습+테스트 공통)."""
    raw: dict[str, np.ndarray] = {}
    for c in DECIMALS:
        v = np.round(df[c].to_numpy() * 100).astype(np.int64)  # 1371.37 → 137137
        whole = v // 100
        raw[f"{c}|value"] = v
        if digits:
            lead = np.char.zfill(whole.astype(str), 1)
            ndig = np.char.str_len(lead)
            raw[f"{c}|last_digit"] = v % 10
            raw[f"{c}|first_decimal"] = (v // 10) % 10
            raw[f"{c}|cents"] = v % 100
            raw[f"{c}|int_10x"] = v // 10
            raw[f"{c}|int"] = whole
            raw[f"{c}|int_div10"] = whole // 10
            raw[f"{c}|lead1"] = ndig * 10 + np.array([int(s[0]) for s in lead])
            raw[f"{c}|lead2"] = ndig * 100 + np.array([int(s[:2]) for s in lead])
    for c in INTS:
        v = df[c].to_numpy().astype(np.int64)
        raw[f"{c}|value"] = v
        if digits and v.max() >= 20:
            raw[f"{c}|last_digit"] = v % 10
    for c in CATS:
        raw[f"{c}|value"] = df[c].astype(str).to_numpy()

    keys: dict[str, np.ndarray] = {}
    seen: dict[tuple, str] = {}
    for name, arr in raw.items():
        codes = np.unique(arr, return_inverse=True)[1].astype(np.int64)
        col = name.split("|")[0]
        # 같은 열에서 행을 똑같이 묶는 키는 하나만 남긴다
        first = np.unique(codes, return_index=True)[1]
        sig = (col, len(first), hash(np.sort(first).tobytes()))
        if sig in seen:
            continue
        seen[sig] = name
        keys[name] = codes
    return keys


class FoldCounts:
    """키별·겹별 사기 건수와 행 수. 한 번 세어 두고 겹을 빼는 식으로 사기율을 만든다."""

    def __init__(self, codes: np.ndarray, y: np.ndarray, fold: np.ndarray, n_folds: int):
        n_train = len(y)
        self.codes, self.fold, self.n_train = codes, fold, n_train
        size = int(codes.max()) + 1
        self.cnt = np.zeros((n_folds, size))
        self.pos = np.zeros((n_folds, size))
        for g in range(n_folds):
            rows = codes[:n_train][fold == g]
            self.cnt[g] = np.bincount(rows, minlength=size)
            self.pos[g] = np.bincount(rows, weights=y[fold == g], minlength=size)
        self.cnt_all, self.pos_all = self.cnt.sum(0), self.pos.sum(0)
        self.pos_fold, self.cnt_fold = self.pos.sum(1), self.cnt.sum(1)

    def encode(self, holdout: int | None) -> np.ndarray:
        """holdout 겹을 검증으로 뗐을 때의 사기율(로짓). holdout=None이면 전체 학습용.

        학습 행은 (holdout과) 자기 겹을 뺀 나머지에서, 검증·테스트 행은 holdout만 뺀 나머지에서 센다.
        """
        codes, fold, n = self.codes, self.fold, self.n_train
        cnt, pos = self.cnt_all.copy(), self.pos_all.copy()
        n_tot, p_tot = self.cnt_fold.sum(), self.pos_fold.sum()
        if holdout is not None:
            cnt -= self.cnt[holdout]; pos -= self.pos[holdout]
            n_tot -= self.cnt_fold[holdout]; p_tot -= self.pos_fold[holdout]
        out = np.empty(len(codes))
        # 검증·테스트 행(그리고 일단 전부)
        out[:] = (pos[codes] + M * p_tot / n_tot) / (cnt[codes] + M)
        # 학습 행은 자기 겹을 추가로 뺀다
        for g in range(len(self.cnt)):
            if g == holdout:
                continue
            rows = np.flatnonzero(fold == g)
            c = codes[rows]
            prior = (p_tot - self.pos_fold[g]) / (n_tot - self.cnt_fold[g])
            out[rows] = (pos[c] - self.pos[g][c] + M * prior) / (cnt[c] - self.cnt[g][c] + M)
        out = np.clip(out, 1e-6, 1 - 1e-6)
        return np.log(out / (1 - out))


def build_matrix(df, keys, counts, holdout):
    """모델 입력. 인코딩(로짓) + 원본 수치 + 값 빈도 + 참/거짓."""
    cols = {f"te|{k}": fc.encode(holdout) for k, fc in counts.items()}
    for c in DECIMALS + INTS:
        cols[c] = df[c].to_numpy(dtype=float)
    for c in DECIMALS:  # 정확히 같은 값이 학습+테스트에 몇 번 나오는지(정답을 쓰지 않는다)
        code = keys[f"{c}|value"]
        cols[f"freq|{c}"] = np.log(np.bincount(code)[code])
        cols[f"cents|{c}"] = np.round(df[c].to_numpy() * 100) % 100
    for c in FLAGS:
        cols[c] = df[c].to_numpy(dtype=float)
    return pd.DataFrame(cols)


def fit_glm(X, y, fit_rows):
    """분위 정규화 뒤 로지스틱 회귀. 로그 오즈를 돌려주는 함수를 반환한다."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler

    raw = [c for c in X.columns if not c.startswith("te|") and c not in FLAGS]
    te = [c for c in X.columns if c.startswith("te|")]
    from sklearn.compose import ColumnTransformer
    pre = ColumnTransformer([
        ("raw", QuantileTransformer(n_quantiles=200, output_distribution="normal",
                                    subsample=200_000, random_state=SEED), raw),
        ("te", StandardScaler(), te),
        ("flag", "passthrough", list(FLAGS)),
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.1))
    model.fit(X.iloc[fit_rows], y[fit_rows])
    return lambda rows: model.decision_function(X.iloc[rows])


def fit_trees(X, y, margin, fit_rows, stop_rows=None, rounds=None):
    """GLM의 로그 오즈에서 출발하는 잎 4개 나무. 나무는 GLM이 놓친 부분만 맞춘다."""
    import lightgbm as lgb

    params = dict(objective="binary", learning_rate=0.03, num_leaves=4, min_data_in_leaf=200,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
                  seed=SEED, verbose=-1, num_threads=4)
    dfit = lgb.Dataset(X.iloc[fit_rows], y[fit_rows], init_score=margin[fit_rows])
    if rounds is None:
        dstop = lgb.Dataset(X.iloc[stop_rows], y[stop_rows], init_score=margin[stop_rows])
        booster = lgb.train(params, dfit, 5000, valid_sets=[dstop],
                            callbacks=[lgb.early_stopping(200, verbose=False)])
        rounds = booster.best_iteration
    else:
        booster = lgb.train(params, dfit, rounds)
    return (lambda rows: margin[rows] + booster.predict(X.iloc[rows], raw_score=True,
                                                        num_iteration=rounds)), rounds


def run(name: str, stage: str = "trees", digits: bool = True, shuffle: bool = False) -> float:
    from sklearn.model_selection import train_test_split

    train, test = load()
    y = train[TARGET].astype(int).to_numpy()
    n = len(train)
    fold = np.empty(n, dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(y, y)):
        fold[va] = k
    if shuffle:  # 정답을 섞으면 배울 것이 없으므로 AUC가 0.5 근처여야 한다. 높으면 누수.
        y = np.random.default_rng(SEED).permutation(y)

    both = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    keys = make_keys(both, digits)
    counts = {k: FoldCounts(code, y.astype(float), fold, FOLDS) for k, code in keys.items()}
    print(f"키 {len(keys)}개", flush=True)

    all_rows, test_rows = np.arange(n), np.arange(n, len(both))
    oof, scores, rounds_used = np.zeros(n), [], []
    for f in range(1 if shuffle else FOLDS):
        X = build_matrix(both, keys, counts, holdout=f)
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        glm = fit_glm(X, y, tr)
        margin = np.zeros(len(both))
        margin[:] = glm(np.arange(len(both)))
        if stage == "glm":
            oof[va] = margin[va]
            note = ""
        else:
            fit_rows, stop_rows = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=SEED)
            predict, r = fit_trees(X, y, margin, fit_rows, stop_rows)
            oof[va] = predict(va)
            rounds_used.append(r)
            note = f" (GLM {roc_auc_score(y[va], margin[va]):.5f}, 나무 {r}개)"
        scores.append(roc_auc_score(y[va], oof[va]))
        print(f"  fold {f}: AUC {scores[-1]:.5f}{note}", flush=True)
    if shuffle:
        print(f"{name}: 정답을 섞었을 때 fold 0 AUC {scores[0]:.5f} (0.5 근처면 누수 없음)")
        return scores[0]

    total = roc_auc_score(y, oof)
    print(f"{name}: CV AUC {total:.5f} (겹별 평균 {np.mean(scores):.5f} ± {np.std(scores):.5f})")

    # 테스트 예측은 전체 학습 행으로 한 번 더 맞춘 모델로 만든다(인코딩을 더 많은 행에서 센다).
    X = build_matrix(both, keys, counts, holdout=None)
    glm = fit_glm(X, y, all_rows)
    margin = np.zeros(len(both))
    margin[:] = glm(np.arange(len(both)))
    if stage == "glm":
        pred = margin[test_rows]
    else:
        predict, _ = fit_trees(X, y, margin, all_rows, rounds=int(1.25 * np.mean(rounds_used)))
        pred = predict(test_rows)

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    np.save(out / f"{name}_oof.npy", oof)
    pd.DataFrame({"id": test["id"], TARGET: 1 / (1 + np.exp(-pred))}).to_csv(
        out / f"{name}_submission.csv", index=False)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--stage", choices=["glm", "trees"], default="trees")
    ap.add_argument("--no-digits", action="store_true", help="자릿수 키 없이 값 키만 쓴다")
    ap.add_argument("--shuffle-check", action="store_true", help="정답을 섞어 누수를 점검한다")
    a = ap.parse_args()
    run(a.name, a.stage, not a.no_digits, a.shuffle_check)

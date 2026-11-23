"""Playground S6E11 (카드 사기 탐지) — 5겹 교차검증으로 학습하고 제출 파일을 만든다.

사용법:  python train.py <실험이름> [--model logreg|lgbm] [--features base|v1]
결과:    out/<실험이름>_oof.npy, out/<실험이름>_submission.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
TARGET = "is_fraud"
CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
SEED = 42
FOLDS = 5


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(HERE / "data" / "train.csv")
    test = pd.read_csv(HERE / "data" / "test.csv")
    return train, test


def features_base(df: pd.DataFrame) -> pd.DataFrame:
    """원본 열 그대로. 범주형은 category, 참/거짓은 0/1."""
    X = df.drop(columns=["id", TARGET], errors="ignore").copy()
    for c in CATS:
        X[c] = X[c].astype("category")
    for c in X.select_dtypes("bool").columns:
        X[c] = X[c].astype("int8")
    return X


FEATURES = {"base": features_base}


def fit_predict_logreg(Xtr, ytr, Xva, Xte):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cats = [c for c in Xtr.columns if Xtr[c].dtype.name == "category"]
    nums = [c for c in Xtr.columns if c not in cats]
    model = make_pipeline(
        ColumnTransformer([("num", StandardScaler(), nums),
                           ("cat", OneHotEncoder(handle_unknown="ignore"), cats)]),
        LogisticRegression(max_iter=1000, C=1.0),
    )
    model.fit(Xtr, ytr)
    return model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1], None


def fit_predict_lgbm(Xtr, ytr, Xva, Xte, yva=None):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=5000, learning_rate=0.05, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, verbose=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1], model.best_iteration_


def run(name: str, model: str, feats: str) -> float:
    train, test = load()
    y = train[TARGET].astype(int).to_numpy()
    both = FEATURES[feats](pd.concat([train, test], ignore_index=True))  # 범주 목록을 맞추려고 함께 변환
    X, Xte = both.iloc[: len(train)], both.iloc[len(train):]

    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    scores = []
    for k, (tr, va) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y)):
        if model == "logreg":
            p_va, p_te, it = fit_predict_logreg(X.iloc[tr], y[tr], X.iloc[va], Xte)
        else:
            p_va, p_te, it = fit_predict_lgbm(X.iloc[tr], y[tr], X.iloc[va], Xte, y[va])
        oof[va] = p_va
        pred += p_te / FOLDS
        scores.append(roc_auc_score(y[va], p_va))
        print(f"  fold {k}: AUC {scores[-1]:.5f}" + (f" (나무 {it}개)" if it else ""), flush=True)

    total = roc_auc_score(y, oof)
    print(f"{name}: CV AUC {total:.5f} (겹별 평균 {np.mean(scores):.5f} ± {np.std(scores):.5f})")
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    np.save(out / f"{name}_oof.npy", oof)
    pd.DataFrame({"id": test["id"], TARGET: pred}).to_csv(out / f"{name}_submission.csv", index=False)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--model", choices=["logreg", "lgbm"], default="lgbm")
    ap.add_argument("--features", choices=sorted(FEATURES), default="base")
    a = ap.parse_args()
    run(a.name, a.model, a.features)

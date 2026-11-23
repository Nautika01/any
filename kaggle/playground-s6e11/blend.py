"""여러 실험의 예측을 순위 평균으로 섞는다. 가중치는 모두 같다(OOF로 가중치를 맞추지 않는다).

사용법:  python blend.py <새 이름> <실험이름> <실험이름> ...
결과:    out/<새 이름>_oof.npy, out/<새 이름>_submission.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train import FOLDS, HERE, SEED, TARGET


def rank01(a: np.ndarray) -> np.ndarray:
    return rankdata(a) / len(a)


def blend(name: str, parts: list[str]) -> float:
    out = HERE / "out"
    y = pd.read_csv(HERE / "data" / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    oof = np.mean([rank01(np.load(out / f"{p}_oof.npy")) for p in parts], axis=0)
    subs = [pd.read_csv(out / f"{p}_submission.csv") for p in parts]
    assert all((s["id"] == subs[0]["id"]).all() for s in subs)
    pred = np.mean([rank01(s[TARGET].to_numpy()) for s in subs], axis=0)

    scores = [roc_auc_score(y[va], oof[va])
              for _, va in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(y, y)]
    total = roc_auc_score(y, oof)
    print(f"{name}: CV AUC {total:.5f} (겹별 평균 {np.mean(scores):.5f} ± {np.std(scores):.5f})")
    np.save(out / f"{name}_oof.npy", oof)
    pd.DataFrame({"id": subs[0]["id"], TARGET: pred}).to_csv(out / f"{name}_submission.csv", index=False)
    return total


if __name__ == "__main__":
    blend(sys.argv[1], sys.argv[2:])

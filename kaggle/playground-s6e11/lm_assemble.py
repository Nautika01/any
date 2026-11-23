"""Kaggle에서 돌린 언어 모델(lm_kaggle.py) 결과를 모아 다른 실험과 같은 형식으로 바꾼다.

사용법:
    python lm_assemble.py <실험이름> [--dir out/lm]   # 겹별 결과 → out/<실험이름>_oof.npy, _submission.csv
    python lm_assemble.py --make-notebook             # lm_kaggle.py를 담은 lm_kaggle.ipynb를 다시 만든다

겹이 일부만 끝났으면 끝난 겹의 AUC만 보여 주고 파일은 만들지 않는다(OOF가 전 행을 덮어야 혼합에 쓸 수 있다).
테스트 예측은 겹 모델들의 로그 오즈 평균이고, 전체 행으로 다시 학습한 결과(fold all)가 있으면 그것과 순위 평균한다.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train import FOLDS, HERE, SEED, TARGET


def log_odds(ll: np.ndarray, prior: float) -> np.ndarray:
    """lm_kaggle.log_odds와 같다. ll은 [n, 열 순서 수, 2(정상, 사기)] 로그 가능도."""
    ll = ll.astype(np.float64)
    return (np.logaddexp.reduce(ll[:, :, 1], axis=1) - np.logaddexp.reduce(ll[:, :, 0], axis=1)
            + np.log(prior / (1 - prior)))


def assemble(name: str, src) -> float | None:
    train = pd.read_csv(HERE / "data" / "train.csv", usecols=[TARGET])
    test = pd.read_csv(HERE / "data" / "test.csv", usecols=["id"])
    y = train[TARGET].astype(int).to_numpy()
    oof, test_parts, scores, missing = np.full(len(y), np.nan), [], [], []
    for f, (_, va) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(y, y)):
        meta_path = src / f"lm_fold{f}.json"
        if not meta_path.exists():
            missing.append(f)
            continue
        meta = json.loads(meta_path.read_text())
        ll_oof, ll_test = np.load(src / f"lm_fold{f}_oof.npy"), np.load(src / f"lm_fold{f}_test.npy")
        if meta.get("limit_rows") or len(ll_oof) != len(va) or len(ll_test) != len(test):
            raise SystemExit(f"fold {f}: 행 수가 맞지 않는다(시험용 실행 결과인가?)")
        oof[va] = log_odds(ll_oof, meta["prior"])
        test_parts.append(log_odds(ll_test, meta["prior"]))
        scores.append(roc_auc_score(y[va], oof[va]))
        print(f"  fold {f}: AUC {scores[-1]:.5f} (에폭 {meta['epochs']}, 열 순서 {meta['k_oof']}/{meta['k_test']}, "
              f"{meta['hours']}시간)")
    if missing:
        print(f"{name}: 겹 {missing}이 아직 없다. 끝난 겹 평균 {np.mean(scores):.5f}" if scores else f"{name}: 결과가 없다")
        return None

    total = roc_auc_score(y, oof)
    print(f"{name}: CV AUC {total:.5f} (겹별 평균 {np.mean(scores):.5f} ± {np.std(scores):.5f})")
    pred = np.mean(test_parts, axis=0)
    rank = rankdata(pred) / len(pred)
    if (src / "lm_foldall.json").exists():
        refit = log_odds(np.load(src / "lm_foldall_test.npy"), json.loads((src / "lm_foldall.json").read_text())["prior"])
        rank = (rank + rankdata(refit) / len(refit)) / 2
        print("  전체 행 재학습 결과와 순위 평균했다")
    out = HERE / "out"
    np.save(out / f"{name}_oof.npy", oof)
    pd.DataFrame({"id": test["id"], TARGET: rank}).to_csv(out / f"{name}_submission.csv", index=False)
    return total


NOTEBOOK_INTRO = """# S6E11 언어 모델 (정답을 먼저 쓰는 GPT-2)

설정: **Accelerator = GPU T4 x2**, **Internet = On**, 입력에 대회 데이터 `playground-series-s6e11` 추가.

아래 `FOLDS`에 이번 세션에서 돌릴 겹을 적고(GPU가 2개라 한 번에 2개까지) **Save Version → Save & Run All**로 실행한다.
세 번에 나눠 돌린다: `[0, 1]`, `[2, 3]`, `[4]`. 끝나면 Output 탭의 `lm_fold*.npy`, `lm_fold*.json`을 내려받는다.
"""

NOTEBOOK_RUN = """import subprocess

FOLDS = [0, 1]  # 이번 세션에서 돌릴 겹(최대 2개). 다음 세션: [2, 3], 그다음: [4]

procs = [subprocess.Popen(["python", "lm_kaggle.py", "--fold", str(f), "--gpu", str(i)])
         for i, f in enumerate(FOLDS)]
codes = [p.wait() for p in procs]
print("종료 코드", codes)
assert all(c == 0 for c in codes)
"""


def make_notebook() -> None:
    code = (HERE / "lm_kaggle.py").read_text()
    cell = lambda kind, text: {"cell_type": kind, "metadata": {}, "source": text,
                               **({"outputs": [], "execution_count": None} if kind == "code" else {})}
    nb = {"cells": [cell("markdown", NOTEBOOK_INTRO),
                    cell("code", "%%writefile lm_kaggle.py\n" + code),
                    cell("code", "!python lm_kaggle.py --check --limit-rows 2000"),
                    cell("code", NOTEBOOK_RUN)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (HERE / "lm_kaggle.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print("lm_kaggle.ipynb를 만들었다")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--dir", default=str(HERE / "out" / "lm"))
    ap.add_argument("--make-notebook", action="store_true")
    a = ap.parse_args()
    if a.make_notebook:
        make_notebook()
    else:
        from pathlib import Path
        assemble(a.name or "lm", Path(a.dir))

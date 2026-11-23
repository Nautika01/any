"""Playground S6E11 — RealMLP(pytabkit). 나무·GLM과 다르게 틀리는 모델을 혼합에 넣기 위한 것.

값을 범주로 넣어 신경망이 값마다 벡터를 배우게 한다(목표 인코딩의 신경망판).
입력 구성과 하이퍼파라미터는 공개 노트북(yekenot의 RealMLP · PyTabKit, davutkursun의 정리)을 따랐고 코드는 새로 썼다.

누수 방지:
  - 조합 범주의 사기율은 train_te.encode로 겹 안에서만 센다.
  - pytabkit은 기본으로 검증 세트에서 가장 좋은 에폭을 고르는데, 여기서는 검증 세트가 채점 겹이다.
    stop_epoch를 n_epochs와 같게 두어 마지막 에폭을 그대로 쓴다. 채점 겹은 아무것도 고르지 않는다.

사용법:  python train_mlp.py <실험이름> [--seed N] [--folds 0 1 ...] [--n-ens N] [--digit-cats]
결과:    out/<실험이름>_oof.npy, out/<실험이름>_submission.csv (테스트 예측은 5겹 모델의 평균)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train import CATS, FLAGS, FOLDS, HERE, SEED, TARGET, load
from train_te import DECIMALS, INTS, encode, make_groups

CENTS = ["amount_usd", "hours_since_last_txn", "velocity_score", "distance_from_home_km"]
BINS = [("merchant_risk_score", 7), ("hours_since_last_txn", 500), ("hours_since_last_txn", 1000)]
CROSSES = [("merchant_category", "card_type", "txn_count_last_24h"), ("merchant_category", "customer_age")]

PARAMS = dict(verbosity=0, val_metric_name="1-auc_ovr",
              n_ens=8, n_epochs=2, stop_epoch=2, batch_size=288, use_early_stopping=False,
              lr=0.053, wd=0.018, sq_mom=0.988, lr_sched="flat_anneal", wd_sched="cos_log_15",
              first_layer_lr_factor=0.25, embedding_size=5, max_one_hot_cat_size=16,
              hidden_sizes=[512, 256, 128], act="silu", p_drop=0.05, p_drop_sched="pow5",
              plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=0.75,
              ls_eps=0.01, ls_eps_sched="sqrt_cos", add_front_scale=False,
              bias_init_mode="neg-uniform-dynamic-2",
              tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"])


def build_inputs(both: pd.DataFrame, n_train: int, digit_cats: bool = False):
    """정답을 쓰지 않는 입력. 반환: (표, 범주형 열 이름, 겹 안에서 인코딩할 키 코드)."""
    X = pd.DataFrame(index=both.index)
    cats: list[str] = []

    def add_cat(name: str, values: np.ndarray) -> None:
        # 코드는 학습 행에서만 배운다. 학습에 없던 테스트 값은 한 범주("-1")로 모은다.
        codes, uniques = pd.factorize(values[:n_train])
        lookup = pd.Series(np.arange(len(uniques)), index=uniques)
        full = lookup.reindex(values).fillna(-1).astype("int32").to_numpy()
        X[name] = pd.Categorical(full)
        cats.append(name)

    for c in DECIMALS + INTS:
        X[c] = both[c].astype("float32")
    for c in CATS:
        add_cat(c, both[c].astype(str).to_numpy())
    for c in FLAGS:
        add_cat(c, both[c].astype(int).to_numpy())
    for c in DECIMALS + INTS:  # 정수 부분을 범주로
        add_cat(f"{c}|int", np.floor(both[c].to_numpy()).astype(np.int64))
    for c in CENTS:  # 센트를 범주로
        add_cat(f"{c}|cents", (np.round(both[c].to_numpy() * 100) % 100).astype(np.int64))
    if digit_cats:  # 숫자가 쓰인 자릿수를 범주로: 나머지 열의 센트, 첫 소수 자리, 정수부 끝 두 자리, 앞 두 자리
        for c in DECIMALS:
            v = np.round(both[c].to_numpy() * 100).astype(np.int64)
            whole = v // 100
            if c not in CENTS:
                add_cat(f"{c}|cents", v % 100)
            add_cat(f"{c}|first_decimal", (v // 10) % 10)
            add_cat(f"{c}|int_last2", whole % 100)
            lead = whole.astype(str)
            add_cat(f"{c}|lead2", np.char.str_len(lead) * 100 + np.array([int(t[:2]) for t in lead]))
    for c, nb in BINS:  # 분위 구간(정답을 쓰지 않는다). 경계는 학습 행에서만 잡는다
        edges = np.unique(np.quantile(both[c].to_numpy()[:n_train], np.linspace(0, 1, nb + 1)[1:-1]))
        add_cat(f"{c}|q{nb}", np.searchsorted(edges, both[c].to_numpy(), side="right"))

    te_keys: dict[str, np.ndarray] = {}
    for cols in CROSSES:
        joined = both[cols[0]].astype(str)
        for c in cols[1:]:
            joined = joined + "_" + both[c].astype(str)
        name = "x".join(cols)
        add_cat(name, joined.to_numpy())
        te_keys[f"te|{name}"] = np.unique(joined.to_numpy(), return_inverse=True)[1].astype(np.int64)
    hours = np.round(both["hours_since_last_txn"].to_numpy() * 100).astype(np.int64)
    te_keys["te|hours_since_last_txn"] = np.unique(hours, return_inverse=True)[1].astype(np.int64)
    return X, cats, te_keys


def run(name: str, seed: int = SEED, only_folds: list[int] | None = None, n_ens: int = 8,
        digit_cats: bool = False) -> float:
    import torch
    from pytabkit import RealMLP_TD_Classifier

    torch.set_num_threads(4)
    train, test = load()
    y = train[TARGET].astype(int).to_numpy()
    n = len(train)
    fold = np.empty(n, dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(y, y)):
        fold[va] = k

    both = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    X, cats, te_keys = build_inputs(both, n, digit_cats)
    print(f"입력 {X.shape[1] + len(te_keys)}열, 그중 범주형 {len(cats)}열", flush=True)

    oof, pred, scores = np.zeros(n), np.zeros(len(test)), []
    test_rows = np.arange(n, len(both))
    folds = only_folds if only_folds is not None else list(range(FOLDS))
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    for f in folds:
        start = time.time()
        part = out / f"{name}_fold{f}.npz"  # 겹마다 저장해 두고, 다시 실행하면 이어서 한다
        if part.exists():
            saved = np.load(part)
            va = np.flatnonzero(fold == f)
            oof[va], scores = saved["oof"], scores + [roc_auc_score(y[va], saved["oof"])]
            pred += saved["test"] / len(folds)
            print(f"  fold {f}: AUC {scores[-1]:.5f} (저장된 결과)", flush=True)
            continue
        group = make_groups(fold, f, enc_seed=seed)  # 학습 행을 무작위 5묶음으로 나눠 센다
        Xf = X.copy()
        for k, code in te_keys.items():
            Xf[k] = encode(code, y.astype(float), group, FOLDS).astype("float32")
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        model = RealMLP_TD_Classifier(**{**PARAMS, "n_ens": n_ens}, random_state=seed, device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(Xf.iloc[tr], y[tr], Xf.iloc[va], y[va], cat_col_names=cats)
        oof[va] = model.predict_proba(Xf.iloc[va])[:, 1]
        p_test = model.predict_proba(Xf.iloc[test_rows])[:, 1]
        pred += p_test / len(folds)
        np.savez(part, oof=oof[va], test=p_test)
        scores.append(roc_auc_score(y[va], oof[va]))
        print(f"  fold {f}: AUC {scores[-1]:.5f} ({time.time() - start:.0f}s)", flush=True)

    if len(folds) < FOLDS:
        print(f"{name}: 일부 겹만 실행, 겹별 평균 {np.mean(scores):.5f}")
        return float(np.mean(scores))
    total = roc_auc_score(y, oof)
    print(f"{name}: CV AUC {total:.5f} (겹별 평균 {np.mean(scores):.5f} ± {np.std(scores):.5f})")
    np.save(out / f"{name}_oof.npy", oof)
    pd.DataFrame({"id": test["id"], TARGET: pred}).to_csv(out / f"{name}_submission.csv", index=False)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="일부 겹만 돌려 볼 때")
    ap.add_argument("--n-ens", type=int, default=8, help="함께 학습해 평균하는 신경망 수")
    ap.add_argument("--digit-cats", action="store_true", help="자릿수(센트·첫 소수 자리·앞뒤 두 자리)를 범주로 추가")
    a = ap.parse_args()
    run(a.name, a.seed, a.folds, a.n_ens, a.digit_cats)

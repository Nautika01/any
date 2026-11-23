"""Playground S6E11 — 정답을 먼저 쓰는 언어 모델(GPT-2 small). Kaggle GPU 노트북에서 겹 하나를 학습·채점한다.

원리: 이 대회 데이터는 언어 모델이 "정답 → 각 열의 값" 순서로 써서 만든 합성 데이터다. 같은 방식으로
GPT-2가 정답 뒤에 행을 쓰도록 학습시킨 뒤, 한 행을 "사기"와 "정상" 두 정답 아래에서 각각 얼마나 그럴듯한지
계산해 베이즈 규칙으로 사기 로그 오즈를 얻는다. 분류용 층은 없다.

방법 출처: Chris Deotte(S6E10 1위 풀이), masayakawamata(S6E11 적용), megayak(행 압축). 코드는 새로 썼다.
행 압축: 열 이름 글자 대신 열마다 학습되는 토큰 하나를 쓰고, 출력은 실제 쓰이는 토큰(약 1,400개)만 대상으로 한다.
공개 구현과 다른 점: 원본 2만 행으로 먼저 학습하는 단계는 없다(대회 데이터만 쓴다).

누수 방지: 겹 f의 모델은 겹 f의 행을 학습에 쓰지 않는다. 사전 확률은 학습 행의 사기 비율이다.
분할은 다른 실험과 같은 StratifiedKFold(5, shuffle=True, random_state=42).

사용법(Kaggle 노트북, GPU T4 x2, 인터넷 켬, 대회 데이터 추가):
    python lm_kaggle.py --fold 0 --gpu 0          # 겹 하나. 두 GPU에 겹을 하나씩 동시에 돌린다
    python lm_kaggle.py --fold all --gpu 0        # 선택: 전체 행으로 다시 학습(테스트 예측만)
    python lm_kaggle.py --check                   # 행이 어떻게 토큰으로 바뀌는지 출력
결과:  <out>/lm_fold{f}_oof.npy, lm_fold{f}_test.npy  ([행, 열 순서 수, 2] 로그 가능도), lm_fold{f}.json
       저장소의 lm_assemble.py가 이 파일들을 모아 out/<이름>_oof.npy와 제출 파일로 바꾼다.

시간: 세션 제한 안에 끝나도록 --max-hours(기본 10.5)에 맞춰 에폭 수와 채점 횟수를 스스로 줄인다.
에폭마다 저장하므로 중간에 끊겨도 같은 명령으로 이어서 한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--fold", default="0", help="0~4 또는 all")
ap.add_argument("--gpu", default=None, help="쓸 GPU 번호")
ap.add_argument("--data", default=None, help="train.csv, test.csv가 있는 폴더")
ap.add_argument("--out", default=None, help="결과 폴더")
ap.add_argument("--max-hours", type=float, default=10.5, help="학습+채점에 쓸 시간 한도")
ap.add_argument("--epochs", type=int, default=0, help="0이면 시간 한도에 맞춰 최대 8까지 자동")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--k-oof", type=int, default=10, help="검증 행을 채점할 열 순서 수(최대)")
ap.add_argument("--k-test", type=int, default=4, help="테스트 행을 채점할 열 순서 수(최대)")
ap.add_argument("--limit-rows", type=int, default=0, help="동작 시험용: 학습·테스트 행 수를 줄인다")
ap.add_argument("--check", action="store_true")
ap.add_argument("--seed", type=int, default=42)
ARGS = ap.parse_args() if __name__ == "__main__" else ap.parse_args([])
if ARGS.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu  # torch를 불러오기 전에 정해야 한다

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "is_fraud"
MAX_EPOCHS = 8
TRAIN_SHARE = 0.72  # 시간 한도 중 학습에 쓰는 몫. 나머지는 채점
START = time.time()
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find_data() -> Path:
    if ARGS.data:
        return Path(ARGS.data)
    for base in (Path("/kaggle/input"), Path(__file__).resolve().parent / "data"):
        hits = sorted(p.parent for p in base.rglob("train.csv") if (p.parent / "test.csv").exists()) if base.exists() else []
        hits = [h for h in hits if "s6e11" in str(h)] or hits
        if hits:
            return hits[0]
    sys.exit("train.csv를 찾지 못했다. 노트북에 대회 데이터(playground-series-s6e11)를 추가하거나 --data를 준다.")


def out_dir() -> Path:
    if ARGS.out:
        p = Path(ARGS.out)
    elif Path("/kaggle/working").exists():
        p = Path("/kaggle/working")
    else:
        p = Path(__file__).resolve().parent / "out" / "lm"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- 데이터 → 토큰
def build_tokens(data: Path) -> dict:
    """모든 행을 [행, 열, 칸] 토큰 표로 바꾼다. 칸 0은 열 토큰, 그 뒤는 값을 이루는 GPT-2 토큰, 빈칸은 -1.

    값은 CSV에 적힌 글자 그대로 읽는다(0.8과 0.80을 구별). 정답은 쓰지 않는다.
    """
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    read = lambda name: pd.read_csv(data / name, dtype=str, keep_default_na=False)
    train, test = read("train.csv"), read("test.csv")
    if ARGS.limit_rows:
        train, test = train.iloc[: ARGS.limit_rows], test.iloc[: max(200, ARGS.limit_rows // 10)]
    cols = [c for c in train.columns if c not in ("id", TARGET)]
    both = pd.concat([train[cols], test[cols]], ignore_index=True)

    per_col, used = [], set()
    for c in cols:
        codes, uniques = pd.factorize(both[c])
        ids = [tok.encode(" " + (str(v).strip() or "None")) for v in uniques]
        used.update(i for x in ids for i in x)
        per_col.append((codes, ids))
    vocab = np.array(sorted(used))  # 실제로 쓰이는 GPT-2 토큰만
    remap = {g: i for i, g in enumerate(vocab)}
    V, NC = len(vocab), len(cols)
    width = 1 + max(len(x) for _, ids in per_col for x in ids)
    spec = dict(V=V, END=V, PAD=V + 1, LAB=V + 2, COL0=V + 4, NC=NC, NTOK=V + 4 + NC, WIDTH=width)

    toks = np.full((len(both), NC, width), -1, np.int16)
    for j, (codes, ids) in enumerate(per_col):
        table = np.full((len(ids), width), -1, np.int16)
        table[:, 0] = spec["COL0"] + j
        for i, x in enumerate(ids):
            table[i, 1 : 1 + len(x)] = [remap[g] for g in x]
        toks[:, j, :] = table[codes]

    y = train[TARGET].astype(float).astype(int).to_numpy()
    col_ids = [tok.encode(" " + c) for c in cols]  # 열 토큰의 초깃값: 열 이름 토큰들의 평균
    lab_ids = [tok.encode(" False")[0], tok.encode(" True")[0]]
    return dict(toks=toks, y=y, n_train=len(train), n_test=len(test), vocab=vocab, cols=cols,
                col_ids=col_ids, lab_ids=lab_ids, comma=tok.encode(",")[0], tok=tok, spec=spec,
                test_ids=test["id"].to_numpy())


class Rows:
    """토큰 표를 장치에 올려 두고, 열 순서를 섞은 입력과 목표를 만든다."""

    def __init__(self, g: dict):
        self.s = g["spec"]
        self.t = torch.from_numpy(g["toks"]).to(DEV)
        self.lens = (g["toks"] >= 0).sum((1, 2)) + 1  # 정답 토큰 포함 길이

    def perm(self, n: int, gen: torch.Generator) -> torch.Tensor:
        return torch.rand(n, self.s["NC"], device=DEV, generator=gen).argsort(1)

    def assemble(self, idx: torch.Tensor, lab: torch.Tensor, perm: torch.Tensor, T: int):
        """입력 [B, T]와 목표 [B, T]. 자리 0은 정답 토큰, 그 뒤로 (열 토큰, 값 토큰들)이 perm 순서로 온다.

        목표: 값 토큰 다음 자리는 그 값의 다음 토큰, 값의 마지막 토큰 다음은 END.
        정답 자리와 빈칸은 채점하지 않는다(-100). 열 토큰은 입력으로만 쓴다: 열 순서는 우리가 섞은 것이라 정보가 없다.
        """
        s, B = self.s, idx.numel()
        seg = self.t[idx].long().gather(1, perm[:, :, None].expand(-1, -1, s["WIDTH"])).reshape(B, -1)
        valid = seg >= 0
        pos = torch.where(valid, valid.cumsum(1), torch.full_like(seg, T + 1))  # T+1은 버리는 칸
        buf = torch.full((B, T + 2), s["PAD"], dtype=torch.long, device=DEV)
        buf.scatter_(1, pos, seg.clamp_min(0))
        buf[:, 0] = s["LAB"] + lab
        cur, nxt = buf[:, :T], buf[:, 1 : T + 1]
        tgt = torch.where(nxt < s["V"], nxt, torch.full_like(nxt, s["END"]))
        tgt = torch.where(cur == s["PAD"], torch.full_like(tgt, -100), tgt)
        tgt[:, 0] = -100
        return cur.contiguous(), tgt


# ---------------------------------------------------------------- 모델
class RowLM(torch.nn.Module):
    """GPT-2 small의 몸통을 그대로 쓰고, 입력·출력 토큰만 이 데이터에 쓰이는 것으로 바꾼다."""

    def __init__(self, g: dict, pretrained: bool = True):
        super().__init__()
        from transformers import GPT2Config, GPT2Model

        s = g["spec"]
        self.n_out = s["V"] + 1  # 값 토큰 + END
        self.body = GPT2Model.from_pretrained("gpt2") if pretrained else GPT2Model(GPT2Config())
        old = self.body.wte.weight.data
        new = torch.nn.Embedding(s["NTOK"], old.shape[1])
        with torch.no_grad():
            new.weight[: s["V"]] = old[torch.from_numpy(g["vocab"])]
            new.weight[s["END"]] = old[g["comma"]]
            new.weight[s["PAD"]] = old.mean(0)
            new.weight[s["LAB"]] = old[g["lab_ids"][0]]
            new.weight[s["LAB"] + 1] = old[g["lab_ids"][1]]
            for j, ids in enumerate(g["col_ids"]):
                new.weight[s["COL0"] + j] = old[torch.tensor(ids)].mean(0)
        self.body.wte = new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(input_ids=x).last_hidden_state
        # 로짓은 float32로 계산한다. 반정밀도에서는 큰 로짓이 뭉개져 가능도 차이가 틀어진다.
        with torch.autocast(DEV, enabled=False):
            return h.float() @ self.body.wte.weight[: self.n_out].float().T


def autocast():
    return torch.autocast("cuda", dtype=torch.float16) if DEV == "cuda" else torch.autocast("cpu", enabled=False)


# ---------------------------------------------------------------- 학습
def find_checkpoint(out: Path, name: str) -> Path | None:
    for base in (out, Path("/kaggle/input")):
        hits = sorted(base.rglob(name)) if base.exists() else []
        if hits:
            return hits[0]
    return None


def train(model: RowLM, R: Rows, y_dev: torch.Tensor, rows: np.ndarray, out: Path, tag: str) -> int:
    bs, lr = ARGS.batch, ARGS.lr
    per = len(rows) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    epochs, start_ep = ARGS.epochs, 0
    ck_name = f"lm_ckpt_{tag}.pt"
    ck = find_checkpoint(out, ck_name)
    if ck is not None:
        state = torch.load(ck, map_location=DEV, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start_ep, epochs = int(state["epoch"]) + 1, int(state["epochs"])
        log(f"{tag}: {ck}에서 이어서 한다(에폭 {start_ep}/{epochs} 끝난 상태)")
    budget = ARGS.max_hours * 3600 * TRAIN_SHARE
    gen = torch.Generator(device=DEV)
    model.train()
    for ep in range(start_ep, MAX_EPOCHS):
        if epochs and ep >= epochs:
            break
        gen.manual_seed(ARGS.seed * 1000 + ep)
        order = np.random.RandomState(ARGS.seed * 1000 + ep).permutation(rows)[: per * bs].reshape(per, bs)
        t_ep, running = time.time(), 0.0
        for step_in_ep in range(per):
            if not epochs and step_in_ep == min(300, per - 1):
                # 처음 300걸음의 속도로 에폭 시간을 재고, 시간 한도 안에 들어가는 에폭 수를 정한다
                ep_time = (time.time() - t_ep) / (step_in_ep + 1) * per
                epochs = int(max(1, min(MAX_EPOCHS, (budget - (time.time() - START)) // ep_time + 1)))
                log(f"{tag}: 에폭당 약 {ep_time / 60:.0f}분 → {epochs}에폭으로 정한다(한도 {ARGS.max_hours}시간)")
            step = ep * per + step_in_ep
            total = per * (epochs or MAX_EPOCHS)
            for group in opt.param_groups:  # 100걸음 예열 뒤 0까지 선형 감소
                group["lr"] = lr * min(1.0, (step + 1) / 100) * max(0.0, 1 - step / total)
            idx_np = order[step_in_ep]
            idx = torch.as_tensor(idx_np, device=DEV)
            inp, tgt = R.assemble(idx, y_dev[idx], R.perm(bs, gen), int(R.lens[idx_np].max()))
            with autocast():
                logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            running += loss.item() if (step_in_ep + 1) % 50 == 0 else 0.0
            if (step_in_ep + 1) % 2000 == 0:
                rate = (step_in_ep + 1) * bs / (time.time() - t_ep)
                log(f"{tag} 에폭 {ep + 1}/{epochs} 걸음 {step_in_ep + 1}/{per} 손실 {running / 40:.4f} {rate:.0f}행/초")
                running = 0.0
        tmp = out / (ck_name + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep, "epochs": epochs}, tmp)
        os.replace(tmp, out / ck_name)
        log(f"{tag}: 에폭 {ep + 1}/{epochs} 끝, {(time.time() - t_ep) / 60:.0f}분 (누적 {(time.time() - START) / 3600:.1f}시간)")
    return epochs


# ---------------------------------------------------------------- 채점
@torch.no_grad()
def score_once(model: RowLM, R: Rows, rows: np.ndarray, seed: int, bs: int = 256) -> np.ndarray:
    """열 순서 하나에 대해, 각 행의 로그 가능도를 정답이 정상일 때와 사기일 때 각각 구한다: [n, 2]."""
    model.eval()
    by_len = np.argsort(R.lens[rows], kind="stable")  # 길이가 비슷한 행끼리 묶어 빈칸을 줄인다
    rows_sorted = rows[by_len]
    out = np.zeros((len(rows), 2), np.float32)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for s in range(0, len(rows), bs):
        chunk = rows_sorted[s : s + bs]
        idx = torch.as_tensor(chunk, device=DEV)
        n = idx.numel()
        inp, tgt = R.assemble(idx, torch.zeros(n, dtype=torch.long, device=DEV), R.perm(n, gen),
                              int(R.lens[chunk].max()))
        both = torch.cat([inp, inp])
        both[n:, 0] = R.s["LAB"] + 1  # 같은 행, 같은 열 순서에 정답만 바꾼다
        with autocast():
            logits = model(both)
        nll = F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.repeat(2, 1).reshape(-1),
                              ignore_index=-100, reduction="none").view(2 * n, -1).sum(1)
        out[by_len[s : s + n], 0] = -nll[:n].cpu().numpy()
        out[by_len[s : s + n], 1] = -nll[n:].cpu().numpy()
    return out


def log_odds(ll: np.ndarray, prior: float) -> np.ndarray:
    """ll은 [n, K, 2]. 정답별로 열 순서에 대해 가능도를 평균(log-sum-exp)하고, 차이에 사전 로그 오즈를 더한다."""
    ll = ll.astype(np.float64)
    return (np.logaddexp.reduce(ll[:, :, 1], axis=1) - np.logaddexp.reduce(ll[:, :, 0], axis=1)
            + np.log(prior / (1 - prior)))


def score_all(model, R, val, test, y, prior, out: Path, tag: str) -> dict:
    """검증 행과 테스트 행을 번갈아 채점한다. 한 번 돌 때마다 저장하고, 시간 한도가 다가오면 멈춘다."""
    deadline = START + ARGS.max_hours * 3600
    passes = {"oof": [], "test": []}
    paths = {"oof": out / f"lm_{tag}_oof.npy", "test": out / f"lm_{tag}_test.npy"}
    rows = {"oof": val, "test": test}
    limit = {"oof": ARGS.k_oof if val is not None else 0, "test": ARGS.k_test}
    took = {"oof": 0.0, "test": 0.0}
    plan = [k for pair in zip(["oof"] * 4, ["test"] * 4) for k in pair] + ["oof"] * 20
    for kind in plan:
        if len(passes[kind]) >= limit[kind]:
            continue
        first = len(passes[kind]) == 0
        if not first and time.time() + took[kind] * 1.1 > deadline:
            continue  # 종류마다 적어도 한 번은 돈다. 그 뒤로는 시간이 남을 때만
        t0 = time.time()
        seed = ARGS.seed + (100 if kind == "oof" else 200) + 10 * len(passes[kind])
        passes[kind].append(score_once(model, R, rows[kind], seed))
        took[kind] = time.time() - t0
        np.save(paths[kind], np.stack(passes[kind], axis=1))
        msg = f"{tag}: {kind} 열 순서 {len(passes[kind])}개째, {took[kind] / 60:.0f}분"
        if kind == "oof":
            z = log_odds(np.stack(passes[kind], axis=1), prior)
            msg += f", AUC {roc_auc_score(y[val], z):.5f}"
        log(msg)
    result = dict(k_oof=len(passes["oof"]), k_test=len(passes["test"]))
    if passes["oof"]:
        result["auc"] = float(roc_auc_score(y[val], log_odds(np.stack(passes["oof"], axis=1), prior)))
    return result


# ---------------------------------------------------------------- 실행
def check(g: dict) -> None:
    """한 행이 어떤 입력>목표 쌍으로 바뀌는지 보여 주고, 목표 개수가 맞는지 확인한다."""
    R, s, tok = Rows(g), g["spec"], g["tok"]

    def name(t: int) -> str:
        if t == -100:
            return "·"
        if t == s["END"]:
            return "END"
        if t == s["PAD"]:
            return "PAD"
        if t in (s["LAB"], s["LAB"] + 1):
            return "정상" if t == s["LAB"] else "사기"
        if t >= s["COL0"]:
            return f"<{g['cols'][t - s['COL0']]}>"
        return repr(tok.decode([int(g["vocab"][t])]))

    gen = torch.Generator(device=DEV).manual_seed(0)
    idx = torch.tensor([0, 1], device=DEV)
    inp, tgt = R.assemble(idx, torch.tensor([0, 1], device=DEV), R.perm(2, gen), int(R.lens[[0, 1]].max()))
    print(f"값 토큰 {s['V']}종, 전체 토큰 {s['NTOK']}종, 행당 평균 {R.lens[: g['n_train']].mean():.1f}토큰, 최대 {R.lens.max()}")
    print(" ".join(f"{name(a)}>{name(b)}" for a, b in zip(inp[0].tolist(), tgt[0].tolist()))[:1500])
    for r in range(2):
        n_tgt, n_end = int((tgt[r] >= 0).sum()), int((tgt[r] == s["END"]).sum())
        # 값 토큰마다 목표 하나 + 열마다 END 하나 = 정답 토큰을 뺀 행 길이
        assert n_tgt == R.lens[r] - 1 and n_end == s["NC"], (n_tgt, R.lens[r], n_end)
    print("목표 개수 확인 통과")


def main() -> None:
    torch.manual_seed(ARGS.seed)
    data, out = find_data(), out_dir()
    log(f"장치 {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV == "cuda" else "") + f", 데이터 {data}, 결과 {out}")
    g = build_tokens(data)
    if ARGS.check:
        return check(g)
    y, n_train = g["y"], g["n_train"]
    tag = f"fold{ARGS.fold}"
    if (out / f"lm_{tag}.json").exists():
        return log(f"{tag}: 이미 끝난 결과가 있다")
    all_rows = np.arange(n_train)
    test = np.arange(n_train, n_train + g["n_test"])
    if ARGS.fold == "all":
        fit, val = all_rows, None
    else:
        fit, val = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n_train), y))[int(ARGS.fold)]
    R = Rows(g)
    y_dev = torch.as_tensor(y, device=DEV)
    model = RowLM(g).to(DEV)
    log(f"{tag}: 학습 {len(fit):,}행, 검증 {0 if val is None else len(val):,}행, 테스트 {len(test):,}행")
    epochs = train(model, R, y_dev, fit, out, tag)
    prior = float(y[fit].mean())
    result = score_all(model, R, val, test, y, prior, out, tag)
    result.update(fold=ARGS.fold, epochs=epochs, prior=prior, n_train=n_train, n_test=g["n_test"],
                  limit_rows=ARGS.limit_rows, hours=round((time.time() - START) / 3600, 2))
    (out / f"lm_{tag}.json").write_text(json.dumps(result, indent=2))
    (out / f"lm_ckpt_{tag}.pt").unlink(missing_ok=True)  # 끝났으면 큰 저장 파일은 지운다
    log(f"{tag}: 끝. {result}")


if __name__ == "__main__":
    main()

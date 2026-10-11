"""
[DDI 실험 - 03단계] 지문 융합 3방법(concat/sum/diff) × XGBoost 비교
====================================================================
이번 실험의 '핵심 질문':
  약물 A·B의 Morgan 지문을 '어떻게 합치면' DDI 이진 예측이 가장 잘 되나?
  분류기(XGBoost)와 데이터·분할은 똑같이 두고, '합치는 법'만 바꿔 공정 비교.

융합 3방법:
  (a) concat : [fpA, fpB]        4096차원   ← baseline
  (b) sum    : fpA + fpB         2048차원
  (c) diff   : |fpA - fpB|       2048차원   (= 두 지문의 대칭차/XOR)

공정 비교 설계:
  - 약물 지문은 1,705개만 1번 계산 후 재사용(쌍마다 lookup).
  - seed 5개 각각 80/20 stratified split(random_state=seed).
  - ★한 seed의 '같은 분할'을 세 방법이 공유 → 순수하게 융합법 차이만 비교.
  - 지표 4종: ROC-AUC, PR-AUC, F1(@0.5), accuracy → seed 평균±표준편차.

산출: results/fusion_results.json  (그림/표 보고서는 04에서 이 JSON을 읽어 생성)
실행: conda run -n admet python src/03_train_fusion.py
env : admet
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ★지문·융합·평가는 공용 모듈에서(02·03 공유, 중복 제거). 로직 동일.
from ddi_common import FP_BITS, XGB_PARAMS, build_fp_matrix, make_features, evaluate

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
JSON_OUT = os.path.join(BASE, "results", "fusion_results.json")

SEEDS = [0, 1, 2, 3, 4]          # 5개 반복 → 평균±표준편차
TEST_SIZE = 0.2


def main():
    t0 = time.time()
    print("[03] 융합 3방법 실험 시작 ...", flush=True)

    # 1) 데이터 로드
    smi = pd.read_csv(DRUG_SMI_CSV)
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"]))
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["drug_a"] = pairs["drug_a"].astype(str)
    pairs["drug_b"] = pairs["drug_b"].astype(str)
    y = pairs["label"].to_numpy()
    print(f"  약물 {len(id2smi):,} · 학습 쌍 {len(pairs):,} · 양성비율 {y.mean():.3f}", flush=True)

    # 2) 약물 지문(한 번만) + 약물ID→행번호
    drug_ids = list(id2smi.keys())
    did2row = {d: i for i, d in enumerate(drug_ids)}
    FP = build_fp_matrix(drug_ids, id2smi)
    print(f"  지문행렬 {FP.shape} (희소, nnz/행 평균 {FP.nnz/FP.shape[0]:.0f})", flush=True)

    # 3) 각 쌍 → 두 약물의 행번호
    idxA = pairs["drug_a"].map(did2row).to_numpy()
    idxB = pairs["drug_b"].map(did2row).to_numpy()

    # 4) 세 융합법 특징행렬(한 번만 만들고 seed마다 분할만 다르게)
    feats = make_features(FP, idxA, idxB)
    for k, v in feats.items():
        print(f"  특징[{k}] shape {v.shape} nnz {v.nnz:,}", flush=True)

    # 5) seed × 방법 반복
    methods = ["concat", "sum", "diff"]
    per = {m: {mt: [] for mt in ["roc_auc", "pr_auc", "f1", "accuracy"]} for m in methods}
    idx_all = np.arange(len(pairs))
    for seed in SEEDS:
        # ★세 방법이 '같은 분할' 공유 (공정 비교)
        tr, te = train_test_split(idx_all, test_size=TEST_SIZE, random_state=seed, stratify=y)
        for m in methods:
            ts = time.time()
            res = evaluate(feats[m][tr], y[tr], feats[m][te], y[te])
            for mt, val in res.items():
                per[m][mt].append(val)
            print(f"  seed {seed} · {m:6s} → ROC {res['roc_auc']:.4f} PR {res['pr_auc']:.4f} "
                  f"F1 {res['f1']:.4f} acc {res['accuracy']:.4f}  ({time.time()-ts:.0f}s)", flush=True)

    # 6) 집계(평균±표준편차)
    agg = {}
    for m in methods:
        agg[m] = {mt: {"mean": float(np.mean(per[m][mt])), "std": float(np.std(per[m][mt]))}
                  for mt in per[m]}

    out = {
        "config": {"n_pairs": int(len(pairs)), "n_drugs": int(len(id2smi)), "pos_frac": float(y.mean()),
                   "seeds": SEEDS, "test_size": TEST_SIZE, "fp_bits": FP_BITS,
                   "xgb_params": {k: v for k, v in XGB_PARAMS.items()},
                   "methods": {"concat": "[fpA,fpB] 4096", "sum": "fpA+fpB 2048", "diff": "|fpA-fpB| 2048"},
                   "runtime_s": round(time.time() - t0, 1)},
        "results": agg, "per_seed": per,
    }
    json.dump(out, open(JSON_OUT, "w"), ensure_ascii=False, indent=1)

    # 7) 콘솔 요약(baseline 대비 승패)
    print("\n================ 결과(seed 평균±std) ================")
    base = agg["concat"]["roc_auc"]["mean"]
    for m in methods:
        r = agg[m]
        tag = "(baseline)" if m == "concat" else f"(vs concat ROC {r['roc_auc']['mean']-base:+.4f})"
        print(f"  {m:6s} ROC {r['roc_auc']['mean']:.4f}±{r['roc_auc']['std']:.4f} · "
              f"PR {r['pr_auc']['mean']:.4f} · F1 {r['f1']['mean']:.4f} · acc {r['accuracy']['mean']:.4f}  {tag}")
    print(f"\n[03] 완료 ({out['config']['runtime_s']:.0f}s). 다음: 04 보고서 생성.")
    return out


if __name__ == "__main__":
    main()

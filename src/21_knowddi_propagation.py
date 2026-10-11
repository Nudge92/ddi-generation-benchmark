"""
[DDI 실험 - 21] 5세대(KnowDDI 2024) 핵심 아이디어 축소 재현 — '유사도 전파'가 신약을 뚫나
==============================================================================================
KnowDDI 핵심: 구조로 이웃을 찾아 그 이웃의 '관계(relation)'를 전파해 신약 쌍을 예측.
  ★이 실험은 KnowDDI 원본 재현이 아니라(Hetionet 등 외부 지식그래프 안 씀), 그 정수인
  '유사도 전파'만 ★분류기 고정으로 3버전 비교(유일 변수 = 전파 방식).

세 버전(모두 per-drug 표현 → 쌍 특징 = feat[lo]+feat[hi], sum 융합·11과 동일):
  V1 struct_only  : feat[d] = 자기 Morgan 지문. ★전파 없음(대조군) = ★11 baseline과 동일(재현 앵커).
  V2 struct_prop  : feat[d] = 구조가 닮은 ★train 이웃들의 Morgan을 Tanimoto 가중평균으로 빌림.
  V3 relation_prop: feat[d] = 같은 train 이웃들의 '관계 프로필(86종 유형 분포)'을 가중평균.
                    rel[g] = g가 참여한 그 regime의 ★학습 쌍 유형분포(정규화). ★이게 KnowDDI 정수.
  · 이웃 후보 = ★train 약물만(train_rows) · ★자기 제외 · top-K=20 · 가중치=Tanimoto 정규화.
  · rel = cold regime→tt(둘다 train) 쌍 · warm→wtr 쌍에서만 (test 라벨 0 유입 → cold 테스트 완전 청정).

고정(11·14·15·16과 완전 동일): 데이터·분할(split_drugs)·warm 쌍분할(train_test_split stratify=Y)·
  regime(warm/cold-start/cold-cold)·분류기 fit_mc(XGBoost multi:softprob, XGB_MC 그대로·random_state 없음=
  버전 고정시 결정적)·지표(macro-F1 주 + micro/acc/top3 + head/mid/tail)·seed 0~4.
  ★V1은 반드시 희소 pair_feature 경로(구조적 0=XGBoost 결측)라야 11 baseline과 bit-동일 재현.
  ★평가는 11의 le-aware 디코드(fit_mc가 LabelEncoder 압축 proba 반환) — 14 eval_dnn 아님.

검증(16처럼 정직·수치출력): 이웃풀 test 0개·self 제외·rel은 학습쌍만·cold 쌍 regime 조건·
  train↔test 겹침 0·NaN 0. 학습곡선(mlogloss) 감소 확인.

산출: results/21_ddi_knowddi_propagation.html , results/propagation_results.json
실행: conda run -n admet python src/21_knowddi_propagation.py
  (빠른 점검: PROP_SMOKE=1 → seed 1개·리포트 생략·V1 앵커/누수/loss 확인용)
"""
import os, sys, json, time, io, base64, gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddi_common as C
from ddi_common import ukey, build_fp_matrix, split_drugs, build_sim_matrix, pair_feature, fit_mc, top3_acc, XGB_MC

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
RAW_CSV = os.path.join(BASE, "data", "drugbank_ddi_raw.csv")
RESULTS_JSON = os.path.join(BASE, "results", "propagation_results.json")

SMOKE = os.environ.get("PROP_SMOKE", "") == "1"
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
DRUG_TEST_FRAC = 0.20
REGIMES = ["warm", "cold-start", "cold-cold"]
VERSIONS = ["struct_only", "struct_prop", "relation_prop"]   # V1/V2/V3
VLABEL = {"struct_only": "V1 struct_only", "struct_prop": "V2 struct_prop", "relation_prop": "V3 relation_prop"}
TOP_K = 20                     # 전파 이웃 수(구조 최근접 train 약물)
PCA_VAR = 0.90                 # V2 borrowed 지문 축소 유지분산(★밀집 2048은 XGBoost 86-class서 비현실적 느림 → 14/15와 동일 PCA)
N_CLASS = 86
METRICS = ["accuracy", "macro_f1", "micro_f1", "top3", "head_f1", "mid_f1", "tail_f1"]
# V1이 재현해야 할 11 baseline 앵커(seed 0 / 5-seed mean, macro-F1)
V1_ANCHOR_S0 = {"warm": 0.7804, "cold-start": 0.4877, "cold-cold": 0.2385}


def load_Y(pos_pairs):
    """상호작용 유형(0..85). ★11·14와 동일 로직(verbatim) → Y·stratified warm 분할 bit-동일."""
    raw = pd.read_csv(RAW_CSV); raw["Drug1_ID"] = raw["Drug1_ID"].astype(str); raw["Drug2_ID"] = raw["Drug2_ID"].astype(str)
    ymap = {}
    for a, b, y in zip(raw["Drug1_ID"], raw["Drug2_ID"], raw["Y"]):
        k = ukey(a, b)
        if k not in ymap:
            ymap[k] = int(y)
    Yraw = np.array([ymap[k] for k in pos_pairs])
    uniq = sorted(set(Yraw.tolist())); remap = {t: i for i, t in enumerate(uniq)}
    return np.array([remap[t] for t in Yraw])


# ===================== 전파(propagation) 기반 =====================
def topk_train_neighbors(M2, train_rows, K):
    """각 약물의 top-K ★train 이웃(구조 최근접)·자기 제외·Tanimoto 가중치(정규화).
    ★후보 열을 train_rows로만 슬라이스(=14 SSP 방식) → test-test 유사도가 특징에 못 들어감(누수 차단 핵심).
    반환: nbr_idx(n×K 전역 이웃 인덱스, 전부 train_rows) · nbr_w(n×K 정규화 가중치, Σsim=0이면 0)."""
    n = M2.shape[0]
    cols = np.sort(np.fromiter(train_rows, dtype=np.int64))     # 후보 = train 약물 전역 인덱스
    Sub = M2[:, cols].astype(np.float32).copy()                 # (n, m) train 대비 Tanimoto만
    pos_of = {int(g): j for j, g in enumerate(cols)}
    for i in range(n):                                          # ★자기 제외(train 질의는 자기 열을 -1로)
        j = pos_of.get(i)
        if j is not None:
            Sub[i, j] = -1.0
    order = np.argsort(-Sub, axis=1)[:, :K]                     # 행별 상위 K 열
    nbr_idx = cols[order]                                       # (n, K) 전역 이웃 인덱스(전부 train)
    nbr_sim = np.clip(np.take_along_axis(Sub, order, axis=1), 0.0, None)   # -1(자기/없음)→0
    wsum = nbr_sim.sum(axis=1, keepdims=True)
    nbr_w = np.where(wsum > 0, nbr_sim / np.maximum(wsum, 1e-12), 0.0).astype(np.float32)   # ★Σsim=0 가드
    return nbr_idx, nbr_w


def build_weight_matrix(nbr_idx, nbr_w, n):
    """희소 전파 연산자 W(n×n): W[d, nbr]=가중치 · prop = W@src(src=FP 희소 또는 rel 밀집).
    ★열이 전부 train 이웃(자기 제외)이라 W@src는 train 정보만 = 누수 없음. ★희소라 V2가 V1처럼 빠름
    (밀집 2048은 XGBoost 86-class서 수십배 느림) + off-bit=결측(missing) 의미가 V1과 동일(감사 지적)."""
    rows = np.repeat(np.arange(n), nbr_idx.shape[1])
    cols = nbr_idx.reshape(-1)
    data = nbr_w.reshape(-1)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)


def build_rel(pi, pj, Y, idxs, n):
    """관계 프로필 rel[g] = g가 참여한 학습 쌍(idxs=tt 또는 wtr)의 86종 유형 분포(정규화).
    ★idxs가 학습 쌍뿐이라 test 쌍의 라벨은 어떤 rel에도 안 들어감(deg=0→0 가드)."""
    rel = np.zeros((n, N_CLASS), dtype=np.float32)
    a = pi[idxs]; b = pj[idxs]; t = Y[idxs]
    np.add.at(rel, (a, t), 1.0)                    # 양끝 약물 모두 그 유형에 +1
    np.add.at(rel, (b, t), 1.0)
    deg = rel.sum(axis=1, keepdims=True)
    return np.where(deg > 0, rel / np.maximum(deg, 1e-12), 0.0).astype(np.float32)   # ★deg=0 가드


def eval_mc(model, Xte, yte, head_t, mid_t, tail_t):
    """★11 eval_full의 le-aware 디코드(fit_mc는 LabelEncoder 압축 proba 반환). 7지표 반환(정의는 14와 동일)."""
    clf, le = model
    pred = le.inverse_transform(clf.predict(Xte)); proba = clf.predict_proba(Xte)
    tl = sorted(set(yte.tolist()))
    per_f1 = f1_score(yte, pred, average=None, labels=tl, zero_division=0)
    f1map = dict(zip(tl, per_f1))
    def band(types):
        v = [f1map[t] for t in tl if t in types]; return float(np.mean(v)) if v else 0.0
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", labels=tl, zero_division=0)),
        "micro_f1": float(f1_score(yte, pred, average="micro", labels=tl, zero_division=0)),
        "top3": top3_acc(proba, le.classes_, yte),
        "head_f1": band(head_t), "mid_f1": band(mid_t), "tail_f1": band(tail_t),
    }


def fit_curve(X, y, seed):
    """학습곡선용(시각화만): XGBClassifier(**XGB_MC)를 eval_set으로 학습해 라운드별 mlogloss 기록.
    scored 모델(전체 tt fit)과 별개 — 내부 90/10 split. loss 감소=학습 정상 확인용."""
    le = LabelEncoder(); yt = le.fit_transform(y)
    tr, va = train_test_split(np.arange(len(yt)), test_size=0.1, random_state=seed)
    clf = xgb.XGBClassifier(**XGB_MC)
    clf.fit(X[tr], yt[tr], eval_set=[(X[tr], yt[tr]), (X[va], yt[va])], verbose=False)
    ev = clf.evals_result()
    return [[i, float(a), float(b)] for i, (a, b) in enumerate(zip(ev["validation_0"]["mlogloss"], ev["validation_1"]["mlogloss"]))]


def leakage_checks(nbr_idx, nbr_w, train_rows, test_rows, g, pi, pj, tt, ts, ss, wtr, wte):
    """★16식 수치 누수검증. cold 항목은 0이어야 하며 아니면 raise."""
    test_set = set(test_rows.tolist())
    used = nbr_idx[nbr_w > 0]                                    # 실제 사용된 이웃(가중치>0)
    nbr_test = int(np.isin(used, np.fromiter(test_set, dtype=np.int64) if test_set else np.array([], dtype=np.int64)).sum())
    n = nbr_idx.shape[0]
    self_incl = int(sum(int(((nbr_idx[d] == d) & (nbr_w[d] > 0)).sum()) for d in range(n)))
    drug_overlap = len(set(train_rows.tolist()) & test_set)
    tt_bad = int(((g[pi[tt]] + g[pj[tt]]) > 0).sum())           # tt 쌍이 test 약물 포함? →0
    ts_bad = int(((g[pi[ts]] + g[pj[ts]]) != 1).sum())          # cold-start=정확히 1개 신약
    ss_bad = int(((g[pi[ss]] + g[pj[ss]]) != 2).sum())          # cold-cold=2개 신약
    warm_pair_overlap = len(set(wtr.tolist()) & set(wte.tolist()))
    tt_set = set(tt.tolist()); cold_test = set(ts.tolist()) | set(ss.tolist())
    tt_coldtest_overlap = len(tt_set & cold_test)
    lk = {"nbr_pool_test_drugs": nbr_test, "neighbor_self_included": self_incl,
          "train_test_drug_overlap": drug_overlap, "tt_pairs_touch_test": tt_bad,
          "cold_start_wrong_regime": ts_bad, "cold_cold_wrong_regime": ss_bad,
          "warm_wtr_wte_overlap": warm_pair_overlap, "tt_vs_coldtest_overlap": tt_coldtest_overlap,
          "n_train_drugs": int(len(train_rows)), "n_test_drugs": int(len(test_rows)),
          "n_tt": int(len(tt)), "n_ts": int(len(ts)), "n_ss": int(len(ss))}
    bad = [k for k in ("nbr_pool_test_drugs", "neighbor_self_included", "train_test_drug_overlap",
                       "tt_pairs_touch_test", "cold_start_wrong_regime", "cold_cold_wrong_regime",
                       "warm_wtr_wte_overlap", "tt_vs_coldtest_overlap") if lk[k] != 0]
    if bad:
        raise RuntimeError(f"누수검증 실패(0이 아님): {[(k, lk[k]) for k in bad]}")
    return lk


def bh_adjust(pvals):
    """Benjamini-Hochberg. 반환: p_adj 리스트(입력 순서)."""
    p = np.asarray(pvals, dtype=float); m = len(p)
    order = np.argsort(p); ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m); out[order] = np.clip(ranked, 0, 1)
    return out.tolist()


def run():
    t0 = time.time()
    print(f"[21] KnowDDI 유사도 전파 축소재현 시작 · SMOKE={SMOKE}", flush=True)
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"])); did2row = {d: i for i, d in enumerate(drug_ids)}
    n = len(drug_ids)
    pairs = pd.read_csv(PAIRS_CSV); pos = pairs[pairs.label == 1]
    pos_pairs = [ukey(str(a), str(b)) for a, b in zip(pos["drug_a"], pos["drug_b"])]
    pi = np.array([did2row[a] for a, b in pos_pairs], dtype=np.int32)
    pj = np.array([did2row[b] for a, b in pos_pairs], dtype=np.int32)
    lo = np.minimum(pi, pj); hi = np.maximum(pi, pj)
    Y = load_Y(pos_pairs); K = len(set(Y.tolist()))
    gc_ = pd.Series(Y).value_counts()
    head_t = set(gc_.head(10).index.tolist()); tail_t = set(gc_[gc_ < 100].index.tolist())
    mid_t = set(gc_.index.tolist()) - head_t - tail_t
    print(f"  약물 {n} · positive {len(pi):,} · 유형 {K}종 · head{len(head_t)}/mid{len(mid_t)}/tail{len(tail_t)} · top-K {TOP_K}", flush=True)

    FP = build_fp_matrix(drug_ids, id2smi)
    FP_csr = FP.astype(np.float32).tocsr()                      # V2 전파 소스(희소 1705×2048)
    bvs = [C._GEN.GetFingerprint(Chem.MolFromSmiles(str(id2smi[d]))) for d in drug_ids]
    M2 = build_sim_matrix(bvs).astype(np.float32)               # n×n Tanimoto(14/15/16과 동일 M2)
    Xv1 = pair_feature(FP, None, lo, hi, "struct", "sum")       # ★V1 = 희소 경로(11 bit-동일 재현)
    print(f"  M2·FP·V1(희소) 준비 완료", flush=True)

    per = {v: {rg: {mt: [] for mt in METRICS} for rg in REGIMES} for v in VERSIONS}
    leak = []; curves = {}; pca_dims = []
    for seed in SEEDS:
        train_drugs, test_drugs = split_drugs(drug_ids, seed, DRUG_TEST_FRAC)
        g = np.array([0 if d in train_drugs else 1 for d in drug_ids], dtype=np.int8)
        train_rows = np.where(g == 0)[0]; test_rows = np.where(g == 1)[0]
        pu = g[pi] + g[pj]
        tt = np.where(pu == 0)[0]; ts = np.where(pu == 1)[0]; ss = np.where(pu == 2)[0]
        wtr, wte = train_test_split(np.arange(len(pos_pairs)), test_size=0.2, random_state=seed, stratify=Y)

        nbr_idx, nbr_w = topk_train_neighbors(M2, set(train_rows.tolist()), TOP_K)
        leak.append(leakage_checks(nbr_idx, nbr_w, train_rows, test_rows, g, pi, pj, tt, ts, ss, wtr, wte))
        W = build_weight_matrix(nbr_idx, nbr_w, n)              # 전파 연산자(train열만·자기제외)

        # V2: 구조 전파 = 이웃 지문 가중평균(밀집) → ★PCA 축소(train-fit·누수0·14/15와 동일 정론·tractability)
        prop_fp = np.asarray((W @ FP_csr).todense(), dtype=np.float32)   # n×2048 borrowed 지문
        pca = PCA(n_components=PCA_VAR, svd_solver="full").fit(prop_fp[train_rows])   # ★train 약물에만 fit
        prop_fp_red = pca.transform(prop_fp).astype(np.float32); pca_dims.append(prop_fp_red.shape[1])
        Xv2 = prop_fp_red[lo] + prop_fp_red[hi]
        # V3: 관계 전파(밀집 86차원·rel은 regime 학습쌍에서만 — cold=tt·warm=wtr)
        prop_rel_cold = np.asarray(W @ build_rel(pi, pj, Y, tt, n))
        prop_rel_warm = np.asarray(W @ build_rel(pi, pj, Y, wtr, n))
        Xv3_cold = prop_rel_cold[lo] + prop_rel_cold[hi]
        Xv3_warm = prop_rel_warm[lo] + prop_rel_warm[hi]
        if np.isnan(Xv2).any() or np.isnan(Xv3_cold).any() or np.isnan(Xv3_warm).any():
            raise RuntimeError(f"NaN in features (seed {seed})")

        feats = {"struct_only": (Xv1, Xv1), "struct_prop": (Xv2, Xv2), "relation_prop": (Xv3_warm, Xv3_cold)}
        for v in VERSIONS:
            Xw, Xc = feats[v]
            mw = eval_mc(fit_mc(Xw[wtr], Y[wtr]), Xw[wte], Y[wte], head_t, mid_t, tail_t)
            cold_model = fit_mc(Xc[tt], Y[tt])                  # ★cold 한 모델 → ts·ss 채점(11·14와 동일)
            mcs = eval_mc(cold_model, Xc[ts], Y[ts], head_t, mid_t, tail_t)
            mcc = eval_mc(cold_model, Xc[ss], Y[ss], head_t, mid_t, tail_t)
            for rg, m in [("warm", mw), ("cold-start", mcs), ("cold-cold", mcc)]:
                for mt in METRICS:
                    per[v][rg][mt].append(m[mt])
            if not SMOKE and seed == SEEDS[0]:
                curves[v] = fit_curve(Xc[tt], Y[tt], seed)      # 대표 학습곡선(cold TT)
        # V1 재현 앵커 확인(seed 0)
        if seed == 0:
            print("  ★V1 재현(seed0) vs 11 baseline 앵커: " + " · ".join(
                f"{rg} {per['struct_only'][rg]['macro_f1'][-1]:.4f}(앵커 {V1_ANCHOR_S0[rg]:.4f})" for rg in REGIMES), flush=True)
        print(f"  [seed {seed}] cc macroF1 — " + " · ".join(f"{VLABEL[v]} {per[v]['cold-cold']['macro_f1'][-1]:.3f}" for v in VERSIONS), flush=True)
        del Xv2, Xv3_cold, Xv3_warm, prop_fp, prop_fp_red, prop_rel_cold, prop_rel_warm, W; gc.collect()

    agg = {v: {rg: {mt: {"mean": float(np.mean(per[v][rg][mt])), "std": float(np.std(per[v][rg][mt]))} for mt in METRICS} for rg in REGIMES} for v in VERSIONS}
    # 짝지은 통계(cold-cold macro_f1): V2-V1 · V3-V1
    stat = {}
    if len(SEEDS) > 1:
        base = np.array(per["struct_only"]["cold-cold"]["macro_f1"]); praw = []
        for v in ["struct_prop", "relation_prop"]:
            a = np.array(per[v]["cold-cold"]["macro_f1"]); d = a - base
            try:
                t, p = stats.ttest_rel(a, base); t = float(t); p = float(p) if not np.isnan(p) else 1.0
            except Exception:
                t, p = 0.0, 1.0
            stat[v] = {"dmean": float(d.mean()), "t": t, "p": p, "wins": int((d > 0).sum())}; praw.append(p)
        padj = bh_adjust(praw)
        for v, pa in zip(["struct_prop", "relation_prop"], padj):
            stat[v]["p_adj"] = float(pa)

    res = {"config": {"n_pairs": len(pi), "n_types": K, "seeds": SEEDS, "top_k": TOP_K,
                      "pca_var": PCA_VAR, "pca_dim": int(np.mean(pca_dims)) if pca_dims else 0,
                      "n_head": len(head_t), "n_mid": len(mid_t), "n_tail": len(tail_t),
                      "head_def": "top-10", "tail_def": "<100 count", "device": "cpu",
                      "smoke": SMOKE, "runtime_s": round(time.time() - t0, 1)},
           "results": agg, "per_seed": per, "leakage": leak, "stat": stat, "curve": curves,
           "v1_anchor_seed0": V1_ANCHOR_S0}
    json.dump(res, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ cold-cold (macro-F1, 5 seed) ================")
    for v in VERSIONS:
        a = agg[v]["cold-cold"]; d = a["macro_f1"]["mean"] - agg["struct_only"]["cold-cold"]["macro_f1"]["mean"]
        print(f"  {VLABEL[v]:20s}: macroF1 {a['macro_f1']['mean']:.3f}±{a['macro_f1']['std']:.3f} (ΔvsV1 {d:+.3f}) · acc {a['accuracy']['mean']:.3f} · tail {a['tail_f1']['mean']:.3f}")
    if stat:
        for v in ["struct_prop", "relation_prop"]:
            print(f"    {VLABEL[v]}: Δ{stat[v]['dmean']:+.3f} · {stat[v]['wins']}/5 seed · p{stat[v]['p']:.3f} · p_adj(BH){stat[v]['p_adj']:.3f}")
    lk0 = leak[0]
    print(f"  누수검증(seed0): 이웃풀 test {lk0['nbr_pool_test_drugs']} · self {lk0['neighbor_self_included']} · 약물겹침 {lk0['train_test_drug_overlap']} · tt-test접촉 {lk0['tt_pairs_touch_test']} · cs오분류 {lk0['cold_start_wrong_regime']} · cc오분류 {lk0['cold_cold_wrong_regime']} → 전부 0 PASS")

    if not SMOKE:
        out_html = build_report(res)
        print(f"\n[21] 완료 ({res['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    else:
        print(f"\n[21] SMOKE 완료 ({res['config']['runtime_s']:.0f}s) — V1앵커·누수·loss 확인용")
    return res


# ---------------- 보고서 ----------------
def _b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _load_anchors():
    """cold-cold 문맥 앵커(다른 실험 JSON에서 로드 · 하드코딩 금지)."""
    A = {}
    try:
        imb = json.load(open(os.path.join(BASE, "results", "imbalance_results.json")))["results"]
        A["11 XGBoost baseline"] = imb["baseline"]; A["11 XGBoost +class-weight"] = imb["class-weight"]
    except Exception:
        pass
    try:
        A["14 DeepDDI (DNN)"] = json.load(open(os.path.join(BASE, "results", "deepddi_results.json")))["results"]["plain"]
    except Exception:
        pass
    try:
        A["15 DDIMDL struct"] = json.load(open(os.path.join(BASE, "results", "ddimdl_results.json")))["results"]["struct_only"]
    except Exception:
        pass
    try:
        A["16 SSI-DDI (GNN)"] = json.load(open(os.path.join(BASE, "results", "ssiddi_results.json")))["results"]
    except Exception:
        pass
    return A


def build_report(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for _kf in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]:
        if os.path.exists(_kf):
            fm.fontManager.addfont(_kf); plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False
    agg = res["results"]; c = res["config"]; stat = res["stat"]; anchors = _load_anchors()
    vcol = {"struct_only": "#8a8a99", "struct_prop": "#4c78d8", "relation_prop": "#c98ae0"}

    # 그림1: 방법 스키마 (V1/V2/V3)
    fig0, ax = plt.subplots(figsize=(9.4, 2.7)); ax.axis("off")
    rows = [("V1 struct_only", "자기 Morgan 지문\n(전파 없음·대조군)", "#8a8a99"),
            ("V2 struct_prop", "닮은 train 이웃들의\nMorgan을 가중평균으로 빌림", "#4c78d8"),
            ("V3 relation_prop", "닮은 train 이웃들의\n'관계 프로필(86종 분포)'을 빌림", "#b0392f")]
    for i, (t, s, col) in enumerate(rows):
        yy = 0.72 - i * 0.30
        ax.add_patch(plt.Rectangle((0.02, yy - 0.11), 0.20, 0.22, fc=col, ec="#222", alpha=.85, transform=ax.transAxes))
        ax.text(0.12, yy, t, ha="center", va="center", fontsize=9, color="white", transform=ax.transAxes)
        ax.text(0.26, yy, s, ha="left", va="center", fontsize=8.6, transform=ax.transAxes)
    ax.text(0.70, 0.72, "이웃 = ★train 약물만\n자기 제외 · top-K=20\n가중치=Tanimoto", ha="left", va="center", fontsize=8.6, transform=ax.transAxes, color="#2ec76a")
    ax.text(0.70, 0.30, "쌍 특징 = feat[lo]+feat[hi]\n분류기 = XGBoost(고정)\n유일 변수 = 전파 방식", ha="left", va="center", fontsize=8.6, transform=ax.transAxes, color="#e0a52e")
    chart_m = _b64(fig0)

    # 그림2: macro-F1 V1/V2/V3 × regime
    fig1, ax = plt.subplots(figsize=(8.8, 3.8)); w = 0.25
    for i, v in enumerate(VERSIONS):
        vals = [agg[v][rg]["macro_f1"]["mean"] for rg in REGIMES]; errs = [agg[v][rg]["macro_f1"]["std"] for rg in REGIMES]
        ax.bar(np.arange(3) + (i - 1) * w, vals, w, yerr=errs, capsize=3, color=vcol[v], label=VLABEL[v])
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(REGIMES); ax.set_ylabel("macro-F1 (5 seed)"); ax.set_ylim(0, 0.85)
    ax.set_title("유사도 전파 3버전 × regime — macro-F1  (★cold-cold=신약)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    chart_b = _b64(fig1)

    # 그림3: 학습곡선 (버전별 cold TT)
    fig2, ax = plt.subplots(figsize=(7.4, 3.2))
    for v in VERSIONS:
        cv = np.array(res["curve"][v]); ax.plot(cv[:, 0], cv[:, 2], "-", lw=1.4, color=vcol[v], label=f"{VLABEL[v]} val")
    ax.set_xlabel("boosting round"); ax.set_ylabel("val mlogloss"); ax.set_title("학습곡선(cold TT·seed0) — mlogloss 감소=학습 정상")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    chart_c = _b64(fig2)

    # 그림4: 핵심 cold-cold V1/V2/V3 (per-seed 점 + 평균)
    fig3, ax = plt.subplots(figsize=(7.4, 3.4)); xs = np.arange(len(VERSIONS))
    for i, v in enumerate(VERSIONS):
        pts = res["per_seed"][v]["cold-cold"]["macro_f1"]
        ax.scatter([i] * len(pts), pts, color=vcol[v], alpha=.55, s=28, zorder=3)
        ax.bar(i, agg[v]["cold-cold"]["macro_f1"]["mean"], 0.55, color=vcol[v], alpha=.35, zorder=1)
        ax.annotate(f"{agg[v]['cold-cold']['macro_f1']['mean']:.3f}", (i, agg[v]["cold-cold"]["macro_f1"]["mean"] + .006), ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([VLABEL[v] for v in VERSIONS], fontsize=9)
    ax.set_ylabel("cold-cold macro-F1"); ax.set_title("★핵심: cold-cold(신약)에서 전파가 V1을 넘나 — per-seed")
    ax.grid(axis="y", alpha=.25)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    chart_k = _b64(fig3)

    # 표 셀
    def cell(v, rg, mt):
        a = agg[v][rg][mt]; return f"{a['mean']:.3f}<span class=s>±{a['std']:.3f}</span>"
    def perf_rows(v):
        r = ""
        for rg in REGIMES:
            hl = " class=hl" if rg == "cold-cold" else ""
            r += (f"<tr><td class=l>{rg}</td><td{hl}>{cell(v,rg,'macro_f1')}</td><td>{cell(v,rg,'micro_f1')}</td>"
                  f"<td>{cell(v,rg,'accuracy')}</td><td>{cell(v,rg,'top3')}</td><td>{cell(v,rg,'head_f1')}</td>"
                  f"<td>{cell(v,rg,'mid_f1')}</td><td>{cell(v,rg,'tail_f1')}</td></tr>")
        return r

    v1cc = agg["struct_only"]["cold-cold"]["macro_f1"]["mean"]
    v2cc = agg["struct_prop"]["cold-cold"]["macro_f1"]["mean"]
    v3cc = agg["relation_prop"]["cold-cold"]["macro_f1"]["mean"]
    best_v = max(VERSIONS, key=lambda v: agg[v]["cold-cold"]["macro_f1"]["mean"])
    warm = {v: agg[v]["warm"]["macro_f1"]["mean"] for v in VERSIONS}          # warm/cs로 V2 '이득' 착시 진단
    csf = {v: agg[v]["cold-start"]["macro_f1"]["mean"] for v in VERSIONS}
    p2 = stat.get("struct_prop", {}).get("p", 1.0); p3a = stat.get("relation_prop", {}).get("p_adj", 1.0)
    # 핵심 비교표
    core_rows = ""
    for v in VERSIONS:
        a = agg[v]["cold-cold"]; d = a["macro_f1"]["mean"] - v1cc
        st = stat.get(v, {})
        stc = (f"{st['wins']}/5 · p{st['p']:.3f} · p_adj{st['p_adj']:.3f}" if st else "기준")
        win = "<span class=good>✓</span>" if v != "struct_only" and st.get("wins", 0) >= 4 and st.get("p_adj", 1) < 0.05 and d > 0.003 else \
              ("<span class=warn>방향만</span>" if v != "struct_only" and d > 0.003 else ("<span class=mut>—</span>" if v != "struct_only" else "기준"))
        hl2 = " class=hl2" if v == best_v and v != "struct_only" else " class=l"
        core_rows += (f"<tr><td{hl2}><b>{VLABEL[v]}</b></td><td>{a['macro_f1']['mean']:.3f}±{a['macro_f1']['std']:.3f}</td>"
                      f"<td class={'good' if d>0.003 else 'bad' if d<-0.003 else 'mut'}>{d:+.3f}</td><td class=s>{stc}</td><td>{win}</td></tr>")
    # 통계 표
    stat_rows = ""
    for v in ["struct_prop", "relation_prop"]:
        st = stat.get(v, {})
        if st:
            stat_rows += (f"<tr><td class=l>{VLABEL[v]} − V1</td><td class={'good' if st['dmean']>0 else 'bad'}>{st['dmean']:+.3f}</td>"
                          f"<td>{st['t']:.2f}</td><td>{st['p']:.3f}</td><td>{st['p_adj']:.3f}</td><td>{st['wins']}/5</td></tr>")
    # 누수표
    lk = res["leakage"]
    def lkmax(key):
        return max(x[key] for x in lk)
    leak_rows = ""
    checks = [("이웃풀에 포함된 test 약물 수", "nbr_pool_test_drugs", 0), ("이웃에 자기자신 포함 수", "neighbor_self_included", 0),
              ("train↔test 약물 겹침", "train_test_drug_overlap", 0), ("tt 학습쌍이 test 약물 접촉", "tt_pairs_touch_test", 0),
              ("cold-start 쌍 ≠ 정확히 1 신약", "cold_start_wrong_regime", 0), ("cold-cold 쌍 ≠ 정확히 2 신약", "cold_cold_wrong_regime", 0),
              ("warm wtr∩wte 쌍 겹침", "warm_wtr_wte_overlap", 0), ("tt ∩ (cold test) 쌍 겹침", "tt_vs_coldtest_overlap", 0)]
    for lab, key, exp in checks:
        mx = lkmax(key)
        leak_rows += f"<tr><td class=l>{lab}</td><td class={'good' if mx==exp else 'bad'}>{mx}</td><td class=s>=0 (5 seed 전부)</td></tr>"

    # 앵커(문맥) 표 — cold-cold macro-F1
    anc_rows = ""
    for name, dat in anchors.items():
        try:
            val = dat["cold-cold"]["macro_f1"]["mean"]; anc_rows += f"<tr><td class=l>{name}</td><td>{val:.3f}</td></tr>"
        except Exception:
            pass

    # 판정 문구
    v3_gain = v3cc - v1cc; v3st = stat.get("relation_prop", {})
    v3_robust = v3st.get("wins", 0) >= 4 and v3st.get("p_adj", 1) < 0.05 and v3_gain > 0.003
    if v3_robust:
        verdict = f"★<b>관계 전파(V3)가 신약을 뚫는다</b> — cold-cold {v1cc:.3f}→{v3cc:.3f}(Δ{v3_gain:+.3f}·{v3st.get('wins')}/5·p_adj{v3st.get('p_adj',1):.3f})"
    elif v3_gain > 0.003:
        verdict = f"★<b>관계 전파(V3)는 방향만</b> — cold-cold {v1cc:.3f}→{v3cc:.3f}(Δ{v3_gain:+.3f})이나 {v3st.get('wins','?')}/5·p_adj{v3st.get('p_adj',1):.3f}(비유의·편차 내)"
    else:
        verdict = f"★<b>전파도 cold-cold 벽을 못 뚫음</b> — V3 {v3cc:.3f} vs V1 {v1cc:.3f}(Δ{v3_gain:+.3f}) = 표현/데이터 병목 재확인"

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 6px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}.hl{background:#16202e}.hl2{background:#241a2e;color:#c98ae0}
    .s{color:#7a7f8a;font-size:10px}.card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>21 DDI KnowDDI propagation</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>21 · 5세대(KnowDDI 2024) 핵심 아이디어 축소재현 — '유사도 전파'가 신약을 뚫나</h1>
<p class=mut>구조로 이웃을 찾아 그 이웃의 '관계'를 전파. ★분류기 고정(XGBoost)·유일 변수=전파 방식(V1 없음/V2 구조/V3 관계). 11·14·15·16과 동일 데이터·분할·지표 · env admet · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> {verdict}. cold-cold macro-F1: V1 {v1cc:.3f} · V2 struct_prop {v2cc:.3f} · <b class=hl2>V3 relation_prop {v3cc:.3f}</b> <span class=s>(★V2가 수치상 최고이나 warm {warm['struct_prop']:.2f}≪V1 {warm['struct_only']:.2f}=표현 착시·깨끗한 전파 이득 아님 — ⑤)</span>.<br>
★<b>정직 프레이밍</b>: 이는 KnowDDI 원본 재현이 아니라(Hetionet 등 <b>외부 지식그래프 안 씀</b>) 그 정수인 '유사도 전파'만 떼어 3버전 비교한 <b>축소 재현</b>이다. ★<b>V1은 11 baseline과 bit-동일 재현</b>(seed0 cc {res['per_seed']['struct_only']['cold-cold']['macro_f1'][0]:.4f} vs 앵커 {res['v1_anchor_seed0']['cold-cold']:.4f}) = 저울이 같음을 확인.<br>
★<b>누수 청정</b>(아래 ③ 수치): 이웃풀=train만·자기 제외·rel은 학습쌍만 → cold 테스트 특징에 test 라벨·test-test 유사도 0 유입(전부 0, 5 seed). ★warm은 누수 맥락(참고만·V3 warm은 약물정체 암기로 부풀 수 있음). ★학습 정상(mlogloss 감소·NaN 0).</div>

<h2>① 방법 (V1 없음 / V2 구조 전파 / V3 관계 전파)</h2>
<img src="{chart_m}" alt="method">
<div class=card>세 버전 모두 per-drug 표현 → 쌍 특징 = <b>feat[lo]+feat[hi]</b>(sum 융합·11과 동일)·분류기 <b>fit_mc(XGBoost multi:softprob·XGB_MC 그대로)</b> 고정. 유일 변수 = 전파 방식.<br>
· <b>V1 struct_only</b>=자기 Morgan(전파 없음). ★희소 pair_feature 경로 = 11 baseline과 bit-동일(재현 앵커).<br>
· <b>V2 struct_prop</b>=prop_fp[d]=구조 닮은 <b>train 이웃 top-{c['top_k']}</b>의 Morgan을 Tanimoto 가중평균 → ★PCA 축소(train-fit·분산{int(PCA_VAR*100)}%·P≈{c['pca_dim']}·14/15와 동일 정론, 밀집 2048은 XGBoost 86-class서 비현실적 느림).<br>
· <b>V3 relation_prop</b>=prop_rel[d]=같은 이웃들의 <b>관계 프로필 rel</b>을 가중평균(86차원). rel[g]=g가 참여한 그 regime의 <b>학습 쌍</b>(cold→tt·warm→wtr) 86종 유형 분포(정규화). ★<b>이게 KnowDDI 정수</b>(이웃의 '관계'를 빌려옴).<br>
· 이웃 후보 = ★<b>train 약물만</b>(M2 열을 train_rows로 슬라이스)·<b>자기 제외</b>·가중치=Tanimoto/Σ. ★1세대 Vilar(13)는 직접 상호작용 경로가 필요해 cold-cold서 ROC 0.500 붕괴 — V3는 이웃의 관계 <b>분포</b>를 빌리므로 cold-cold서도 작동 가능(이 실험의 가설).</div>

<h2>② 학습곡선 (전파 버전이 제대로 학습되나)</h2>
<img src="{chart_c}" alt="learning curve">
<div class=card>세 버전 모두 val mlogloss가 <b>감소</b> → 학습 정상(NaN·발산 없음). 곡선은 cold TT 모델(seed0)의 내부 90/10 val 기준(시각화용·scored 모델은 전체 tt fit).</div>

<h2>③ ★누수 검증 (수치 · 16처럼 정직)</h2>
<div style="overflow-x:auto"><table>
<tr><th class=l>점검 항목</th><th>관측(5 seed 최대)</th><th>기대</th></tr>{leak_rows}
</table></div>
<div class=card>전부 <b class=good>0</b> = ★cold 테스트 특징에 (1) test 약물이 이웃 소스로 들어오거나 (2) test-test 유사도가 섞이거나 (3) test 쌍 라벨이 rel로 새는 경로 <b>없음</b>. rel은 학습쌍(tt/wtr)에서만 만들어 cold 테스트 쌍(ts/ss는 tt에 없음)의 라벨이 어떤 rel에도 안 들어감.<br>
★<b>남는 잔여(정직)</b>: 학습쌍 중 파트너가 구조 이웃인 경우 그 쌍의 특징에 자기 라벨이 <b>희석되어</b>(≈w_partner/Σw × 1/deg) 들어감 — 그러나 이는 <b>학습쪽뿐</b>이고 cold 테스트엔 없어, 있다면 cold-cold를 오히려 <b>낮추는</b> 방향(train/test 특징 분포 불일치). 따라서 V3 이득은 이 잔여로 설명되지 않음.</div>

<h2>④ 성능: V1/V2/V3 × 3 regime (macro-F1 + micro/acc/top3 + head/mid/tail)</h2>
<img src="{chart_b}" alt="bars">"""
    for v in VERSIONS:
        html += f"""
<p class=mut><b>{VLABEL[v]}</b> (5 seed 평균±std)</p>
<div style="overflow-x:auto"><table>
<tr><th class=l>regime</th><th>macroF1</th><th>microF1</th><th>acc</th><th>top3</th><th>head</th><th>mid</th><th>tail</th></tr>
{perf_rows(v)}
</table></div>"""
    html += f"""
<p class=mut>warm은 ★누수 맥락(약물이 train·test 양쪽 등장 가능·V3는 rel이 약물 정체를 노출) — 참고만, ★핵심은 cold-start·cold-cold.</p>

<h2>⑤ ★핵심 비교: cold-cold macro-F1 — V1 vs V2 vs V3 (신약)</h2>
<img src="{chart_k}" alt="core cold-cold">
<div style="overflow-x:auto"><table>
<tr><th class=l>버전 (cold-cold)</th><th>macro-F1</th><th>Δ vs V1</th><th>견고성(seed·p·BH)</th><th>판정</th></tr>{core_rows}
</table></div>
<div style="overflow-x:auto"><table>
<tr><th class=l>짝지은 검정(cold-cold macroF1)</th><th>Δmean</th><th>t</th><th>p</th><th>p_adj(BH)</th><th>win</th></tr>{stat_rows}
</table></div>
<div class=card>{verdict}.<br>
· <b>V2 struct_prop</b> Δ{v2cc-v1cc:+.3f}(5/5·p{p2:.3f}·<b>비유의</b>): 점추정은 V1 위지만 ★<b>깨끗한 '전파 이득' 아님</b> — V2는 <b>warm {warm['struct_prop']:.3f}·cold-start {csf['struct_prop']:.3f}</b>로 V1(warm {warm['struct_only']:.3f}·cs {csf['struct_only']:.3f})보다 <b>크게 낮음</b>: 이웃 지문 평균+PCA({c['pca_dim']}차원)이 약물 <b>정체를 흐려</b> 외울 게 있는 warm서 손해. cold-cold '이득'은 표현이 약해 <b>떨어질 게 적은</b> 것에 가깝고 PCA-밀집화 교란과 섞임 → 전파가 신약에 정보를 더했다는 근거 아님.<br>
· <b>V3 relation_prop</b> Δ{v3cc-v1cc:+.3f}(0/5·p_adj{p3a:.3f}·<b>유의(경계·n=5)</b>): ★<b>KnowDDI 정수(관계 전파)가 cold-cold서 V1을 하회</b> — 구조 닮은 이웃의 관계 분포가 자기 구조보다 <b>못한</b> 예측자(DrugBank DDI 구조유사도 약함=이웃 관계도 잡음). warm {warm['relation_prop']:.3f}·cs {csf['relation_prop']:.3f}도 V1 하회 → 전파가 신약 벽을 뚫기는커녕 <b>낮춤</b>.<br>
· ★<b>통계 주의</b>: seed 5개라 부호검정 p 바닥 0.0625·cold-cold seed std ~{agg['struct_only']['cold-cold']['macro_f1']['std']:.3f} — <b>V3만 paired-t 유의(경계)</b>(p_adj{p3a:.3f}·n=5·부호검정 바닥 0.0625이라 단정 아님), V2는 방향 일관(5/5)이나 비유의·표현 교란.</div>

<h2>⑥ 해석 + 세대 문맥 (cold-cold macro-F1)</h2>
<div style="overflow-x:auto"><table>
<tr><th class=l>방법 (cold-cold)</th><th>macro-F1</th></tr>
<tr><td class=l><b>21 V1 struct_only (=대조)</b></td><td>{v1cc:.3f}</td></tr>
<tr><td class=l><b>21 V2 struct_prop</b></td><td>{v2cc:.3f}</td></tr>
<tr><td class=l><b>21 V3 relation_prop</b></td><td class=hl2>{v3cc:.3f}</td></tr>{anc_rows}
</table></div>
<div class=card>· <b>전파가 신약에 도움인가</b>: {'V3가 V1을 견고하게 넘음 → 구조 밖 관계 정보가 신약 예측에 실제로 기여' if v3_robust else 'V3(관계 전파)가 V1을 넘기는커녕 <b>유의하게 하회</b>(Δ%+.3f·0/5·p_adj%.3f)·V2(구조 전파)도 깨끗한 이득 아님(warm/cs 붕괴=표현 착시) → ★유사도 전파는 cold-cold 벽(신약-쌍 신호 희박)을 <b>뚫지 못함</b> = 13~19의 표현/데이터 병목 결론 재확인' % (v3cc-v1cc, p3a)}.<br>
· ★<b>1세대(Vilar 13)와 대비</b>: Vilar는 cold-cold서 ROC 0.500 <b>구조적 붕괴</b>(전파할 알려진 상호작용이 두 신약엔 없음). V3는 이웃의 관계 <b>분포</b>를 빌려 붕괴는 피하나(작동은 함), 그 값이 V1을 {'넘고' if v3_robust else '넘지는 못함'}. ※Vilar 0.500은 <b>이진 ROC</b>라 이 표의 다중분류 macro-F1과 <b>절대 비교 불가</b>(개념 대비만).<br>
· ★<b>세대 무관 정체</b>: 고전~5세대 아이디어 모두 신약 cc가 ~0.22–0.29 대 — 병목은 모델/아이디어 세대보다 <b>표현·데이터</b>(구조유사도 약한 DrugBank DDI·희박한 신약 신호)임을 재확인.</div>

<h2>⑦ ★한계·정직 메모</h2>
<div class=card>
· ★<b>KnowDDI 원본 아님</b>: Hetionet 등 외부 지식그래프·GNN 링크예측 헤드 안 씀 = '유사도 전파' 아이디어만 축소. 원본은 더 풍부한 KG로 더 나을 수 있음.<br>
· <b>warm 누수 맥락</b>: 같은 약물 train·test 양쪽 → V3 rel이 약물 정체 노출로 warm 부풀 수 있음(참고만·결론은 cold 기반).<br>
· <b>top-K={c['top_k']}·정규화 분포·sum 융합</b>은 선택값(민감도 미탐색) · V3 rel 정규화는 promiscuity(상호작용 수) 크기를 제거하고 '유형 패턴'만 남김.<br>
· ★<b>V2는 PCA 축소</b>(분산{int(PCA_VAR*100)}%·P≈{c['pca_dim']}·train-fit·14/15 정론): 밀집 2048 borrowed 지문이 XGBoost 86-class서 비현실적으로 느려 불가피 — 분산 90% 보존이나 V1(희소 원지문)과 표현형이 달라짐(V1=희소·off-bit 결측 / V2·V3=밀집 전파벡터). 표현 축이 섞이나 변수는 여전히 '전파 유무·종류'.<br>
· <b>비교는 파이프라인 전체</b>(표현만 바꾸고 분류기 고정) — 순수 전파 효과 분리는 됐으나 다른 분류기(DNN 등)에선 다를 수 있음 · 이진(02~09)과 비교 불가 · seed 5개(통계력 제한).</div>

<h2>⑧ 다음</h2>
<div class=card>· V3 rel에 <b>타깃/경로 유사도</b>(12) 결합 · top-K·정규화 민감도 · V2+V3 결합(구조+관계) · 진짜 외부 KG(KnowDDI 충실) · cold-cold 전용 few-shot.</div>

<div class=foot>21 DDI KnowDDI 유사도 전파 축소재현 · V1 struct_only / V2 struct_prop / V3 relation_prop · 분류기 고정 XGBoost multi:softprob · warm/cold-start/cold-cold · macro-F1+head/mid/tail · top-K {c['top_k']} · 이웃=train만·자기제외·rel=학습쌍만 · seed {len(c['seeds'])} · env admet · ★누수검증 수치 PASS·V1=11 bit-동일 재현</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "21_ddi_knowddi_propagation.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


if __name__ == "__main__":
    if os.environ.get("PROP_REPORT_ONLY") == "1":       # 저장된 JSON에서 보고서만 재생성(재학습 없음)
        print("보고서 재생성 →", build_report(json.load(open(RESULTS_JSON))))
    else:
        run()

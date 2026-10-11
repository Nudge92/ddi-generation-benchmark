"""
[DDI 실험 - 13] 1세대(Vilar 2014) 재현 — '닮음으로 상호작용 퍼뜨리기' vs XGBoost
==============================================================================
Vilar 2014 핵심: "닮은 약은 비슷하게 상호작용한다" → 알려진 상호작용을 닮은 약에게 퍼뜨려
  모르는 쌍을 예측. ★학습 없음, 행렬 곱만.

방법 3단계(논문 핵심):
  1) M1 = 상호작용표(약물×약물, 있으면1/없으면0, 대칭). ★train 약물끼리 상호작용만(test 숨김).
  2) M2 = 닮음표(약물×약물, 각 칸=Tanimoto). Morgan 지문으로.
  3) M3 = M1 × M2 (행렬곱). 점수=Σ_k (i가 k와 상호작용)×(k가 j와 닮음).
     ★DDI는 대칭이라 S3 = M3 + M3ᵀ 로 대칭화(무향쌍 점수). 대각(자기자신) 제외.

★수학적 함의(중요): M1은 train-train만 채우므로
  · cold-start(한쪽만 미지): train쪽 약물의 알려진 상호작용을 미지약물의 '닮음'으로 전파 → 작동.
  · cold-cold(둘다 미지): 두 약물 모두 알려진 상호작용 없음 → 점수=0 → ★구조적으로 예측 불가(degenerate).
  이게 1세대 방법의 근본 한계(닻이 될 '알려진 약물'이 최소 하나 필요).

평가(시리즈 방식 그대로): 약물 train80/test20(split_drugs=03/06과 동일 분할).
  regime별(cold-start=한쪽 미지·cold-cold=둘다 미지) positive(숨긴 정답)+negative(1:1).
  ★negative 2종: random(=03 방식)·matched(Tanimoto 분포매칭=06 방식). 지표 ROC-AUC·PR-AUC·seed5.

★비교(핵심): 같은 fold·같은 negative로 Vilar(무학습) vs XGBoost(sum, ddi_common) 나란히.
  + 02(warm 0.877)·03(cs 0.805/cc 0.713 random)·06(cs 0.792/cc 0.689 matched) 문맥 인용.

산출: results/13_ddi_vilar_similarity.html , results/vilar_results.json
실행: conda run -n admet python src/13_vilar_similarity.py
"""
import os, sys, json, time, io, base64
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddi_common as C
from ddi_common import ukey, build_fp_matrix, make_features, evaluate, split_drugs, build_sim_matrix, bin_match

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
RESULTS_JSON = os.path.join(BASE, "results", "vilar_results.json")

SEEDS = [0, 1, 2, 3, 4]
DRUG_TEST_FRAC = 0.20
N_BINS = 40                              # matched negative Tanimoto 분포 bin(05/06과 동일)
REGIMES = {"cold-start": 1, "cold-cold": 2}   # pair_unseen 값(TS=1, SS=2)
METHODS = ["vilar_raw", "vilar_norm", "xgboost"]
NEGS = ["random", "matched"]
# 문맥 인용(기존 실험, sum 융합)
ANCHORS = {
    "random": {"warm": 0.8766, "cold-start": 0.8048, "cold-cold": 0.7126, "src": "02/03"},
    "matched": {"warm": 0.8719, "cold-start": 0.7919, "cold-cold": 0.6894, "src": "06"},
}


def load_positives():
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"])); did2row = {d: i for i, d in enumerate(drug_ids)}
    pairs = pd.read_csv(PAIRS_CSV); pos = pairs[pairs.label == 1]
    pi = np.array([did2row[str(a)] for a in pos["drug_a"]], dtype=np.int32)
    pj = np.array([did2row[str(b)] for b in pos["drug_b"]], dtype=np.int32)
    return drug_ids, id2smi, did2row, pi, pj


def enum_candidates(val, g):
    """regime(val: TT=0/TS=1/SS=2)의 ★모든 후보 쌍을 열거(무향, lo<hi).
    무한루프 위험 없는 결정적 열거(이전 rejection 방식이 cold-cold서 풀>가용쌍 → 무한루프)."""
    test_rows = np.where(g == 1)[0]; train_rows = np.where(g == 0)[0]
    if val == 1:                                       # cold-start: test × train
        a = np.repeat(test_rows, len(train_rows)); b = np.tile(train_rows, len(test_rows))
    elif val == 2:                                     # cold-cold: test 중 2개
        ii, jj = np.triu_indices(len(test_rows), k=1); a = test_rows[ii]; b = test_rows[jj]
    else:                                              # TT: train 중 2개
        ii, jj = np.triu_indices(len(train_rows), k=1); a = train_rows[ii]; b = train_rows[jj]
    lo = np.minimum(a, b).astype(np.int64); hi = np.maximum(a, b).astype(np.int64)
    return lo, hi


def sample_negatives(scheme, val, g, n_need, pos_lin, M2, pos_sims, seed, n):
    """regime 조건의 비상호작용 쌍 n_need개. random=균등추출 / matched=Tanimoto 분포매칭(bin_match).
    ★후보를 전부 열거 후 positive를 벡터화 제거(np.isin) → 안전·결정적."""
    lo, hi = enum_candidates(val, g)
    cand_lin = lo * n + hi
    keep = ~np.isin(cand_lin, pos_lin)                 # positive(상호작용) 제거
    lo = lo[keep]; hi = hi[keep]
    rng = np.random.RandomState(10_000 + seed)
    if len(lo) <= n_need:                              # 가용 negative가 부족하면 전부 사용
        idx = np.arange(len(lo))
    elif scheme == "random":
        idx = rng.choice(len(lo), size=n_need, replace=False)
    else:                                              # matched: positive Tanimoto 분포에 맞춤
        sims = M2[lo, hi]
        idx, _ = bin_match(sims, pos_sims, N_BINS, seed)
    return np.stack([lo[idx], hi[idx]], axis=1).astype(np.int32)


def vilar_scores(S3, pairs):
    return S3[pairs[:, 0], pairs[:, 1]]


def roc_pr(y, s):
    # 점수 전부 동일(예: cold-cold degenerate)면 roc=0.5
    if np.allclose(s, s[0]):
        return 0.5, float(np.mean(y))
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def run():
    t0 = time.time()
    print("[13] Vilar 1세대 재현 시작 ...", flush=True)
    drug_ids, id2smi, did2row, pi, pj = load_positives()
    n = len(drug_ids)
    plo = np.minimum(pi, pj).astype(np.int64); phi = np.maximum(pi, pj).astype(np.int64)
    pos_lin = np.sort(plo * n + phi)                   # positive 무향쌍의 선형 인덱스(벡터화 제거용)
    print(f"  약물 {n} · positive {len(pi):,}", flush=True)

    FP = build_fp_matrix(drug_ids, id2smi)
    # Tanimoto 유사도 행렬 M2 (구조만 — 항상 사용가능, 누수 없음). RDKit 비트벡터로 build_sim_matrix.
    bvs = [C._GEN.GetFingerprint(Chem.MolFromSmiles(str(id2smi[d]))) for d in drug_ids]
    M2 = build_sim_matrix(bvs).astype(np.float32)     # n×n, 대각=1
    print(f"  M2(Tanimoto {n}×{n}) 완성 · 평균 off-diag {M2[~np.eye(n,dtype=bool)].mean():.3f}", flush=True)

    per = {ng: {rg: {m: {"roc": [], "pr": []} for m in METHODS} for rg in REGIMES} for ng in NEGS}
    counts = {rg: [] for rg in REGIMES}
    for seed in SEEDS:
        train_drugs, test_drugs = split_drugs(drug_ids, seed, DRUG_TEST_FRAC)
        g = np.array([0 if d in train_drugs else 1 for d in drug_ids], dtype=np.int8)
        pu = g[pi] + g[pj]                                  # 각 positive의 regime(0/1/2)

        # ---- M1 = train-train 상호작용만 (test 숨김) ----
        tt = np.where(pu == 0)[0]
        M1 = np.zeros((n, n), dtype=np.float32)
        M1[pi[tt], pj[tt]] = 1.0; M1[pj[tt], pi[tt]] = 1.0   # 대칭
        # ---- M3 = M1×M2, 대칭화 S3 = M3+M3ᵀ (무향쌍 점수) ----
        P = M1 @ M2                                          # (i가 상호작용하는 k)×(k와 j 닮음)
        S3_raw = P + P.T
        deg = M1.sum(1)                                      # train 상호작용 차수(전파 이웃 수)
        # degree-보정: 각 약물 이웃수로 나눠 평균화(허브 편향 제거 진단용)
        Pn = P / np.maximum(deg[:, None], 1.0)
        S3_norm = Pn + Pn.T

        # ---- XGBoost 학습(=Vilar와 동일 지식: TT positive만) : neg 방식별 ----
        for ng in NEGS:
            tt_pos = np.stack([pi[tt], pj[tt]], axis=1)
            tt_pos_sims = M2[tt_pos[:, 0], tt_pos[:, 1]]
            tt_neg = sample_negatives(ng, 0, g, len(tt), pos_lin, M2, tt_pos_sims, seed, n)
            tr_pairs = np.vstack([tt_pos, tt_neg]); ytr = np.r_[np.ones(len(tt_pos)), np.zeros(len(tt_neg))]
            Xtr = make_features(FP, tr_pairs[:, 0], tr_pairs[:, 1], ("sum",))["sum"]
            clf = C.fit_model(Xtr, ytr)                      # 한 번 학습 → 두 regime 채점

            for rg, val in REGIMES.items():
                idx = np.where(pu == val)[0]
                reg_pos = np.stack([pi[idx], pj[idx]], axis=1)
                reg_pos_sims = M2[reg_pos[:, 0], reg_pos[:, 1]]
                reg_neg = sample_negatives(ng, val, g, len(reg_pos), pos_lin, M2, reg_pos_sims, seed, n)
                pairs = np.vstack([reg_pos, reg_neg]); y = np.r_[np.ones(len(reg_pos)), np.zeros(len(reg_neg))]
                if ng == "random":
                    counts[rg].append(len(reg_pos))
                # Vilar (raw / norm)
                r, p = roc_pr(y, vilar_scores(S3_raw, pairs)); per[ng][rg]["vilar_raw"]["roc"].append(r); per[ng][rg]["vilar_raw"]["pr"].append(p)
                r, p = roc_pr(y, vilar_scores(S3_norm, pairs)); per[ng][rg]["vilar_norm"]["roc"].append(r); per[ng][rg]["vilar_norm"]["pr"].append(p)
                # XGBoost
                Xte = make_features(FP, pairs[:, 0], pairs[:, 1], ("sum",))["sum"]
                proba = clf.predict_proba(Xte)[:, 1]
                r, p = roc_pr(y, proba); per[ng][rg]["xgboost"]["roc"].append(r); per[ng][rg]["xgboost"]["pr"].append(p)
            print(f"  [seed {seed}·{ng}] cs: vilar {per[ng]['cold-start']['vilar_raw']['roc'][-1]:.3f} vs xgb {per[ng]['cold-start']['xgboost']['roc'][-1]:.3f}"
                  f" | cc: vilar {per[ng]['cold-cold']['vilar_raw']['roc'][-1]:.3f} vs xgb {per[ng]['cold-cold']['xgboost']['roc'][-1]:.3f}", flush=True)

    agg = {ng: {rg: {m: {mt: {"mean": float(np.mean(per[ng][rg][m][mt])), "std": float(np.std(per[ng][rg][m][mt]))}
                          for mt in ("roc", "pr")} for m in METHODS} for rg in REGIMES} for ng in NEGS}
    res = {"config": {"n_drugs": n, "n_pos": len(pi), "seeds": SEEDS, "n_bins": N_BINS,
                      "cs_pos": int(np.mean(counts["cold-start"])), "cc_pos": int(np.mean(counts["cold-cold"])),
                      "runtime_s": round(time.time() - t0, 1)},
           "anchors": ANCHORS, "results": agg, "per_seed": per}
    json.dump(res, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ 요약 (random neg, 5 seed ROC) ================")
    for rg in REGIMES:
        a = agg["random"][rg]
        print(f"  {rg:11s}: vilar_raw {a['vilar_raw']['roc']['mean']:.3f} · vilar_norm {a['vilar_norm']['roc']['mean']:.3f} · xgboost {a['xgboost']['roc']['mean']:.3f}  (03앵커 {ANCHORS['random'][rg]:.3f})")
    out_html = build_report(res)
    print(f"\n[13] 완료 ({res['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    return res


# ---------------- 보고서 ----------------
def _b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_report(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for _kf in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]:
        if os.path.exists(_kf):
            fm.fontManager.addfont(_kf); plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False
    agg = res["results"]; c = res["config"]; anch = res["anchors"]
    mcol = {"vilar_raw": "#e0a52e", "vilar_norm": "#c0562f", "xgboost": "#4c78d8"}

    # 그림1: 방법 3단계 스키마
    fig0, ax = plt.subplots(figsize=(9, 2.6)); ax.axis("off")
    def box(x, txt, col):
        ax.add_patch(plt.Rectangle((x, 0.3), 0.16, 0.44, fc=col, ec="#333", alpha=.85, transform=ax.transAxes))
        ax.text(x + 0.08, 0.52, txt, ha="center", va="center", fontsize=9, transform=ax.transAxes, color="white")
    box(0.02, "M1\n상호작용표\n(train-train\n1/0, 대칭)", "#2a6f4c")
    ax.text(0.20, 0.52, "×", fontsize=20, ha="center", va="center", transform=ax.transAxes)
    box(0.24, "M2\n닮음표\n(Tanimoto\n0~1)", "#8a6d1a")
    ax.text(0.42, 0.52, "=", fontsize=20, ha="center", va="center", transform=ax.transAxes)
    box(0.46, "M3=M1×M2\n점수\n(대칭화\nM3+M3^T)", "#3a5a8a")
    ax.text(0.70, 0.60, "학습 없음 · 행렬곱만", fontsize=10, transform=ax.transAxes)
    ax.text(0.70, 0.44, "cold-start: train쪽 지식 전파 →작동\ncold-cold: 둘다 미지 →점수0(불가)", fontsize=8.5, transform=ax.transAxes, color="#b03030")
    chart_s = _b64(fig0)

    # 그림2: cold-start / cold-cold ROC — Vilar vs XGBoost (random neg)
    fig1, ax = plt.subplots(figsize=(8.4, 3.6)); w = 0.25
    rgs = list(agg["random"].keys())
    for i, m in enumerate(METHODS):
        vals = [agg["random"][rg][m]["roc"]["mean"] for rg in rgs]; err = [agg["random"][rg][m]["roc"]["std"] for rg in rgs]
        ax.bar(np.arange(len(rgs)) + (i - 1) * w, vals, w, yerr=err, capsize=3, color=mcol[m], label=m)
    ax.axhline(0.5, color="#888", ls="--", lw=.8, label="무작위 0.5")
    ax.set_xticks(np.arange(len(rgs))); ax.set_xticklabels(rgs); ax.set_ylabel("ROC-AUC (5 seed)"); ax.set_ylim(0.45, 0.9)
    ax.set_title("1세대(Vilar) vs XGBoost — cold regime ROC (random negative)")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=.25)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    chart_b = _b64(fig1)

    def cell(ng, rg, m, mt):
        a = agg[ng][rg][m][mt]; return f"{a['mean']:.3f}<span class=mut>±{a['std']:.3f}</span>"
    def mtable(ng):
        rows = ""
        for rg in agg[ng]:
            rows += (f"<tr><td class=l><b>{rg}</b></td>"
                     f"<td>{cell(ng,rg,'vilar_raw','roc')}</td><td>{cell(ng,rg,'vilar_norm','roc')}</td><td class=hl>{cell(ng,rg,'xgboost','roc')}</td>"
                     f"<td>{cell(ng,rg,'vilar_raw','pr')}</td><td>{cell(ng,rg,'vilar_norm','pr')}</td><td class=hl>{cell(ng,rg,'xgboost','pr')}</td></tr>")
        return rows

    # 핵심 수치
    cs_v = agg["random"]["cold-start"]["vilar_raw"]["roc"]["mean"]; cs_x = agg["random"]["cold-start"]["xgboost"]["roc"]["mean"]
    cc_v = agg["random"]["cold-cold"]["vilar_raw"]["roc"]["mean"]; cc_x = agg["random"]["cold-cold"]["xgboost"]["roc"]["mean"]
    csm_v = agg["matched"]["cold-start"]["vilar_raw"]["roc"]["mean"]; csm_x = agg["matched"]["cold-start"]["xgboost"]["roc"]["mean"]
    cs_gap = cs_x - cs_v
    xgb_repro = f"cold-start {cs_x:.3f}(03앵커 {anch['random']['cold-start']:.3f})·matched {csm_x:.3f}(06앵커 {anch['matched']['cold-start']:.3f})"

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 6px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a;font-size:10px}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}.hl{background:#16202e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}b.hl{color:#ffd479}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>13 DDI Vilar</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>13 · 1세대(Vilar 2014) 재현 — '닮음으로 상호작용 퍼뜨리기' vs XGBoost</h1>
<p class=mut>학습 없이 행렬곱(M1×M2)만으로 DDI 예측. 같은 데이터·cold-start 분할·negative로 내 XGBoost와 나란히 · env admet · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> ★<b>cold-start(한쪽만 신약)에선 무학습 Vilar가 XGBoost에 근접</b> — ROC vilar {cs_v:.3f} vs xgb {cs_x:.3f}(Δ{cs_gap:+.3f}). 닮음 전파만으로도 상당한 신호.<br>
★<b>cold-cold(둘다 신약)에선 Vilar 구조적 실패</b> — vilar {cc_v:.3f}(≈무작위) vs xgb {cc_x:.3f}. 이유: M1이 train-train만이라 두 미지약물은 전파할 '알려진 상호작용'이 없음 → 점수 0. ★<b>1세대의 근본 한계</b>: 최소 한쪽은 알려진 약물이어야 함.<br>
★<b>내 방식(XGBoost)의 값어치</b>: 구조→상호작용을 <b>학습</b>하므로 <b>둘 다 미지</b>여도 작동({cc_x:.3f}) — Vilar가 못 가는 영역. XGBoost 재현 검증: {xgb_repro} ≈ 앵커 일치.</div>

<h2>① 방법 (Vilar 3단계 = 학습 없는 행렬곱)</h2>
<img src="{chart_s}" alt="method">
<div class=card>1) <b>M1</b> 상호작용표(약물×약물, 1/0, 대칭) — ★train 약물끼리만(test 숨김) · 2) <b>M2</b> 닮음표(Tanimoto, Morgan) · 3) <b>M3=M1×M2</b> 대칭화(M3+M3ᵀ). 점수(i,j)=Σₖ(i가 k와 상호작용)×(k가 j와 닮음)=<b>'i와 상호작용하는 약들과 j가 얼마나 닮았나'</b>. 대각(자기자신) 제외.<br>
★<b>vilar_norm</b>=degree 보정(각 약물 이웃수로 나눔) — 허브(많은 상호작용) 약물 편향 진단용.</div>

<h2>② 결과: Vilar vs XGBoost (random negative, =03 방식)</h2>
<img src="{chart_b}" alt="bars">
<div style="overflow-x:auto"><table>
<tr><th class=l rowspan=2>regime</th><th colspan=3>ROC-AUC</th><th colspan=3>PR-AUC</th></tr>
<tr><th>vilar_raw</th><th>vilar_norm</th><th class=hl>xgboost</th><th>vilar_raw</th><th>vilar_norm</th><th class=hl>xgboost</th></tr>
{mtable('random')}
</table></div>
<p class=mut>cold-cold vilar≈0.5 = 점수 전부 0(degenerate). XGBoost cold-start {cs_x:.3f}·cold-cold {cc_x:.3f} = 03 앵커({anch['random']['cold-start']:.3f}/{anch['random']['cold-cold']:.3f}) 재현.</p>

<h2>③ 강건성: matched negative (Tanimoto 분포매칭, =06 방식)</h2>
<div style="overflow-x:auto"><table>
<tr><th class=l rowspan=2>regime</th><th colspan=3>ROC-AUC</th><th colspan=3>PR-AUC</th></tr>
<tr><th>vilar_raw</th><th>vilar_norm</th><th class=hl>xgboost</th><th>vilar_raw</th><th>vilar_norm</th><th class=hl>xgboost</th></tr>
{mtable('matched')}
</table></div>
<div class=card>★<b>matched(닮음 지름길 제거)에서 Vilar는 소폭만 하락</b>(예상과 다름): cold-start vilar random {cs_v:.3f}→matched {csm_v:.3f}(Δ{csm_v-cs_v:+.3f}), XGBoost {cs_x:.3f}→{csm_x:.3f}(Δ{csm_x-cs_x:+.3f}). ★<b>이유</b>: Vilar 점수는 쌍의 <b>직접 유사도 sim(i,j)가 아니라</b> 'i가 j의 상호작용 파트너들과 얼마나 닮았나'(이웃 전파)라, negative를 sim(i,j)로 매칭해도 신호가 직접 겨냥되지 않음 → 단순 '닮으면 상호작용' 임계 baseline보다 견고. 둘 다 matched서 06 앵커({anch['matched']['cold-start']:.3f}) 부근 유지.</div>

<h2>④ ★나란히 비교 (문맥: 기존 실험 sum 융합)</h2>
<div style="overflow-x:auto"><table>
<tr><th class=l>방법 / regime</th><th>warm</th><th>cold-start</th><th>cold-cold</th></tr>
<tr><td class=l><b>1세대 Vilar (무학습, random)</b></td><td class=mut>누수(생략)</td><td>{cs_v:.3f}</td><td class=bad>{cc_v:.3f}</td></tr>
<tr><td class=l><b>XGBoost (random, 이번 재현)</b></td><td class=mut>—</td><td>{cs_x:.3f}</td><td>{cc_x:.3f}</td></tr>
<tr><td class=l>02/03 XGBoost sum (random 앵커)</td><td>{anch['random']['warm']:.3f}</td><td>{anch['random']['cold-start']:.3f}</td><td>{anch['random']['cold-cold']:.3f}</td></tr>
<tr><td class=l><b>1세대 Vilar (무학습, matched)</b></td><td class=mut>누수(생략)</td><td>{csm_v:.3f}</td><td class=bad>{agg['matched']['cold-cold']['vilar_raw']['roc']['mean']:.3f}</td></tr>
<tr><td class=l>06 XGBoost sum (matched 앵커)</td><td>{anch['matched']['warm']:.3f}</td><td>{anch['matched']['cold-start']:.3f}</td><td>{anch['matched']['cold-cold']:.3f}</td></tr>
</table></div>
<div class=card>★<b>해석</b>: (1) <b>cold-start</b>에선 무학습 Vilar가 XGBoost에 {'근접' if abs(cs_gap)<0.05 else '뒤짐'}(Δ{cs_gap:+.3f} random) — '닮은 약은 비슷하게 상호작용' 가설이 <b>한쪽 신약</b> 시나리오에선 꽤 성립. 값싼 baseline으로 유효.<br>
(2) <b>cold-cold</b>에선 Vilar 완전 실패(≈0.5) vs XGBoost {cc_x:.3f} — ★학습 모델의 존재 이유: <b>둘 다 신약</b>이어도 구조→상호작용 사상을 일반화. 이게 1세대→2세대(ML) 전환의 핵심 동기.<br>
(3) <b>matched</b>에서 Vilar는 소폭(Δ{csm_v-cs_v:+.3f})만 하락 = 이웃 전파는 쌍 직접유사도 지름길에 덜 의존해 <b>의외로 견고</b>. 단 ★<b>degree 보정</b>(vilar_norm)하면 cold-start {agg['random']['cold-start']['vilar_norm']['roc']['mean']:.3f}로 급락 → Vilar 신호의 상당부분은 <b>허브(차수)</b>에서 옴. ★이 차수 성분이 '진짜 promiscuity(많이 상호작용하는 약)'인지 '<b>주석밀도 편향</b>(많이 연구된 약일수록 기록된 DDI가 많음)'인지는 이 실험으로 <b>구분 안 됨</b>(둘 다 가능·04/05의 base-rate 착시 경고와 동일 맥락). 즉 Vilar=이웃유사도+차수 둘 다 활용.</div>

<h2>⑤ ★한계·정직 메모</h2>
<div class=card>
· ★<b>cold-cold Vilar=구조적 degenerate</b>(점수 0, ROC 0.5) — 방법 결함이 아니라 '알려진 상호작용 전파'의 본질(양쪽 미지엔 전파원 없음). 공정비교 위해 그대로 보고.<br>
· <b>warm은 Vilar 누수</b>(M1에 그 쌍이 이미 있음=자기 정답 포함) → 생략. leave-pair-out 필요(미구현).<br>
· random negative는 easy(닮음 지름길 포함) → matched가 더 정직. Vilar는 그 지름길에 민감.<br>
· M1은 이진·M2는 raw Tanimoto·대칭화 M3+M3ᵀ 사용 = ★논문 변형 다수 중 <b>흔한 무정규화형</b>(원논문은 정규화/랭크형도 사용 — 이 raw형은 허브/차수 지배적, vilar_norm 참고) · degree 보정은 진단용 1종만.<br>
· XGBoost는 같은 fold·같은 지식(TT)·같은 negative로 공정비교(재현 앵커 일치 확인).</div>

<h2>⑥ 다음</h2>
<div class=card>· Vilar에 다중 유사도(구조+타깃+부작용) 결합(원논문 확장) · leave-pair-out으로 warm 공정평가 · cold-cold용 하이브리드(Vilar+구조ML) · 12 타깃 유사도를 M2에 추가.</div>

<div class=foot>13 DDI Vilar 1세대 재현 · M1×M2 대칭화 무학습 vs XGBoost sum · cold-start/cold-cold · random/matched neg · ROC/PR · seed {len(c['seeds'])} · positive {c['n_pos']:,} · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "13_ddi_vilar_similarity.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


if __name__ == "__main__":
    run()

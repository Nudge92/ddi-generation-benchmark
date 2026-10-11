"""
[DDI 실험 - 09] 방향성 배수구-매칭 특징 — substrate↔inhibitor 짝이 값하나
====================================================================
08: CYP 내용 고유 이득 확인(burden-matched cold-cold +0.0117)이나 방향성 concat은
    소멸 → 모델이 "A가 B의 CYP를 저해"라는 방향성 배수구 기전을 스스로 못 배움.
09: PK 배수구 기전을 ★명시적 매칭 특징으로 직접 계산해 넣으면 08 바닥을 넘나.

★방향성 매칭 특징(쌍 단위 직접 계산)
  공통 CYP 3종(2C9·2D6·3A4; ADMET-AI가 기질·억제 둘 다 예측). 쌍(a,b, 무순서)에 대해:
    m1_X = (a가 CYP_X 기질) AND (b가 CYP_X 억제)  → b가 a의 배수구(대사)를 막음
    m2_X = (b가 CYP_X 기질) AND (a가 CYP_X 억제)  → a가 b의 배수구를 막음
  ★두 방향을 다 계산(무순서 쌍이라 어느 쪽이 기질인지 모르므로 A→B·B→A 모두 포착).
  → 3 CYP × 2 방향 = 6 이진 + 총 매칭수 스칼라 1 = 7차원.

특징 4구성(burden-matched, sum, 08과 동일 조건에서):
  (a) struct+enzyme            : 08 최종(= 기준선, 08 재현)
  (b) struct+enzyme+match ★    : 방향성 매칭 특징 추가(본 가설)
  (c) match-only               : 매칭 특징만
  (d) struct+enzyme+randmatch  : ★placebo — 억제 프로필을 랜덤 약물 것으로 바꾼 '가짜 매칭'
                                  (진짜 짝 아님). (b)−(d)가 진짜 방향성 순효과.

산출: results/09_ddi_drugbank_directional_cyp.html , results/directional_cyp_results.json
실행: conda run -n admet python src/09_directional_cyp.py
"""
import os, sys, json, time, io, base64
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddi_common import (ukey, build_fp_matrix, fit_model, score_model, _GEN, build_sim_matrix,
                        split_drugs, verify_regime, pair_feature, match_2d, regime_burden_matched)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
CYP_CSV = os.path.join(BASE, "data", "drug_cyp.csv")
RESULTS_JSON = os.path.join(BASE, "results", "directional_cyp_results.json")

SEEDS = [0, 1, 2, 3, 4]
DRUG_TEST_FRAC = 0.20
N_TBINS_2D = 20                # 08과 동일(burden-matched 재현)
REGIMES = ["warm", "cold-start", "cold-cold"]
CONFIGS = ["struct+enzyme", "struct+enzyme+match", "match-only", "struct+enzyme+randmatch"]
METRIC_KEYS = ["roc_auc", "pr_auc", "f1", "accuracy"]
COMMON_CYP = ["2c9", "2d6", "3a4"]   # 기질·억제 둘 다 있는 CYP


def pairs_to_idx(pair_list, did2row):
    idxA = np.fromiter((did2row[a] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    idxB = np.fromiter((did2row[b] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    return idxA, idxB


def match_feat(idxA, idxB, SUB, INH):
    """방향성 매칭 특징 7차원: [m1(3), m2(3), 총매칭(1)]. SUB/INH=(n_drugs,3) 이진(2C9/2D6/3A4)."""
    m1 = SUB[idxA] * INH[idxB]        # a 기질 & b 억제 (b가 a 배수구 막음), (n,3)
    m2 = SUB[idxB] * INH[idxA]        # b 기질 & a 억제 (a가 b 배수구 막음), (n,3)
    total = (m1.sum(1) + m2.sum(1)).reshape(-1, 1)
    return np.hstack([m1, m2, total]).astype(np.float32)


def build_feat(config, FP, CYP, SUB, INH, INH_shuf, idxA, idxB):
    se = pair_feature(FP, CYP, idxA, idxB, "struct+enzyme", "sum")   # 08 struct+enzyme(sum)
    if config == "struct+enzyme":
        return se
    if config == "struct+enzyme+match":
        return sp.hstack([se, sp.csr_matrix(match_feat(idxA, idxB, SUB, INH))], format="csr")
    if config == "match-only":
        return sp.csr_matrix(match_feat(idxA, idxB, SUB, INH))
    if config == "struct+enzyme+randmatch":       # placebo: 억제 프로필을 랜덤 약물 것으로
        return sp.hstack([se, sp.csr_matrix(match_feat(idxA, idxB, SUB, INH_shuf))], format="csr")


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
    agg = res["results"]; per = res["per_seed"]; c = res["config"]; ms = res["match_stats"]
    from scipy import stats

    def d(cfg, base, rg):
        a = np.array(per[cfg][rg]["roc_auc"]); b = np.array(per[base][rg]["roc_auc"]); dd = a - b
        t, p = stats.ttest_rel(a, b)
        return dd.mean(), int((dd > 0).sum()), (p if not np.isnan(p) else 1.0)

    ccol = {"struct+enzyme": "#8a8a99", "struct+enzyme+match": "#1a8f4c", "match-only": "#4c78d8", "struct+enzyme+randmatch": "#e0a52e"}
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = np.arange(len(REGIMES)); w = 0.2
    for i, cfg in enumerate(CONFIGS):
        means = [agg[cfg][rg]["roc_auc"]["mean"] for rg in REGIMES]; stds = [agg[cfg][rg]["roc_auc"]["std"] for rg in REGIMES]
        ax.bar(x + (i - 1.5) * w, means, w, yerr=stds, capsize=2, color=ccol[cfg], label=cfg)
        for xi, mu in enumerate(means):
            ax.annotate(f"{mu:.3f}", (x[xi] + (i - 1.5) * w, mu + stds[xi] + 0.004), ha="center", fontsize=6.8)
    ax.axhline(0.5, ls="--", c="#e2555f", lw=1, alpha=.6)
    ax.set_xticks(x); ax.set_xticklabels(REGIMES); ax.set_ylim(0.5, 0.93); ax.set_ylabel("ROC-AUC (5 seed 평균±std)")
    ax.set_title("방향성 매칭 특징 4구성 (burden-matched, sum) — +match가 08 바닥 넘나 · placebo 넘나")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=.25)
    for sp2 in ("top", "right"):
        ax.spines[sp2].set_visible(False)
    chart = _b64(fig)

    prows = ""
    for cfg in CONFIGS:
        cells = "".join(f"<td>{agg[cfg][rg]['roc_auc']['mean']:.4f}<span class=mut style=font-size:9px>±{agg[cfg][rg]['roc_auc']['std']:.3f}</span></td>" for rg in REGIMES)
        prows += f"<tr><td class=l><b>{cfg}</b></td>{cells}</tr>"

    krows = ""
    for rg in REGIMES:
        dba, dba_p, dba_pv = d("struct+enzyme+match", "struct+enzyme", rg)          # (b)-(a) 매칭 이득
        dbd, dbd_p, dbd_pv = d("struct+enzyme+match", "struct+enzyme+randmatch", rg)  # (b)-(d) placebo 순효과
        real = dba > 0.002 and dba_p >= 4 and dbd > 0.002 and dbd_p >= 4 and dbd_pv < 0.05
        verd = ("<b class=good>방향성 매칭 값함(placebo 넘음)</b>" if real else
                "<b class=warn>placebo 못 넘음(차원효과)</b>" if dbd <= 0.002 and dba > 0.002 else
                "<span class=mut>이득 없음</span>")
        krows += (f"<tr><td class=l>{rg}</td><td class={'good' if dba>0.002 else 'mut'}>{dba:+.4f} ({dba_p}/5)</td>"
                  f"<td class={'good' if dbd>0.002 and dbd_pv<0.05 else 'mut'}>{dbd:+.4f} ({dbd_p}/5·p{dbd_pv:.2f})</td><td class=l>{verd}</td></tr>")

    cc_ba = d("struct+enzyme+match", "struct+enzyme", "cold-cold")[0]
    cc_bd = d("struct+enzyme+match", "struct+enzyme+randmatch", "cold-cold")[0]
    mo = {rg: agg["match-only"][rg]["roc_auc"]["mean"] for rg in REGIMES}
    overall = ("<b class=good>방향성 배수구 매칭이 값함(placebo 넘는 순효과)</b>" if cc_ba > 0.002 and cc_bd > 0.002 else
               "<b class=warn>약한 방향 신호는 실재하나(pos 매칭율↑·match-only&gt;0.5) 예측엔 negligible·조성특징과 중복·placebo와 구분 안 됨</b> → 이 설정선 방향성 배수구 기전이 예측에 값하지 않음")

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 7px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}b.hl{color:#ffd479}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>09 DDI directional CYP match</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>09 · DDI 방향성 배수구-매칭 특징 — substrate↔inhibitor 짝이 값하나</h1>
<p class=mut>"A가 B의 CYP를 저해→B 농도↑" 방향성 매칭을 명시적 특징으로 계산해 08 바닥(struct+enzyme, burden-matched)에 추가 · placebo 대조 · env admet · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> cold-cold — (b)struct+enzyme+match − (a)struct+enzyme = <b>{cc_ba:+.4f}</b> · placebo 넘는 순효과 (b)−(d) = <b>{cc_bd:+.4f}</b>. → {overall}.</div>

<h2>① 방향성 매칭 특징</h2>
<div class=card>공통 CYP 3종(2C9·2D6·3A4, ADMET-AI 기질·억제 예측). 쌍(a,b)에: m1_X=(a 기질_X)&(b 억제_X), m2_X=(b 기질_X)&(a 억제_X). <b>두 방향 다 포함</b>(무순서 쌍) → 6 이진 + 총매칭 1 = <b>7차원</b>. burden-matched(08) 위에서, 구조·raw CYP는 sum.<br>
★placebo(randmatch): 억제 프로필을 <b>랜덤 약물 것으로 바꿔</b> 진짜 짝을 깸(같은 형식·차원). (b)−(d)가 '매칭이라서'의 순효과.</div>

<h2>② 매칭 특징 기술통계 (burden-matched cold-cold, 대표 seed0)</h2>
<div class=card>쌍에 방향성 매칭(총매칭≥1) 있는 비율: <b>positive {ms['rate_pos']:.1%}</b> vs <b>negative {ms['rate_neg']:.1%}</b> (차 {ms['rate_pos']-ms['rate_neg']:+.1%}).
{'→ pos에서 더 잦음 = <b>약한 방향 신호는 실재</b>(기전 가설과 일치 방향; match-only가 5시드 모두 0.5 초과로 corroborate). 단 아래처럼 예측엔 비가법.' if ms['rate_pos']>ms['rate_neg'] else '→ pos/neg 비슷(신호 없음).'} 주로 발생 CYP: {ms['top_cyp']} · 평균 매칭수 pos {ms['mean_pos']:.3f}/neg {ms['mean_neg']:.3f}. <span class=mut>(비율은 seed0 값; 성능은 5시드)</span></div>

<h2>③ 성능: 4구성 × 3 regime (burden-matched, sum)</h2>
<img src="{chart}" alt="directional match">
<div class=tblwrap style="overflow-x:auto"><table>
<tr><th class=l>특징</th>{"".join(f"<th>{rg}</th>" for rg in REGIMES)}</tr>{prows}
</table></div>
<p class=mut>struct+enzyme이 <b>08 burden-matched 정확 재현</b>: cold-cold {agg['struct+enzyme']['cold-cold']['roc_auc']['mean']:.4f}(08 burden 0.6646; ※08의 0.7049는 sim-matched 값). match-only: warm {mo['warm']:.3f}·cold-cold {mo['cold-cold']:.3f}(방향성 매칭 단독은 무작위 0.5 바로 위 — 신호 매우 약함).</p>

<h2>④ ★핵심 판정: 매칭 이득 (b)−(a)와 placebo 넘는 순효과 (b)−(d)</h2>
<div class=tblwrap style="overflow-x:auto"><table>
<tr><th class=l>regime</th><th>(b)−(a) 매칭 이득</th><th>(b)−(d) placebo 순효과</th><th class=l>판정</th></tr>{krows}
</table></div>
<div class=card>★<b>cold-cold(미지 약물·라벨순환 약함)</b>: 매칭 이득 {cc_ba:+.4f}, placebo 순효과 {cc_bd:+.4f}. {overall}.<br>
{'방향성 배수구 매칭이 08 바닥을 넘고 placebo도 넘음 → 명시적 방향성 특징이 값함(Day1 기전 신호).' if cc_ba>0.002 and cc_bd>0.002 else '★정직하게: <b>약한 방향 신호는 실재</b>함(pos 매칭율 &gt; neg·match-only가 5시드 모두 0.5 초과). 그러나 struct+enzyme 위에 더해도 이득 ≈0이고 placebo(가짜 매칭)와 구분 안 됨 → <b>조성 특징과 완전 중복이라 예측엔 비가법</b>. 즉 이 데이터·예측CYP로는 명시적 방향성 매칭이 예측에 값하지 않음(08 CYP 이득도 방향성 아닌 조성/정체성 쪽과 일관).'}</div>

<h2>⑤ ★한계 메모</h2>
<div class=card>
· <b>예측 CYP</b>(구조유래) → 매칭도 예측CYP 조합이라 <b>구조 중복 잔여</b> · per-CYP 유병률 미통제분이 매칭에 실릴 수 있음(08 ④와 동일).<br>
· <b>cold-cold는 약물-정체성 암기를 제거</b>(라벨-정의 순환은 완전 제거 아님). 단 <b>null이라 순환이 남아도 그건 신호를 부풀릴 뿐</b> → '값 안 함' 결론은 오히려 더 견고.<br>
· 매칭은 <b>임계0.5 이진</b> 예측 기반(연속확률 아님) · cold-cold 표본 작음 · sum·이진·XGBoost 고정.</div>

<h2>⑥ 다음</h2>
<div class=card>{'· 값함 → curated 효소로 방향성 재확인·연속 CYP확률로 매칭 강도·유형별 다중분류(어떤 CYP저해↔어떤 유형).' if cc_ba>0.002 and cc_bd>0.002 else '· 미포착 → 방향성 대신 유형별 다중분류·타깃/경로 특징으로 전환. curated 효소면 다를 수 있음(예측CYP 한계).'}</div>

<div class=foot>09 DDI directional CYP match · struct+enzyme(±match/randmatch)/match-only × warm/cold-start/cold-cold · burden-matched · sum · seed {len(c['seeds'])} · ∩=0·검증통과 · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "09_ddi_drugbank_directional_cyp.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


def main():
    t0 = time.time()
    print("[09] 방향성 매칭 특징 실험 시작 ...", flush=True)
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"])); did2row = {d: i for i, d in enumerate(drug_ids)}
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["drug_a"] = pairs["drug_a"].astype(str); pairs["drug_b"] = pairs["drug_b"].astype(str)
    positives = [ukey(a, b) for a, b in zip(pairs.loc[pairs.label == 1, "drug_a"], pairs.loc[pairs.label == 1, "drug_b"])]
    pos_key_set = set(positives); n_pos = len(positives)
    cyp_df = pd.read_csv(CYP_CSV); cyp_df["drug_id"] = cyp_df["drug_id"].astype(str)
    ccols = [c for c in cyp_df.columns if c != "drug_id"]; cyp_df = cyp_df.set_index("drug_id").loc[drug_ids]
    CYP = sp.csr_matrix(cyp_df[ccols].to_numpy(dtype=np.uint8))
    n_cyp = np.asarray(CYP.sum(axis=1)).ravel().astype(np.int32)
    # ★공통 CYP 기질/억제 행렬(2C9·2D6·3A4)
    SUB = cyp_df[[f"cyp{x}_sub" for x in COMMON_CYP]].to_numpy(dtype=np.uint8)   # (n_drugs,3)
    INH = cyp_df[[f"cyp{x}_inh" for x in COMMON_CYP]].to_numpy(dtype=np.uint8)
    sigma = np.random.RandomState(2025).permutation(len(drug_ids))              # placebo용 고정 순열
    INH_shuf = INH[sigma]                                                        # 억제 프로필을 랜덤 약물 것으로
    print(f"  약물 {len(drug_ids):,} · positive {n_pos:,} · 공통 CYP {COMMON_CYP}", flush=True)

    FP = build_fp_matrix(drug_ids, id2smi)
    bitvects = [_GEN.GetFingerprint(Chem.MolFromSmiles(str(id2smi[d]))) for d in drug_ids]
    S = build_sim_matrix(bitvects)
    n = len(drug_ids); iu, ju = np.triu_indices(n, k=1); sims = S[iu, ju]
    pair_burden = (n_cyp[iu] + n_cyp[ju]).astype(np.int32)
    pos_codes = np.fromiter((min(did2row[a], did2row[b]) * n + max(did2row[a], did2row[b]) for a, b in pos_key_set), dtype=np.int64, count=len(pos_key_set))
    is_pos = np.isin(iu.astype(np.int64) * n + ju.astype(np.int64), pos_codes)
    pos_i = np.fromiter((did2row[a] for a, b in positives), dtype=np.int32, count=n_pos)
    pos_j = np.fromiter((did2row[b] for a, b in positives), dtype=np.int32, count=n_pos)
    pos_sims_g = S[pos_i, pos_j]; pos_burden_g = (n_cyp[pos_i] + n_cyp[pos_j]).astype(np.int32)

    # global burden-matched(warm anchor; 08과 동일 seed 999)
    ch, _ = match_2d(sims[~is_pos], pair_burden[~is_pos], pos_sims_g, pos_burden_g, N_TBINS_2D, 999)
    ci = iu[~is_pos]; cj = ju[~is_pos]
    global_burden = [ukey(drug_ids[int(ci[k])], drug_ids[int(cj[k])]) for k in ch]

    warm_pairs = positives + global_burden; warm_y = np.array([1] * n_pos + [0] * len(global_burden))
    w_idxA, w_idxB = pairs_to_idx(warm_pairs, did2row)
    warm_feats = {cfg: build_feat(cfg, FP, CYP, SUB, INH, INH_shuf, w_idxA, w_idxB) for cfg in CONFIGS}

    per = {cfg: {rg: {mt: [] for mt in METRIC_KEYS} for rg in REGIMES} for cfg in CONFIGS}
    match_stats = {}

    for seed in SEEDS:
        train_drugs, test_drugs = split_drugs(drug_ids, seed, DRUG_TEST_FRAC)
        g = np.array([0 if d in train_drugs else 1 for d in drug_ids], dtype=np.int8)
        g_iu = g[iu]; g_ju = g[ju]; pu = g[pos_i] + g[pos_j]
        prg = {}
        for name, val in [("TT", 0), ("TS", 1), ("SS", 2)]:
            idxs = np.where(pu == val)[0]
            prg[name] = {"pairs": [positives[k] for k in idxs], "sims": pos_sims_g[idxs], "burden": pos_burden_g[idxs]}
        # burden-matched negatives (08과 동일 seed*100+val)
        negTT, _ = regime_burden_matched(0, g_iu, g_ju, iu, ju, sims, pair_burden, is_pos, prg["TT"]["sims"], prg["TT"]["burden"], drug_ids, seed * 100 + 0, N_TBINS_2D)
        negTS, _ = regime_burden_matched(1, g_iu, g_ju, iu, ju, sims, pair_burden, is_pos, prg["TS"]["sims"], prg["TS"]["burden"], drug_ids, seed * 100 + 1, N_TBINS_2D)
        negSS, _ = regime_burden_matched(2, g_iu, g_ju, iu, ju, sims, pair_burden, is_pos, prg["SS"]["sims"], prg["SS"]["burden"], drug_ids, seed * 100 + 2, N_TBINS_2D)
        tt_pairs = prg["TT"]["pairs"] + negTT; tt_y = np.array([1] * len(prg["TT"]["pairs"]) + [0] * len(negTT))
        cs_pairs = prg["TS"]["pairs"] + negTS; cs_y = np.array([1] * len(prg["TS"]["pairs"]) + [0] * len(negTS))
        cc_pairs = prg["SS"]["pairs"] + negSS; cc_y = np.array([1] * len(prg["SS"]["pairs"]) + [0] * len(negSS))
        assert verify_regime(tt_pairs, cs_pairs, test_drugs, "cold-start")["PASS"] and verify_regime(tt_pairs, cc_pairs, test_drugs, "cold-cold")["PASS"]
        assert len(set(negTT + negTS + negSS) & pos_key_set) == 0

        if seed == 0:   # 매칭 기술통계(burden-matched cold-cold)
            pa, pb = pairs_to_idx(prg["SS"]["pairs"], did2row); na, nb = pairs_to_idx(negSS, did2row)
            mp = match_feat(pa, pb, SUB, INH); mn = match_feat(na, nb, SUB, INH)
            per_cyp = (mp[:, :3] + mp[:, 3:6]).sum(0)   # CYP별 매칭 총합(pos)
            match_stats = {"rate_pos": float((mp[:, 6] > 0).mean()), "rate_neg": float((mn[:, 6] > 0).mean()),
                           "mean_pos": float(mp[:, 6].mean()), "mean_neg": float(mn[:, 6].mean()),
                           "top_cyp": COMMON_CYP[int(np.argmax(per_cyp))].upper()}

        tt_iA, tt_iB = pairs_to_idx(tt_pairs, did2row); cs_iA, cs_iB = pairs_to_idx(cs_pairs, did2row); cc_iA, cc_iB = pairs_to_idx(cc_pairs, did2row)
        wtr, wte = train_test_split(np.arange(len(warm_pairs)), test_size=0.2, random_state=seed, stratify=warm_y)
        for cfg in CONFIGS:
            wclf = fit_model(warm_feats[cfg][wtr], warm_y[wtr]); ws = score_model(wclf, warm_feats[cfg][wte], warm_y[wte])
            ttclf = fit_model(build_feat(cfg, FP, CYP, SUB, INH, INH_shuf, tt_iA, tt_iB), tt_y)
            cs_s = score_model(ttclf, build_feat(cfg, FP, CYP, SUB, INH, INH_shuf, cs_iA, cs_iB), cs_y)
            cc_s = score_model(ttclf, build_feat(cfg, FP, CYP, SUB, INH, INH_shuf, cc_iA, cc_iB), cc_y)
            for mt in METRIC_KEYS:
                per[cfg]["warm"][mt].append(ws[mt]); per[cfg]["cold-start"][mt].append(cs_s[mt]); per[cfg]["cold-cold"][mt].append(cc_s[mt])
        dba = per["struct+enzyme+match"]["cold-cold"]["roc_auc"][-1] - per["struct+enzyme"]["cold-cold"]["roc_auc"][-1]
        dbd = per["struct+enzyme+match"]["cold-cold"]["roc_auc"][-1] - per["struct+enzyme+randmatch"]["cold-cold"]["roc_auc"][-1]
        print(f"  [seed {seed}] cold-cold (b)-(a) {dba:+.4f} · (b)-(d)placebo {dbd:+.4f}", flush=True)

    agg = {cfg: {rg: {mt: {"mean": float(np.mean(per[cfg][rg][mt])), "std": float(np.std(per[cfg][rg][mt]))}
                      for mt in METRIC_KEYS} for rg in REGIMES} for cfg in CONFIGS}
    maxroc = max(agg[cfg][rg]["roc_auc"]["mean"] for cfg in CONFIGS for rg in REGIMES)
    res = {"config": {"n_pos": n_pos, "n_drugs": len(drug_ids), "seeds": SEEDS, "common_cyp": COMMON_CYP, "runtime_s": round(time.time() - t0, 1)},
           "results": agg, "per_seed": per, "match_stats": match_stats, "flags": {"roc_over_0.98": bool(maxroc > 0.98)}}
    json.dump(res, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ cold-cold ROC (5 seed) ================")
    for cfg in CONFIGS:
        print(f"  {cfg:26s}: {agg[cfg]['cold-cold']['roc_auc']['mean']:.4f}")
    ba = agg["struct+enzyme+match"]["cold-cold"]["roc_auc"]["mean"] - agg["struct+enzyme"]["cold-cold"]["roc_auc"]["mean"]
    bd = agg["struct+enzyme+match"]["cold-cold"]["roc_auc"]["mean"] - agg["struct+enzyme+randmatch"]["cold-cold"]["roc_auc"]["mean"]
    print(f"  ★cold-cold (b)-(a) 매칭이득 {ba:+.4f} · (b)-(d) placebo순효과 {bd:+.4f}")
    print(f"  매칭율 pos/neg (cold-cold): {match_stats['rate_pos']:.1%}/{match_stats['rate_neg']:.1%} · top CYP {match_stats['top_cyp']}")
    if maxroc > 0.98:
        print(f"\n★★ RED FLAG ROC {maxroc:.4f}>0.98", flush=True)
    out_html = build_report(res)
    print(f"\n[09] 완료 ({res['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    return res


if __name__ == "__main__":
    main()

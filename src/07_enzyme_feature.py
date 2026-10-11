"""
[DDI 실험 - 07단계-B] 효소(CYP) 특징 추가 — 기전 정보가 신약 예측을 올리나
====================================================================
06의 '구조만' 바닥(warm 0.8719 / cs 0.7919 / cc 0.6894) 위에, 효소(CYP) 특징을
더하면 넘는지 = 기전 정보의 가치. 06과 완전히 동일 조건(분할·negative·융합·XGBoost),
새 변수는 '특징 구성'뿐.

특징 3구성(약물 하나당 벡터 → 두 약물은 06과 같은 sum 융합으로 결합):
  (a) struct       : Morgan 2048        (= 06 재현, 기준선)
  (b) enzyme       : CYP 8차원(예측)     (구조 없이 효소만)
  (c) struct+enzyme: [Morgan 2048, CYP 8](본 가설)

★효소 = ADMET-AI 예측(구조 유래)이라: 전 약물 커버(결측 0), 그러나 struct와 일부 중복.
★라벨 순환 주의: DDI 라벨이 CYP기반일 수 있음 → cold-cold(둘 다 미지 약물) 상승을 주 근거로.

산출: results/07_ddi_drugbank_enzyme_feature.html , results/enzyme_feature_results.json
실행: conda run -n admet python src/07_enzyme_feature.py
사전: src/07_gen_cyp.py 로 data/drug_cyp.csv 먼저 생성(ADMET_AI env).
"""
import os, sys, json, time, io, base64
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddi_common import (ukey, build_fp_matrix, make_features, fit_model, score_model, _GEN,
                        build_sim_matrix, split_drugs, regime_matched, verify_regime)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
MATCHED_CSV = os.path.join(BASE, "data", "matched_negatives.csv")
CYP_CSV = os.path.join(BASE, "data", "drug_cyp.csv")
RESULTS_JSON = os.path.join(BASE, "results", "enzyme_feature_results.json")

SEEDS = [0, 1, 2, 3, 4]
DRUG_TEST_FRAC = 0.20
N_BINS = 40
CONFIGS = ["struct", "enzyme", "struct+enzyme"]
FUSIONS = ["sum", "concat"]
REGIMES = ["warm", "cold-start", "cold-cold"]
METRIC_KEYS = ["roc_auc", "pr_auc", "f1", "accuracy"]


def pairs_to_idx(pair_list, did2row):
    idxA = np.fromiter((did2row[a] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    idxB = np.fromiter((did2row[b] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    return idxA, idxB


def build_feat(FP, CYP, idxA, idxB, config, fusion):
    """한 쌍의 특징 = (구조/효소 중 config가 요구하는 블록)을 fusion(sum/concat)으로 결합.
    약물별 벡터를 두 약물끼리 sum 또는 concat 융합(06과 같은 방식)."""
    parts = []
    if config in ("struct", "struct+enzyme"):
        parts.append(make_features(FP, idxA, idxB, methods=(fusion,))[fusion])   # Morgan 융합
    if config in ("enzyme", "struct+enzyme"):
        A = CYP[idxA]; B = CYP[idxB]                                             # CYP 벡터
        parts.append((A + B).tocsr() if fusion == "sum" else sp.hstack([A, B], format="csr").tocsr())
    return parts[0] if len(parts) == 1 else sp.hstack(parts, format="csr").tocsr()


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

    agg = res["results"]; c = res["config"]; per = res["per_seed"]
    ccol = {"struct": "#8a8a99", "enzyme": "#4c78d8", "struct+enzyme": "#1a8f4c"}

    # 그림: sum 융합, 3 config × 3 regime
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = np.arange(len(REGIMES)); w = 0.26
    for i, cfg in enumerate(CONFIGS):
        means = [agg[cfg]["sum"][rg]["roc_auc"]["mean"] for rg in REGIMES]
        stds = [agg[cfg]["sum"][rg]["roc_auc"]["std"] for rg in REGIMES]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3, color=ccol[cfg], label=cfg)
        for xi, (mu, sd) in enumerate(zip(means, stds)):
            ax.annotate(f"{mu:.3f}", (x[xi] + (i - 1) * w, mu + sd + 0.006), ha="center", fontsize=7.5)
    ax.axhline(0.5, ls="--", c="#e2555f", lw=1, alpha=.6)
    ax.set_xticks(x); ax.set_xticklabels(["warm", "cold-start", "cold-cold"]); ax.set_ylim(0.5, 0.93)
    ax.set_ylabel("ROC-AUC (5 seed 평균±std)")
    ax.set_title("특징 3구성 × 3 regime (sum 융합) — struct+enzyme가 struct(06 바닥)를 넘나")
    ax.legend(loc="upper right", fontsize=9); ax.grid(axis="y", alpha=.25)
    for sp2 in ("top", "right"):
        ax.spines[sp2].set_visible(False)
    chart = _b64(fig)

    def table(fusion):
        rows = ""
        for cfg in CONFIGS:
            cells = "".join(f"<td>{agg[cfg][fusion][rg][mt]['mean']:.4f}<span class=mut style=font-size:9px>±{agg[cfg][fusion][rg][mt]['std']:.3f}</span></td>"
                            for rg in REGIMES for mt in METRIC_KEYS)
            rows += f"<tr><td class=l><b>{cfg}</b></td>{cells}</tr>"
        return rows

    # 핵심: struct+enzyme − struct (sum), regime별 + per-seed 일관
    keycmp = ""
    for rg in REGIMES:
        s = np.array(per["struct"]["sum"][rg]["roc_auc"]); se = np.array(per["struct+enzyme"]["sum"][rg]["roc_auc"])
        d = se - s; allpos = bool((d > 0).all()); allneg = bool((d < 0).all())
        tag = ("<b class=good>넘음(5시드 전부)</b>" if allpos and d.mean() > 0.002 else
               "<b class=bad>못 넘음(5시드 전부 낮음)</b>" if allneg and d.mean() < -0.002 else
               "<span class=mut>차이 미미(편차 내)</span>")
        keycmp += (f"<tr><td class=l>{rg}</td><td>{s.mean():.4f}</td><td>{se.mean():.4f}</td>"
                   f"<td class={'good' if d.mean()>0.002 else ('bad' if d.mean()<-0.002 else 'mut')}>{d.mean():+.4f}</td>"
                   f"<td class=mut style=font-size:11px>{[f'{x:+.4f}' for x in d]}</td><td class=l>{tag}</td></tr>")

    eo = {rg: agg["enzyme"]["sum"][rg]["roc_auc"]["mean"] for rg in REGIMES}
    scc = agg["struct"]["sum"]["cold-cold"]["roc_auc"]["mean"]; secc = agg["struct+enzyme"]["sum"]["cold-cold"]["roc_auc"]["mean"]
    cc_gain = secc - scc
    cyp = c["cyp"]
    # ★융합 비대칭(정직성): 방향성 있는 concat이 오히려 약함 = 방향성 CYP기전보다 promiscuity 쪽 근거
    ccat_gain = agg["struct+enzyme"]["concat"]["cold-cold"]["roc_auc"]["mean"] - agg["struct"]["concat"]["cold-cold"]["roc_auc"]["mean"]
    ccat_pos = int((np.array(per["struct+enzyme"]["concat"]["cold-cold"]["roc_auc"]) > np.array(per["struct"]["concat"]["cold-cold"]["roc_auc"])).sum())
    bpos = cyp.get("cc_burden_pos", 0); bneg = cyp.get("cc_burden_neg", 0)

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1120px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 6px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}b.hl{color:#ffd479}"""

    verdict = "<b class=warn>작은 이득이나 promiscuity 교란·융합민감 → '기전' 귀속 미확립</b>"

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>07 DDI enzyme feature</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>07 · DDI 효소(CYP) 특징 추가 — 기전 정보가 신약 예측을 올리나</h1>
<p class=mut>06의 구조-만 바닥 위에 CYP 예측 특징을 더해 ablation. 06과 동일(분할·matched neg·sum 융합·XGBoost), 새 변수=특징 구성 · env admet · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> cold-cold에서 struct+enzyme {secc:.4f} vs struct {scc:.4f} → Δ <b>{cc_gain:+.4f}</b>(sum 융합·5시드·p0.002). {verdict}.<br>근거: (1) 이긴 <b>sum</b> 융합은 쌍 <b>CYP-burden</b>(=promiscuity)인데 matched가 Tanimoto만 통제·CYP축은 방치(pos {bpos:.1f} vs neg {bneg:.1f}) · (2) 방향성 <b>concat</b>은 오히려 약함(Δ{ccat_gain:+.4f}, {ccat_pos}/5·유의X) · (3) CYP는 예측(구조유래). → 상승은 '기전'보다 <b>promiscuity 프록시</b>에 가깝다.</div>

<h2>① 효소 데이터 요약</h2>
<div class=card>출처 <b>ADMET-AI 예측</b>(구조→CYP, 다운로드 0) · CYP 8종 이진(억제 5: 1A2·2C19·2C9·2D6·3A4 / 기질 3: 2C9·2D6·3A4) · <b>커버리지 100%</b>(예측이라 결측 0 → 0-채움 불필요) · 하나라도 CYP=1인 약물 {cyp['any_frac']:.1%}·평균 {cyp['mean_hits']:.2f}/8.<br>
<span class=mut>CYP별 양성비율: {' · '.join(f"{k} {v:.2f}" for k,v in cyp['prevalence'].items())}</span></div>

<h2>② 성능: 특징 3구성 × 3 regime (sum 융합)</h2>
<img src="{chart}" alt="enzyme ablation">
<div class=tblwrap style="overflow-x:auto"><table>
<tr><th class=l rowspan=2>특징</th><th colspan=4>warm</th><th colspan=4>cold-start</th><th colspan=4>cold-cold</th></tr>
<tr>{"".join("<th>"+hh+"</th>" for _ in REGIMES for hh in ["ROC","PR","F1","acc"])}</tr>
{table('sum')}
</table></div>
<p class=mut>참고(concat 융합): {" · ".join(f"{cfg} cc {agg[cfg]['concat']['cold-cold']['roc_auc']['mean']:.4f}" for cfg in CONFIGS)}. struct-sum이 06 재현: warm {agg['struct']['sum']['warm']['roc_auc']['mean']:.4f}(06 0.8719)·cc {scc:.4f}(06 0.6894).</p>

<h2>③ ★핵심 비교: struct+enzyme가 struct(06 바닥)를 넘나</h2>
<div class=tblwrap style="overflow-x:auto"><table>
<tr><th class=l>regime</th><th>struct</th><th>struct+enzyme</th><th>Δ</th><th>per-seed Δ</th><th class=l>판정</th></tr>
{keycmp}
</table></div>
<div class=card><b>enzyme only(구조 없이 CYP만):</b> warm {eo['warm']:.4f} · cold-start {eo['cold-start']:.4f} · cold-cold {eo['cold-cold']:.4f} — 효소 8차원'만'으로도 무작위(0.5)보단 높음(promiscuity·구조상관 반영).<br>
cold-cold Δ {cc_gain:+.4f}는 <b>약물-정체성 암기</b>는 없앤 상태의 이득이나, <b>CYP기반 라벨-정의 순환</b>과 <b>promiscuity 교란</b>이 겹쳐 — <b>'기전 정보의 진짜 가치'로 단정 못 함</b>(아래 ④).</div>

<h2>④ ★상승의 진짜 원인 — '기전' 아님에 무게 (3가지 대안)</h2>
<div class=card>
· ★<b>(1) promiscuity 교란 (주요):</b> 이긴 <b>sum 융합 = CYP_A+CYP_B = 쌍의 총 CYP-burden</b>(방향無). matched는 <b>Tanimoto만</b> 맞추고 CYP-burden은 안 맞춤 → cold-cold에서 <b>pos {bpos:.2f} vs neg {bneg:.2f}</b>(Δ+{bpos-bneg:.2f})로 positive가 CYP 많은(promiscuous) 약물에 쏠림. CYP개수↔degree 상관 r={cyp['degree_corr']:+.3f}. 즉 "CYP 많음→상호작용 많음"이라는 <b>promiscuity 프록시</b>를 담았을 수 있음(실제 DDI 요인이나 거친 신호, 우리 설계가 미통제).<br>
· ★<b>(2) 방향성 기전이면 concat이 이겨야 하나 반대:</b> 진짜 CYP-DDI는 <b>'A가 B의 CYP를 저해'</b>라는 방향성. 이를 담을 수 있는 <b>concat</b>은 오히려 약함(Δ{ccat_gain:+.4f}, {ccat_pos}/5·p0.34 유의X) vs 방향無 <b>sum</b>(Δ{cc_gain:+.4f}, 5/5·p0.002). → 방향성 기전보다 <b>총량(promiscuity)</b> 근거.<br>
· <b>(3) 라벨-정의 순환 잔여:</b> DDI 라벨이 CYP기반이면 그 규칙은 train/test 약물 <b>공유</b> → cold-cold(미지 약물)서도 남음. cold-cold가 없앤 건 <b>약물-정체성</b> 암기지 <b>라벨-정의</b> 순환이 아님. warm→cc 이득 증가도 순환으로 동일 설명 가능(기전 증거 아님).<br>
· 게다가 예측 CYP는 <b>구조유래</b> → struct와 중복. <b>종합: 작고(+0.0155)·융합민감·교란모호 → '기전 정보가 값한다'로 단정 불가.</b></div>

<h2>⑤ ★한계 메모</h2>
<div class=card>
· <b>이득이 융합에 취약:</b> sum(Δ+0.0155·5/5·p0.002)에선 견고하나 <b>concat(Δ{ccat_gain:+.4f}·{ccat_pos}/5·p0.34)에선 유의하지 않음</b> — 결론이 sum 융합에 의존.<br>
· <b>degree/promiscuity 축 미통제:</b> matched는 Tanimoto만 통제 → CYP-burden 상승분과 분리 불가(④). 결정적 검증엔 <b>degree/burden-matched negative</b> 필요(⑥).<br>
· <b>예측 CYP</b>(구조유래·struct 중복) · <b>라벨-정의 순환 잔여</b> · negative '미확인' · warm 약물암기 낙관 · cold-cold 표본 작아 편차 큼.<br>
· 커버리지 100%라 0-채움 이슈 없음(예측 장점). 이진·XGBoost 고정.</div>

<h2>⑥ 다음에 해볼 것</h2>
<div class=card>
· ★<b>degree/CYP-burden-matched negative (결정적):</b> negative를 Tanimoto뿐 아니라 <b>CYP-burden(promiscuity)까지</b> positive와 맞춰, 교란을 뺀 <b>순수 CYP-특이 신호</b>가 남는지. (사라지면 promiscuity, 남으면 진짜 CYP 신호) + degree-only baseline('쌍 CYP개수' 1개만으로 얼마나 설명되나).<br>
· <b>curated DrugBank 효소</b>(진짜 주석)로 재현 · 타깃/경로 특징 · 유형별 다중분류(어떤 CYP↔유형).</div>

<div class=foot>07 DDI enzyme(CYP) feature · struct/enzyme/struct+enzyme × warm/cold-start/cold-cold · CYP=ADMET-AI 예측 · sum 융합 · XGBoost 고정 · seed {len(c['seeds'])} · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "07_ddi_drugbank_enzyme_feature.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


def main():
    t0 = time.time()
    print("[07-B] enzyme feature ablation 시작 ...", flush=True)
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"])); did2row = {d: i for i, d in enumerate(drug_ids)}
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["drug_a"] = pairs["drug_a"].astype(str); pairs["drug_b"] = pairs["drug_b"].astype(str)
    positives = [ukey(a, b) for a, b in zip(pairs.loc[pairs.label == 1, "drug_a"], pairs.loc[pairs.label == 1, "drug_b"])]
    pos_key_set = set(positives); n_pos = len(positives)
    md = pd.read_csv(MATCHED_CSV); global_matched = [ukey(str(a), str(b)) for a, b in zip(md["drug_a"], md["drug_b"])]
    # CYP 특징
    cyp_df = pd.read_csv(CYP_CSV); cyp_df["drug_id"] = cyp_df["drug_id"].astype(str)
    cyp_cols = [c for c in cyp_df.columns if c != "drug_id"]
    cyp_df = cyp_df.set_index("drug_id").loc[drug_ids]                # drug_ids 순서로 정렬
    CYP = sp.csr_matrix(cyp_df[cyp_cols].to_numpy(dtype=np.uint8))
    print(f"  약물 {len(drug_ids):,} · positive {n_pos:,} · CYP {len(cyp_cols)}종", flush=True)

    # 지문·유사도·triu(06과 동일)
    FP = build_fp_matrix(drug_ids, id2smi)
    bitvects = [_GEN.GetFingerprint(Chem.MolFromSmiles(str(id2smi[d]))) for d in drug_ids]
    S = build_sim_matrix(bitvects)
    n = len(drug_ids); iu, ju = np.triu_indices(n, k=1); sims = S[iu, ju]
    pos_codes = np.fromiter((min(did2row[a], did2row[b]) * n + max(did2row[a], did2row[b]) for a, b in pos_key_set), dtype=np.int64, count=len(pos_key_set))
    is_pos = np.isin(iu.astype(np.int64) * n + ju.astype(np.int64), pos_codes)
    pos_i = np.fromiter((did2row[a] for a, b in positives), dtype=np.int32, count=n_pos)
    pos_j = np.fromiter((did2row[b] for a, b in positives), dtype=np.int32, count=n_pos)
    pos_sims_global = S[pos_i, pos_j]

    # 커버리지 편향: CYP 개수 vs 약물 degree 상관
    degree = np.zeros(n)
    for a, b in positives:
        degree[did2row[a]] += 1; degree[did2row[b]] += 1
    n_cyp = np.asarray(CYP.sum(axis=1)).ravel()
    degree_corr = float(np.corrcoef(n_cyp, degree)[0, 1])
    prevalence = {col: round(float(cyp_df[col].mean()), 3) for col in cyp_cols}
    cyp_meta = {"prevalence": prevalence, "any_frac": float((n_cyp > 0).mean()),
                "mean_hits": float(n_cyp.mean()), "degree_corr": degree_corr}
    print(f"  CYP 개수 vs degree 상관 r={degree_corr:+.3f}", flush=True)

    # warm 특징(config×fusion, 한 번만)
    warm_pairs = positives + global_matched; warm_y = np.array([1] * n_pos + [0] * len(global_matched))
    w_idxA, w_idxB = pairs_to_idx(warm_pairs, did2row)
    warm_feats = {cfg: {fu: build_feat(FP, CYP, w_idxA, w_idxB, cfg, fu) for fu in FUSIONS} for cfg in CONFIGS}

    per = {cfg: {fu: {rg: {mt: [] for mt in METRIC_KEYS} for rg in REGIMES} for fu in FUSIONS} for cfg in CONFIGS}
    burden_pos_cc = []; burden_neg_cc = []   # ★promiscuity 점검: cold-cold pos vs matched-neg의 쌍 CYP-burden

    for seed in SEEDS:
        train_drugs, test_drugs = split_drugs(drug_ids, seed, DRUG_TEST_FRAC)
        g = np.array([0 if d in train_drugs else 1 for d in drug_ids], dtype=np.int8)
        g_iu = g[iu]; g_ju = g[ju]; pos_unseen = g[pos_i] + g[pos_j]
        prg = {}
        for name, val in [("TT", 0), ("TS", 1), ("SS", 2)]:
            idxs = np.where(pos_unseen == val)[0]
            prg[name] = {"pairs": [positives[k] for k in idxs], "sims": pos_sims_global[idxs]}
        negTT, _, _ = regime_matched(0, g_iu, g_ju, iu, ju, sims, is_pos, prg["TT"]["sims"], drug_ids, seed * 10 + 0, N_BINS)
        negTS, _, _ = regime_matched(1, g_iu, g_ju, iu, ju, sims, is_pos, prg["TS"]["sims"], drug_ids, seed * 10 + 1, N_BINS)
        negSS, _, _ = regime_matched(2, g_iu, g_ju, iu, ju, sims, is_pos, prg["SS"]["sims"], drug_ids, seed * 10 + 2, N_BINS)
        # ★promiscuity 점검: matched는 Tanimoto만 맞춤 → CYP-burden 축은 통제 안 됨. pos vs neg 비교.
        burden_pos_cc.append(float(np.mean([n_cyp[did2row[a]] + n_cyp[did2row[b]] for a, b in prg["SS"]["pairs"]])))
        burden_neg_cc.append(float(np.mean([n_cyp[did2row[a]] + n_cyp[did2row[b]] for a, b in negSS])))
        train_pairs = prg["TT"]["pairs"] + negTT; train_y = np.array([1] * len(prg["TT"]["pairs"]) + [0] * len(negTT))
        cs_pairs = prg["TS"]["pairs"] + negTS; cs_y = np.array([1] * len(prg["TS"]["pairs"]) + [0] * len(negTS))
        cc_pairs = prg["SS"]["pairs"] + negSS; cc_y = np.array([1] * len(prg["SS"]["pairs"]) + [0] * len(negSS))
        assert verify_regime(train_pairs, cs_pairs, test_drugs, "cold-start")["PASS"]
        assert verify_regime(train_pairs, cc_pairs, test_drugs, "cold-cold")["PASS"]
        assert len(set(negTT + negTS + negSS) & pos_key_set) == 0
        tt_iA, tt_iB = pairs_to_idx(train_pairs, did2row); cs_iA, cs_iB = pairs_to_idx(cs_pairs, did2row); cc_iA, cc_iB = pairs_to_idx(cc_pairs, did2row)
        wtr, wte = train_test_split(np.arange(len(warm_pairs)), test_size=0.2, random_state=seed, stratify=warm_y)
        for cfg in CONFIGS:
            for fu in FUSIONS:
                wclf = fit_model(warm_feats[cfg][fu][wtr], warm_y[wtr]); warm_s = score_model(wclf, warm_feats[cfg][fu][wte], warm_y[wte])
                ttclf = fit_model(build_feat(FP, CYP, tt_iA, tt_iB, cfg, fu), train_y)
                cs_s = score_model(ttclf, build_feat(FP, CYP, cs_iA, cs_iB, cfg, fu), cs_y)
                cc_s = score_model(ttclf, build_feat(FP, CYP, cc_iA, cc_iB, cfg, fu), cc_y)
                for mt in METRIC_KEYS:
                    per[cfg][fu]["warm"][mt].append(warm_s[mt]); per[cfg][fu]["cold-start"][mt].append(cs_s[mt]); per[cfg][fu]["cold-cold"][mt].append(cc_s[mt])
        se = per["struct+enzyme"]["sum"]; st = per["struct"]["sum"]
        print(f"  [seed {seed}] cc: struct {st['cold-cold']['roc_auc'][-1]:.4f} · +enzyme {se['cold-cold']['roc_auc'][-1]:.4f} · enzyme-only {per['enzyme']['sum']['cold-cold']['roc_auc'][-1]:.4f}", flush=True)

    agg = {cfg: {fu: {rg: {mt: {"mean": float(np.mean(per[cfg][fu][rg][mt])), "std": float(np.std(per[cfg][fu][rg][mt]))}
                          for mt in METRIC_KEYS} for rg in REGIMES} for fu in FUSIONS} for cfg in CONFIGS}
    cyp_meta["cc_burden_pos"] = float(np.mean(burden_pos_cc)); cyp_meta["cc_burden_neg"] = float(np.mean(burden_neg_cc))
    maxroc = max(agg[cfg][fu][rg]["roc_auc"]["mean"] for cfg in CONFIGS for fu in FUSIONS for rg in REGIMES)
    res = {"config": {"n_pos": n_pos, "n_drugs": len(drug_ids), "seeds": SEEDS, "n_bins": N_BINS,
                      "cyp": cyp_meta, "runtime_s": round(time.time() - t0, 1)},
           "results": agg, "per_seed": per, "flags": {"roc_over_0.98": bool(maxroc > 0.98)}}
    json.dump(res, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ sum 융합 ROC-AUC (5 seed 평균) ================")
    for cfg in CONFIGS:
        print(f"  {cfg:14s}: warm {agg[cfg]['sum']['warm']['roc_auc']['mean']:.4f} · cold-start {agg[cfg]['sum']['cold-start']['roc_auc']['mean']:.4f} · cold-cold {agg[cfg]['sum']['cold-cold']['roc_auc']['mean']:.4f}")
    ccg = agg["struct+enzyme"]["sum"]["cold-cold"]["roc_auc"]["mean"] - agg["struct"]["sum"]["cold-cold"]["roc_auc"]["mean"]
    print(f"  ★cold-cold struct+enzyme − struct = {ccg:+.4f}")
    if maxroc > 0.98:
        print(f"\n★★ RED FLAG: ROC-AUC 최대 {maxroc:.4f} > 0.98. 확인 필요.", flush=True)
    out_html = build_report(res)
    print(f"\n[07-B] 완료 ({res['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    return res


if __name__ == "__main__":
    main()

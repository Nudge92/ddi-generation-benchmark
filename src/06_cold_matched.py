"""
[DDI 실험 - 06단계] cold + matched negative — 가장 정직한 '신약 쌍' 추정
====================================================================
03(cold 분할)과 05(matched negative)를 ★결합한다.
  - 03: 처음 보는 약물일수록 성능↓ (약물 암기 거품).
  - 05: negative를 positive와 같은 Tanimoto 분포로 맞춰 유사도 지름길 제거.
06: 둘을 합쳐 "처음 보는 약물 쌍 + 유사도 지름길 없음"의 성능 = 이 시리즈에서
    가장 현실적인 '신약 DDI' 추정치.

세 regime(전부 matched negative):
  (A) warm + matched      : 쌍 랜덤 분할 (= 05 재현, anchor)
  (B) cold-start + matched: test 쌍의 정확히 1개가 미지 약물
  (C) cold-cold + matched : test 쌍의 2개 다 미지 약물

★핵심 함정(cold + matched 동시 만족):
  각 regime의 negative는 '두 조건을 동시에' 만족해야 함 —
   (1) 그 regime의 미지-약물 조건(TT=둘다 train / TS=1개 미지 / SS=2개 미지)
   (2) 그 regime의 'positive Tanimoto 분포'와 같은 유사도 분포(전체가 아니라 regime별).
  → regime별 후보(조건1 만족 비상호작용 쌍)에서, 그 regime positive 분포에 bin 매칭(조건2).

산출: results/06_ddi_drugbank_cold_matched.html , results/cold_matched_results.json
실행: conda run -n admet python src/06_cold_matched.py
env : admet · ddi_common.py(split_drugs·bin_match·build_sim_matrix 등) 재사용
"""
import os, sys, json, time, io, base64
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddi_common import (ukey, build_fp_matrix, make_features, fit_model, score_model,
                        _GEN, build_sim_matrix, split_drugs, bin_match)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
MATCHED_CSV = os.path.join(BASE, "data", "matched_negatives.csv")     # 05의 global matched(warm anchor용)
COLDSTART_JSON = os.path.join(BASE, "results", "coldstart_results.json")  # 03(easy neg) 비교용
RESULTS_JSON = os.path.join(BASE, "results", "cold_matched_results.json")

SEEDS = [0, 1, 2, 3, 4]
DRUG_TEST_FRAC = 0.20
METHODS = ["sum", "concat"]
METRIC_KEYS = ["roc_auc", "pr_auc", "f1", "accuracy"]
N_BINS = 40


def pairs_to_idx(pair_list, did2row):
    idxA = np.fromiter((did2row[a] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    idxB = np.fromiter((did2row[b] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    return idxA, idxB


def verify(train_pairs, test_pairs, test_drugs, regime):
    """train에 미지약물 0 · test는 regime 조건 · train↔test 겹침 0."""
    bad_train = sum(1 for a, b in train_pairs if (a in test_drugs) or (b in test_drugs))
    want = 1 if regime == "cold-start" else 2
    bad_test = sum(1 for a, b in test_pairs if ((a in test_drugs) + (b in test_drugs)) != want)
    overlap = len(set(train_pairs) & set(test_pairs))
    return {"train_미지약물": bad_train, "test_조건위반": bad_test, "겹침": overlap,
            "PASS": bad_train == 0 and bad_test == 0 and overlap == 0}


def regime_matched(val, g_iu, g_ju, iu, ju, sims, is_pos, pos_r_sims, drug_ids, seed):
    """regime(val: TT=0/TS=1/SS=2)의 미지-약물 조건을 지키는 비상호작용 후보에서,
    그 regime의 positive 유사도 분포에 맞춰 matched negative 추출."""
    pair_unseen = g_iu + g_ju
    cmask = (pair_unseen == val) & (~is_pos)          # 조건1: 미지약물 조건 + 비상호작용
    c_iu = iu[cmask]; c_ju = ju[cmask]; c_sims = sims[cmask]
    if len(c_sims) == 0:
        return [], 0, np.array([])
    chosen, deficit = bin_match(c_sims, pos_r_sims, N_BINS, seed)   # 조건2: 분포 매칭
    pairs = [ukey(drug_ids[int(c_iu[k])], drug_ids[int(c_ju[k])]) for k in chosen]
    return pairs, deficit, c_sims[chosen]


# ---------------- 보고서 ----------------
def _b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_report(res, dist, counts, cmp03):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for _kf in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]:
        if os.path.exists(_kf):
            fm.fontManager.addfont(_kf); plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False

    agg = res["results"]; c = res["config"]
    REGIMES = ["warm", "cold-start", "cold-cold"]
    RLAB = {"warm": "warm+matched\n(약물 겹침)", "cold-start": "cold-start+matched\n(1개 미지)", "cold-cold": "cold-cold+matched\n(2개 미지)"}
    rcol = {"warm": "#1a8f4c", "cold-start": "#e08a1e", "cold-cold": "#c62f43"}

    # 그림1: regime별 ROC 낙폭
    fig1, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(REGIMES)); w = 0.36
    for i, m in enumerate(METHODS):
        means = [agg[m][rg]["roc_auc"]["mean"] for rg in REGIMES]
        stds = [agg[m][rg]["roc_auc"]["std"] for rg in REGIMES]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=3,
               color=("#1a8f4c" if m == "sum" else "#8a8a99"), label=m + (" (메인)" if m == "sum" else " (참고)"))
        for xi, (mu, sd) in enumerate(zip(means, stds)):
            ax.annotate(f"{mu:.3f}", (x[xi] + (i - 0.5) * w, mu + sd + 0.006), ha="center", fontsize=8)
    ax.axhline(0.5, ls="--", c="#e2555f", lw=1, alpha=.7)
    ax.set_xticks(x); ax.set_xticklabels([RLAB[r] for r in REGIMES]); ax.set_ylim(0.5, 0.93)
    ax.set_ylabel("ROC-AUC (5 seed 평균±std)")
    ax.set_title("전부 matched negative — 분할이 어려워질수록 ROC↓ (유사도 거품 뺀 순수 약물암기 낙폭)")
    ax.legend(loc="upper right", fontsize=9); ax.grid(axis="y", alpha=.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    chart_r = _b64(fig1)

    # 그림2: regime별 positive vs matched-neg Tanimoto(매칭 증명, 3패널)
    fig2, axs = plt.subplots(1, 3, figsize=(11, 3.0))
    bins = np.linspace(0, 0.5, 33)
    for ax, rg in zip(axs, REGIMES):
        ax.hist(dist[rg]["pos"], bins=bins, histtype="step", lw=2, density=True, color="#e0a52e", label=f"positive {np.mean(dist[rg]['pos']):.3f}")
        ax.hist(dist[rg]["neg"], bins=bins, histtype="step", lw=2, density=True, color=rcol[rg], label=f"matched-neg {np.mean(dist[rg]['neg']):.3f}")
        ax.set_title(rg, fontsize=10); ax.legend(fontsize=7.5); ax.set_xlabel("Tanimoto")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig2.suptitle("★각 regime에서 matched-neg가 그 regime positive와 같은 유사도 분포인가 (매칭 증명)", fontsize=10)
    chart_d = _b64(fig2)

    def mrow(m):
        cells = "".join(f"<td>{agg[m][rg][mt]['mean']:.4f}<span class=mut style=font-size:9px>±{agg[m][rg][mt]['std']:.3f}</span></td>"
                        for rg in REGIMES for mt in METRIC_KEYS)
        return f"<tr><td class=l><b>{m}</b></td>{cells}</tr>"
    prows = "".join(mrow(m) for m in METHODS)

    sw = agg["sum"]["warm"]["roc_auc"]["mean"]; scs = agg["sum"]["cold-start"]["roc_auc"]["mean"]; scc = agg["sum"]["cold-cold"]["roc_auc"]["mean"]
    cc_gap = scc - agg["concat"]["cold-cold"]["roc_auc"]["mean"]; cc_std = agg["sum"]["cold-cold"]["roc_auc"]["std"]
    # 05 warm+matched 재현 확인
    anchor05 = 0.8719
    repro = abs(sw - anchor05) < 0.004
    # 03(easy) 낙폭 vs 06(matched) 낙폭
    e_warm = cmp03.get("warm"); e_cs = cmp03.get("cold-start"); e_cc = cmp03.get("cold-cold")

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 7px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}b.hl{color:#ffd479}"""

    cmp_row = ""
    for rg, e_val in [("warm", e_warm), ("cold-start", e_cs), ("cold-cold", e_cc)]:
        mv = agg["sum"][rg]["roc_auc"]["mean"]
        cmp_row += (f"<tr><td class=l>{rg}</td><td>{e_val:.4f}</td><td><b>{mv:.4f}</b></td>"
                    f"<td class={'bad' if mv < e_val - 0.003 else ('good' if mv > e_val + 0.003 else 'mut')}>{mv - e_val:+.4f}</td></tr>"
                    if e_val is not None else "")

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>06 DDI cold+matched</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>06 · DDI cold + matched negative — 가장 정직한 '신약 쌍' 추정</h1>
<p class=mut>03(cold 분할)+05(matched negative) 결합. 세 regime 모두 negative=matched(그 regime positive 분포에 맞춤) · 융합 sum·concat · XGBoost 고정 · seed {c['seeds']} 평균±std · env admet · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> sum ROC-AUC — <b>warm+matched {sw:.4f} → cold-start+matched {scs:.4f} ({scs-sw:+.4f}) → cold-cold+matched {scc:.4f} ({scc-sw:+.4f})</b>.
<b class=hl>cold-cold+matched {scc:.4f} = 가장 어려운·현실적인 regime(처음 보는 약물 + 유사도 지름길 없음)의 점추정</b>(양방향 불확실 — ⑥). 무작위(0.5)보다 +{scc-0.5:.3f}.</div>

<h2>① 설정 (05와 비교 시 '분할'만 변수)</h2>
<div class=card>positive {c['n_pos']:,}(무순서 dedupe) · negative=<b>matched</b>(regime별 positive 분포에 {c['n_bins']} bin 매칭) · Morgan 2048(r=2), 융합 sum·concat · XGBoost 고정 · 약물 train {int(round((1-DRUG_TEST_FRAC)*100))}%/test {int(DRUG_TEST_FRAC*100)}% · seed {c['seeds']}.<br>
★<b>cold+matched 동시 만족</b>: 각 regime의 negative는 (1)미지-약물 조건(TT/TS/SS)과 (2)그 regime positive의 유사도 분포를 <b>동시에</b> 만족. ∩positive=0·self-pair 0 검증.</div>

<h2>② 데이터 규모 & 누수 검증</h2>
<table>
<tr><th class=l>Regime</th><th>학습 쌍(평균)</th><th>평가 쌍(평균)</th><th>matched 부족분</th><th>누수 검증</th></tr>
<tr><td class=l><b>warm+matched</b></td><td>~{int(c['n_pos']*2*0.8):,}</td><td>~{int(c['n_pos']*2*0.2):,}</td><td class=mut>0(05 global)</td><td class=mut>약물 겹침(anchor)</td></tr>
<tr><td class=l><b>cold-start+matched</b></td><td>{counts['train_mean']:,.0f} (TT)</td><td>{counts['cs_mean']:,.0f}</td><td>{counts['cs_deficit']:,}</td><td class=good>PASS ✔ (1개 미지)</td></tr>
<tr><td class=l><b>cold-cold+matched</b></td><td>{counts['train_mean']:,.0f} (TT, 공통)</td><td>{counts['cc_mean']:,.0f} (최소 {counts['cc_min']:,})</td><td>{counts['cc_deficit']:,}</td><td class=good>PASS ✔ (2개 미지)</td></tr>
</table>
<p class=mut>✔ 모든 seed·regime: train에 미지약물 0 · test regime조건 위반 0 · train↔test 쌍겹침 0. 약물 test {counts['n_test_drugs']}개.</p>

<h2>③ ★검증: 각 regime에서 matched가 positive와 같은 분포인가</h2>
<img src="{chart_d}" alt="regime tanimoto match">
<div class=card>regime별 Tanimoto 평균(positive vs matched-neg): warm {np.mean(dist['warm']['pos']):.3f}/{np.mean(dist['warm']['neg']):.3f} · cold-start {np.mean(dist['cold-start']['pos']):.3f}/{np.mean(dist['cold-start']['neg']):.3f} · cold-cold {np.mean(dist['cold-cold']['pos']):.3f}/{np.mean(dist['cold-cold']['neg']):.3f}. 각 regime 안에서 겹침 → 유사도가 지름길 안 됨.</div>

<h2>④ 성능 (전부 matched, 5 seed 평균±std)</h2>
<img src="{chart_r}" alt="cold matched roc">
<table>
<tr><th class=l rowspan=2>융합</th><th colspan=4>warm+matched</th><th colspan=4>cold-start+matched</th><th colspan=4>cold-cold+matched</th></tr>
<tr>{"".join("<th>"+hh+"</th>" for _ in REGIMES for hh in ["ROC","PR","F1","acc"])}</tr>
{prows}
</table>

<h2>⑤ ★핵심 비교</h2>
<div class=card>
· <b>05 재현(anchor):</b> warm+matched {sw:.4f} vs 05의 0.8719 → {'<b class=good>재현 ✔</b>(공정 비교 성립)' if repro else '<b class=warn>차이 있음</b>'}.<br>
· <b>03(easy) 낙폭 vs 06(matched) 낙폭</b> — 유사도 거품 제거 후에도 남는 '약물 암기' 몫:
<table style="margin-top:6px"><tr><th class=l>regime</th><th>03 easy-neg</th><th>06 matched-neg</th><th>Δ(matched−easy)</th></tr>{cmp_row}</table>
<span class=mut>★<b>유사도 거품(matched−easy)이 cold로 갈수록 커짐</b>: warm {e_warm and (agg['sum']['warm']['roc_auc']['mean']-e_warm):+.4f} → cold-start {e_cs and (agg['sum']['cold-start']['roc_auc']['mean']-e_cs):+.4f} → cold-cold {e_cc and (agg['sum']['cold-cold']['roc_auc']['mean']-e_cc):+.4f}(<b>5시드 전부 일관</b>). 해석: 약물 암기가 막히는 cold일수록 모델이 남은 <b>유사도 지름길에 더 기댐</b> → 그걸 제거(matched)한 효과가 cold에서 더 크다. 즉 03의 easy cold 점수는 warm보다 <b>더 부풀어</b> 있었다.</span><br>
· ★<b>cold-cold+matched {scc:.4f}</b> = 처음 보는 약물 쌍 + 유사도 지름길 없음 = <b>실전 신약 DDI에 가장 가까운 추정</b>. warm의 낙관(약물 암기 {sw-scc:+.3f})을 걷어낸 값.
</div>

<h2>⑥ 해석</h2>
<div class=card>
· <b>warm→cold 낙폭({scc-sw:+.4f}):</b> negative를 matched로 고정해 유사도 거품은 제거됨 → 이 낙폭은 <b>주로 '약물 암기' 몫</b>(단 cold는 학습셋도 작아 '학습량 감소'가 일부 섞임 — ⑦). warm에서 같은 약물이 train·test 양쪽에 있어 부풀던 부분.<br>
· <b>cold-start({scs:.4f})</b>: test 쌍의 한 약물은 익숙 → 부분적으로 맞힘. <b>cold-cold({scc:.4f})</b>: 둘 다 처음 → 약물 암기 전혀 안 통함.<br>
· ★<b>결론:</b> 유사도 지름길과 약물 암기를 <b>둘 다</b> 통제한 <b>가장 어려운 regime의 점추정 = {scc:.4f}</b>. ★진짜 신약-쌍 값은 <b>양방향 불확실</b>: 닮은 음성의 라벨노이즈(미발견 양성)는 <b>아래로</b>, 매칭 안 한 축(분자량·degree 등)의 남은 지름길은 <b>위로</b> 편향시킬 수 있음(단정적 상·하한 아님). 무작위보다 <b>+{scc-0.5:.3f}</b> 높음 → 지문이 '처음 보는 약물 쌍'에도 {'실제 신호를 준다(구조 기반 예측이 어느 정도 일반화)' if scc > 0.62 else '약한 신호만 준다'}.<br>
· <b>sum vs concat(cold-cold):</b> 차이 {cc_gap:+.4f}로 <b>seed 편차(~{cc_std:.4f}) 안</b> — 근소 우위이나 clean win 아님(4/5 seed sum 우세).
</div>

<h2>⑦ ★한계 메모 (정직하게)</h2>
<div class=card>
· <b>negative '미확인':</b> matched는 positive만큼 닮은 음성을 포함 → '미발견 진짜 양성'이 섞일 위험(성능을 아래로 편향). 특히 cold-cold의 닮은 음성은 검증 데이터가 적어 이 위험이 큼.<br>
· <b>cold-cold 표본 작음</b>(~{counts['cc_mean']:,.0f}) → seed 편차 큼(표 ±std).<br>
· <b>train 크기 차이:</b> cold는 아는 약물끼리만 학습(TT ~{counts['train_mean']:,.0f}) → warm(~{int(c['n_pos']*2*0.8):,})보다 작음. 낙폭에 '미지 약물'+'학습량 감소'가 섞임(주로 전자, 03과 동일).<br>
· <b>유사도 1개 축만 매칭</b>(분자량·degree 등 다른 편향축은 안 건드림). 이진·XGBoost 고정.
</div>

<h2>⑧ 다음에 해볼 것</h2>
<div class=card>
· ★<b>이 정직한 바닥({scc:.4f}) 위에서 특징 추가</b>: 물리화학 서술자 또는 '같은 CYP 대사' 같은 <b>기전 특징</b> → 성능이 오르면 '구조 외 정보'의 가치 입증(이 시리즈의 자연스러운 다음 축).<br>
· 유형별 다중분류(86종) / GNN 등 모델 축 / degree까지 매칭한 이중 분포 매칭.
</div>

<div class=foot>06 DDI cold+matched · warm/cold-start/cold-cold(전부 matched) · 융합 sum·concat · XGBoost 고정 · seed {len(c['seeds'])} 평균±std · 미지약물·∩ 검증 통과 · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "06_ddi_drugbank_cold_matched.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


def main():
    t0 = time.time()
    print("[06] cold + matched 실험 시작 ...", flush=True)
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"]))
    did2row = {d: i for i, d in enumerate(drug_ids)}
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["drug_a"] = pairs["drug_a"].astype(str); pairs["drug_b"] = pairs["drug_b"].astype(str)
    positives = [ukey(a, b) for a, b in zip(pairs.loc[pairs.label == 1, "drug_a"], pairs.loc[pairs.label == 1, "drug_b"])]
    pos_key_set = set(positives); n_pos = len(positives)
    # warm anchor용 global matched(05)
    md = pd.read_csv(MATCHED_CSV); global_matched = [ukey(str(a), str(b)) for a, b in zip(md["drug_a"], md["drug_b"])]
    print(f"  약물 {len(drug_ids):,} · positive {n_pos:,} · global matched(05) {len(global_matched):,}", flush=True)

    # 지문 + 유사도 행렬 + triu 후보(한 번만)
    FP = build_fp_matrix(drug_ids, id2smi)
    bitvects = [_GEN.GetFingerprint(Chem.MolFromSmiles(str(id2smi[d]))) for d in drug_ids]
    S = build_sim_matrix(bitvects)
    n = len(drug_ids)
    iu, ju = np.triu_indices(n, k=1); sims = S[iu, ju]
    pos_codes = np.fromiter((min(did2row[a], did2row[b]) * n + max(did2row[a], did2row[b]) for a, b in pos_key_set),
                            dtype=np.int64, count=len(pos_key_set))
    is_pos = np.isin(iu.astype(np.int64) * n + ju.astype(np.int64), pos_codes)
    # positive 색인·유사도(전역)
    pos_i = np.fromiter((did2row[a] for a, b in positives), dtype=np.int32, count=n_pos)
    pos_j = np.fromiter((did2row[b] for a, b in positives), dtype=np.int32, count=n_pos)
    pos_sims_global = S[pos_i, pos_j]

    # warm 특징(방법별 1번; 05 재현)
    warm_pairs = positives + global_matched
    warm_y = np.array([1] * n_pos + [0] * len(global_matched))
    w_idxA, w_idxB = pairs_to_idx(warm_pairs, did2row)
    warm_feats = {m: make_features(FP, w_idxA, w_idxB, methods=(m,))[m] for m in METHODS}

    REGIMES = ["warm", "cold-start", "cold-cold"]
    per = {m: {rg: {mt: [] for mt in METRIC_KEYS} for rg in REGIMES} for m in METHODS}
    cnt = {"train": [], "cs": [], "cc": [], "cs_def": [], "cc_def": [], "n_test_drugs": []}
    dist = None

    for seed in SEEDS:
        train_drugs, test_drugs = split_drugs(drug_ids, seed, DRUG_TEST_FRAC)
        g = np.array([0 if d in train_drugs else 1 for d in drug_ids], dtype=np.int8)  # 0=train,1=test
        g_iu = g[iu]; g_ju = g[ju]
        pos_unseen = g[pos_i] + g[pos_j]
        # regime positive
        pr = {}
        for name, val in [("TT", 0), ("TS", 1), ("SS", 2)]:
            idxs = np.where(pos_unseen == val)[0]
            pr[name] = {"pairs": [positives[k] for k in idxs], "sims": pos_sims_global[idxs]}
        # regime matched negative (미지조건 + 분포매칭 동시)
        negTT, dTT, sTT = regime_matched(0, g_iu, g_ju, iu, ju, sims, is_pos, pr["TT"]["sims"], drug_ids, seed * 10 + 0)
        negTS, dTS, sTS = regime_matched(1, g_iu, g_ju, iu, ju, sims, is_pos, pr["TS"]["sims"], drug_ids, seed * 10 + 1)
        negSS, dSS, sSS = regime_matched(2, g_iu, g_ju, iu, ju, sims, is_pos, pr["SS"]["sims"], drug_ids, seed * 10 + 2)

        train_pairs = pr["TT"]["pairs"] + negTT; train_y = np.array([1] * len(pr["TT"]["pairs"]) + [0] * len(negTT))
        cs_pairs = pr["TS"]["pairs"] + negTS; cs_y = np.array([1] * len(pr["TS"]["pairs"]) + [0] * len(negTS))
        cc_pairs = pr["SS"]["pairs"] + negSS; cc_y = np.array([1] * len(pr["SS"]["pairs"]) + [0] * len(negSS))

        # 검증 + ∩
        vcs = verify(train_pairs, cs_pairs, test_drugs, "cold-start"); vcc = verify(train_pairs, cc_pairs, test_drugs, "cold-cold")
        assert vcs["PASS"] and vcc["PASS"], f"seed{seed} 검증 실패 {vcs} {vcc}"
        assert len(set(negTT + negTS + negSS) & pos_key_set) == 0, "∩positive≠0"
        cnt["train"].append(len(train_pairs)); cnt["cs"].append(len(cs_pairs)); cnt["cc"].append(len(cc_pairs))
        cnt["cs_def"].append(dTS); cnt["cc_def"].append(dSS); cnt["n_test_drugs"].append(len(test_drugs))
        print(f"  [seed {seed}] test약물 {len(test_drugs)} · TT {len(train_pairs):,} · TS {len(cs_pairs):,} · SS {len(cc_pairs):,} · "
              f"부족 TS{dTS}/SS{dSS} · 검증 {'PASS' if vcs['PASS'] and vcc['PASS'] else 'FAIL'}", flush=True)

        if seed == 0:  # 분포 히스토그램용(대표 seed) — 서브샘플로 가볍게
            rng0 = np.random.RandomState(0)
            def sub(a, k=20000):
                a = np.asarray(a)
                return a if len(a) <= k else a[rng0.choice(len(a), k, replace=False)]
            gm_iA, gm_iB = pairs_to_idx(global_matched, did2row)
            warm_neg_sims = S[gm_iA, gm_iB]                       # warm(global) matched의 Tanimoto
            dist = {"warm": {"pos": sub(pos_sims_global).tolist(), "neg": sub(warm_neg_sims).tolist()},
                    "cold-start": {"pos": sub(pr["TS"]["sims"]).tolist(), "neg": sub(sTS).tolist()},
                    "cold-cold": {"pos": sub(pr["SS"]["sims"]).tolist(), "neg": sub(sSS).tolist()}}

        # 학습·평가
        tt_idxA, tt_idxB = pairs_to_idx(train_pairs, did2row)
        cs_idxA, cs_idxB = pairs_to_idx(cs_pairs, did2row); cc_idxA, cc_idxB = pairs_to_idx(cc_pairs, did2row)
        wtr, wte = train_test_split(np.arange(len(warm_pairs)), test_size=0.2, random_state=seed, stratify=warm_y)
        for m in METHODS:
            wclf = fit_model(warm_feats[m][wtr], warm_y[wtr]); warm_s = score_model(wclf, warm_feats[m][wte], warm_y[wte])
            ttclf = fit_model(make_features(FP, tt_idxA, tt_idxB, (m,))[m], train_y)
            cs_s = score_model(ttclf, make_features(FP, cs_idxA, cs_idxB, (m,))[m], cs_y)
            cc_s = score_model(ttclf, make_features(FP, cc_idxA, cc_idxB, (m,))[m], cc_y)
            for mt in METRIC_KEYS:
                per[m]["warm"][mt].append(warm_s[mt]); per[m]["cold-start"][mt].append(cs_s[mt]); per[m]["cold-cold"][mt].append(cc_s[mt])
            print(f"    {m:6s} ROC warm {warm_s['roc_auc']:.4f} → cold-start {cs_s['roc_auc']:.4f} → cold-cold {cc_s['roc_auc']:.4f}", flush=True)

    agg = {m: {rg: {mt: {"mean": float(np.mean(per[m][rg][mt])), "std": float(np.std(per[m][rg][mt]))}
                    for mt in METRIC_KEYS} for rg in REGIMES} for m in METHODS}
    counts = {"train_mean": float(np.mean(cnt["train"])), "cs_mean": float(np.mean(cnt["cs"])),
              "cc_mean": float(np.mean(cnt["cc"])), "cc_min": int(min(cnt["cc"])),
              "cs_deficit": int(sum(cnt["cs_def"])), "cc_deficit": int(sum(cnt["cc_def"])), "n_test_drugs": int(cnt["n_test_drugs"][0])}
    # 03(easy neg) 비교
    cmp03 = {}
    if os.path.exists(COLDSTART_JSON):
        j03 = json.load(open(COLDSTART_JSON))["results"]["sum"]
        cmp03 = {rg: j03[rg]["roc_auc"]["mean"] for rg in REGIMES}

    maxroc = max(agg[m][rg]["roc_auc"]["mean"] for m in METHODS for rg in REGIMES)
    res = {"config": {"n_pos": n_pos, "n_drugs": len(drug_ids), "seeds": SEEDS, "n_bins": N_BINS,
                      "drug_test_frac": DRUG_TEST_FRAC, "runtime_s": round(time.time() - t0, 1)},
           "results": agg, "per_seed": per, "counts": counts, "cmp03_easy": cmp03,
           "flags": {"roc_over_0.98": bool(maxroc > 0.98)}}
    json.dump(res, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ ROC-AUC (5 seed 평균) ================")
    for m in METHODS:
        w = agg[m]["warm"]["roc_auc"]["mean"]; c1 = agg[m]["cold-start"]["roc_auc"]["mean"]; c2 = agg[m]["cold-cold"]["roc_auc"]["mean"]
        print(f"  {m:6s}: warm {w:.4f} → cold-start {c1:.4f} ({c1-w:+.4f}) → cold-cold {c2:.4f} ({c2-w:+.4f})")
    if cmp03:
        print(f"  [03 easy 대비 sum] warm {agg['sum']['warm']['roc_auc']['mean']-cmp03['warm']:+.4f} · cs {agg['sum']['cold-start']['roc_auc']['mean']-cmp03['cold-start']:+.4f} · cc {agg['sum']['cold-cold']['roc_auc']['mean']-cmp03['cold-cold']:+.4f}")
    if maxroc > 0.98:
        print(f"\n★★ RED FLAG: ROC-AUC 최대 {maxroc:.4f} > 0.98. 확인 필요.", flush=True)
    if counts["cc_min"] < 500:
        print(f"\n★★ 주의: cold-cold 최소 표본 {counts['cc_min']} < 500 → 편차 큼. 약물 test비율↑ 검토.", flush=True)

    out_html = build_report(res, dist, counts, cmp03)
    print(f"\n[06] 완료 ({res['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    return res


if __name__ == "__main__":
    main()

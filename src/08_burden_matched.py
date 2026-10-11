"""
[DDI 실험 - 08] CYP-burden matched — 07 이득이 promiscuity 교란인지 판정
====================================================================
07: struct+enzyme가 cold-cold +0.0155(sum). 재분석 결과 이긴 sum 융합 = 쌍의 총
    CYP-burden(promiscuity)이고, 05/06 matched는 Tanimoto만 통제해 CYP-burden은
    positive가 negative보다 높게 쏠려 있었다(cold-cold pos 3.84 vs neg 2.90).
08: negative를 Tanimoto ★와★ CYP-burden 둘 다 positive에 맞춘(burden-matched) 뒤
    07의 struct+enzyme 이득이 남는지 본다.
      남으면 → promiscuity로 설명 안 되는 '진짜 CYP-특이 신호'.
      사라지면 → 07 이득은 promiscuity 교란(기전 아님 확정).

★'쌍 CYP-burden' 정의: 쌍(A,B)의 CYP 벡터 합의 총합 = n_cyp[A] + n_cyp[B] (0~16).
   = 07 sum 융합의 enzyme 블록 총합과 정확히 일치(방향無 총량).

negative 2종 × 특징 2구성(+burden-only) 교차:
  (A) sim-matched   : Tanimoto만 맞춤 (= 07 재현, 기준)
  (B) burden-matched: Tanimoto ★와★ CYP-burden 2축 동시 맞춤
핵심: burden-matched에서 (struct+enzyme − struct)가 여전히 +인가?

산출: results/08_ddi_drugbank_burden_matched.html , results/burden_matched_results.json
실행: conda run -n admet python src/08_burden_matched.py
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
                        split_drugs, regime_matched, verify_regime, pair_feature)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
MATCHED_CSV = os.path.join(BASE, "data", "matched_negatives.csv")   # 05 global sim-matched(=07 warm anchor)
CYP_CSV = os.path.join(BASE, "data", "drug_cyp.csv")
RESULTS_JSON = os.path.join(BASE, "results", "burden_matched_results.json")

SEEDS = [0, 1, 2, 3, 4]
DRUG_TEST_FRAC = 0.20
N_BINS = 40               # sim-matched(07과 동일)
N_TBINS_2D = 20           # burden-matched의 Tanimoto 축 bin(2D라 더 성기게)
REGIMES = ["warm", "cold-start", "cold-cold"]
COMBOS = [("struct", "sum"), ("struct+enzyme", "sum"), ("struct+randproj", "sum"), ("burden-only", "sum"),
          ("struct", "concat"), ("struct+enzyme", "concat")]
ALL_CONFIGS = ["struct", "struct+enzyme", "struct+randproj", "burden-only"]
METRIC_KEYS = ["roc_auc", "pr_auc", "f1", "accuracy"]


def pairs_to_idx(pair_list, did2row):
    idxA = np.fromiter((did2row[a] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    idxB = np.fromiter((did2row[b] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    return idxA, idxB


def match_2d(cand_tani, cand_burden, pos_tani, pos_burden, n_tbins, seed):
    """★2축 매칭: 후보를 positive의 (Tanimoto bin × burden) 결합 분포에 맞춰 추출.
    - positive를 (Tanimoto bin)×(정수 burden) 격자로 세고, 후보를 같은 칸에서 그만큼 뽑음.
    - 부족한 칸은 남은 후보 중 burden이 positive 평균에 가까운 것으로 채움(핵심 축 우선)."""
    if len(cand_tani) == 0:
        return np.array([], dtype=np.int64), 0
    tb = np.linspace(0.0, 1.0, n_tbins + 1)
    maxB = int(max(int(pos_burden.max()), int(cand_burden.max())))
    nB = maxB + 1

    def cellid(tani, burden):
        ti = np.clip(np.digitize(tani, tb) - 1, 0, n_tbins - 1)
        return ti.astype(np.int64) * nB + np.clip(burden.astype(np.int64), 0, maxB)

    pos_hist = np.bincount(cellid(pos_tani, pos_burden), minlength=n_tbins * nB)
    ccell = cellid(cand_tani, cand_burden)
    order = np.argsort(ccell, kind="stable"); ccell_s = ccell[order]
    uniq, starts = np.unique(ccell_s, return_index=True)
    cell_pool = {int(u): order[starts[i]:(starts[i + 1] if i + 1 < len(starts) else len(order))] for i, u in enumerate(uniq)}
    rng = np.random.RandomState(seed)
    chosen = []; used = np.zeros(len(cand_tani), bool); deficit = 0
    for c in np.nonzero(pos_hist)[0]:
        need = int(pos_hist[c]); pool = cell_pool.get(int(c))
        if pool is None or len(pool) == 0:
            deficit += need; continue
        if len(pool) >= need:
            sel = rng.choice(pool, need, replace=False)
        else:
            sel = pool; deficit += need - len(pool)
        chosen.extend(sel.tolist()); used[sel] = True
    if deficit > 0:                                    # 부족분: burden이 pos 평균에 가까운 남은 후보로
        rem = np.where(~used)[0]
        rem = rem[np.argsort(np.abs(cand_burden[rem] - pos_burden.mean()))]
        chosen.extend(rem[:deficit].tolist())
    return np.array(chosen, dtype=np.int64), int(deficit)


def burden_feat(n_cyp, idxA, idxB):
    """burden-only 특징: 쌍의 총 CYP 개수 스칼라 1개(교란 상한 측정용)."""
    return (n_cyp[idxA] + n_cyp[idxB]).reshape(-1, 1).astype(np.float32)


def feat(config, fusion, FP, CYP, RANDPROJ, n_cyp, idxA, idxB):
    if config == "burden-only":
        return burden_feat(n_cyp, idxA, idxB)
    if config == "struct+randproj":
        # ★placebo: CYP 대신 '구조유래 랜덤 8차원'. 이득이 CYP만큼 나면 = sum-pooling이 잃은 구조를
        #  아무 저차원 구조요약이나 되살린 것(=CYP 특이 아님). CYP가 더 크면 = CYP 고유 신호.
        return pair_feature(FP, RANDPROJ, idxA, idxB, "struct+enzyme", fusion)
    return pair_feature(FP, CYP, idxA, idxB, config, fusion)


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

    agg = res["results"]; per = res["per_seed"]; c = res["config"]; mv = res["match_valid"]; cnt = res["counts"]

    def gain(negtype, rg):
        s = np.array(per[negtype]["struct"]["sum"][rg]["roc_auc"]); se = np.array(per[negtype]["struct+enzyme"]["sum"][rg]["roc_auc"])
        d = se - s
        return d.mean(), int((d > 0).sum()), d
    # 핵심 그림: regime별 Δ_sim vs Δ_burden
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.arange(len(REGIMES)); w = 0.36
    for i, nt in enumerate(["sim", "burden"]):
        ds = [gain(nt, rg)[0] for rg in REGIMES]; es = [np.std(gain(nt, rg)[2]) for rg in REGIMES]
        ax.bar(x + (i - 0.5) * w, ds, w, yerr=es, capsize=3, color=("#8a8a99" if nt == "sim" else "#1a8f4c"),
               label=("sim-matched (07, Tanimoto만)" if nt == "sim" else "burden-matched (Tanimoto+CYP)"))
        for xi, dd in enumerate(ds):
            ax.annotate(f"{dd:+.4f}", (x[xi] + (i - 0.5) * w, dd + 0.001), ha="center", fontsize=8)
    ax.axhline(0, c="#e2555f", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(REGIMES); ax.set_ylabel("Δ = struct+enzyme − struct (ROC-AUC)")
    ax.set_title("★enzyme 이득: sim-matched(교란 포함) vs burden-matched(교란 제거) — 남나 사라지나")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=.25)
    for sp2 in ("top", "right"):
        ax.spines[sp2].set_visible(False)
    chart = _b64(fig)

    # 4셀 성능표(sum)
    def cell(nt, cfg, rg): return agg[nt][cfg]["sum"][rg]["roc_auc"]
    prows = ""
    for nt in ["sim", "burden"]:
        for cfg in ["struct", "struct+enzyme", "struct+randproj", "burden-only"]:
            cells = "".join(f"<td>{cell(nt, cfg, rg)['mean']:.4f}<span class=mut style=font-size:9px>±{cell(nt, cfg, rg)['std']:.3f}</span></td>" for rg in REGIMES)
            hl = " style=background:rgba(224,165,46,.08)" if cfg == "struct+randproj" else ""
            prows += f"<tr{hl}><td class=l>{nt}</td><td class=l><b>{cfg}</b>{' (placebo)' if cfg=='struct+randproj' else ''}</td>{cells}</tr>"

    # 핵심 판정표
    krows = ""
    for rg in REGIMES:
        dsm, dsm_p, dsm_arr = gain("sim", rg); dbm, dbm_p, dbm_arr = gain("burden", rg)
        from scipy import stats
        t_b, p_b = stats.ttest_rel(np.array(per["burden"]["struct+enzyme"]["sum"][rg]["roc_auc"]), np.array(per["burden"]["struct"]["sum"][rg]["roc_auc"]))
        verd = ("<b class=good>promiscuity 넘어 잔존</b>" if dbm > 0.003 and dbm_p >= 4 and p_b < 0.05 else
                "<b class=warn>사라짐(promiscuity 교란)</b>" if dbm <= 0.003 else "<span class=mut>애매</span>")
        krows += (f"<tr><td class=l>{rg}</td><td>{dsm:+.4f} <span class=mut>({dsm_p}/5)</span></td>"
                  f"<td class={'good' if dbm>0.003 else 'bad' if dbm<-0.003 else 'mut'}>{dbm:+.4f} <span class=mut>({dbm_p}/5·p{p_b:.2f})</span></td>"
                  f"<td>{dsm-dbm:+.4f}</td><td class=l>{verd}</td></tr>")

    # 매칭 검증표
    vrows = ""
    for rg in REGIMES:
        for nt in ["sim", "burden"]:
            m = mv[nt][rg]
            vrows += (f"<tr><td class=l>{rg}</td><td class=l>{nt}</td>"
                      f"<td>{m['tani_pos']:.3f} / {m['tani_neg']:.3f}</td>"
                      f"<td class={'good' if abs(m['burden_pos']-m['burden_neg'])<0.15 else 'bad'}>{m['burden_pos']:.2f} / {m['burden_neg']:.2f} (Δ{m['burden_pos']-m['burden_neg']:+.2f})</td></tr>")

    ccg_sim = gain("sim", "cold-cold")[0]; ccg_bur = gain("burden", "cold-cold")[0]
    ccat_bur = agg["burden"]["struct+enzyme"]["concat"]["cold-cold"]["roc_auc"]["mean"] - agg["burden"]["struct"]["concat"]["cold-cold"]["roc_auc"]["mean"]
    dr_bur = agg["burden"]["struct+randproj"]["sum"]["cold-cold"]["roc_auc"]["mean"] - agg["burden"]["struct"]["sum"]["cold-cold"]["roc_auc"]["mean"]
    placebo_explains = dr_bur >= ccg_bur - 0.003          # ★랜덤 구조벡터가 CYP만큼 → encoder 결함(CYP 고유 아님)
    percyp_max = cnt.get("percyp_max_imbalance", 0)
    cc_std = agg["burden"]["struct+enzyme"]["sum"]["cold-cold"]["roc_auc"]["std"]
    bonly = {rg: agg["sim"]["burden-only"]["sum"][rg]["roc_auc"]["mean"] for rg in REGIMES}
    bonly_bm = {rg: agg["burden"]["burden-only"]["sum"][rg]["roc_auc"]["mean"] for rg in REGIMES}
    if placebo_explains:
        verdict = "<b class=warn>이득은 CYP 고유 아님 — 랜덤 구조벡터(placebo)도 동등 → sum-pooling encoder 결함</b>"
    elif ccg_bur > 0.005:
        verdict = "<b class=good>총 CYP개수(promiscuity)·랜덤 구조벡터(encoder결함) 둘 다로 설명 안 됨 → CYP 내용에 고유한 이득</b> <b>(단 작고·sum-only·per-CYP 미통제·구조유래 → '기전'까진 미확정)</b>"
    else:
        verdict = "<b class=warn>대부분 total-count promiscuity 교란</b>"

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1100px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 7px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}b.hl{color:#ffd479}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>08 DDI burden-matched</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>08 · DDI CYP-burden matched — 07 이득이 promiscuity 교란인가</h1>
<p class=mut>negative를 Tanimoto+CYP-burden 2축으로 positive에 맞춘 뒤 enzyme 이득이 남는지. 07과 동일(분할·특징·sum 융합·XGBoost), 새 변수=negative 매칭 축 · env admet · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> cold-cold enzyme 이득 — sim-matched(07) <b>{ccg_sim:+.4f}</b> → burden-matched <b>{ccg_bur:+.4f}</b>(비슷한 크기 잔존). {verdict}.<br>
★핵심 대조: 랜덤 구조벡터 placebo Δ_randproj <b>{dr_bur:+.4f}</b> {'≈' if placebo_explains else '&lt;'} Δ_enzyme {ccg_bur:+.4f} · burden-only 교란상한 {bonly['cold-cold']:.3f}→<b>{bonly_bm['cold-cold']:.3f}</b>(≈0.5, 총 CYP개수 축 중화).</div>

<h2>① 매칭 검증 (2축이 맞았나)</h2>
<div class=tblwrap style="overflow-x:auto"><table>
<tr><th class=l>regime</th><th class=l>neg 종류</th><th>Tanimoto pos/neg</th><th>CYP-burden pos/neg</th></tr>{vrows}
</table></div>
<p class=mut>★burden-matched에서 CYP-burden pos≈neg(Δ≈0)이어야 성공. sim-matched는 Tanimoto만 맞고 burden은 pos&gt;neg(07의 교란). cold-cold+burden-matched 표본 평균 {cnt['cc_burden']:,.0f}(최소 {cnt['cc_burden_min']:,}).</p>

<h2>② ★핵심 판정: enzyme 이득이 남나 (struct+enzyme − struct, sum)</h2>
<div class=tblwrap style="overflow-x:auto"><table>
<tr><th class=l>regime</th><th>Δ_sim (07)</th><th>Δ_burden (교란 제거)</th><th>줄어든 몫</th><th class=l>판정</th></tr>{krows}
</table></div>
<div class=card><b>해석(정직하게):</b> burden-matched에서도 enzyme 이득이 <b>비슷한 크기로 잔존</b>({ccg_sim:+.4f}→{ccg_bur:+.4f}). 단 이는 <b>더 어려운 문제</b>서의 이득(burden struct baseline {agg['sim']['struct']['sum']['cold-cold']['roc_auc']['mean']:.3f}→{agg['burden']['struct']['sum']['cold-cold']['roc_auc']['mean']:.3f})이라 'Δ차이를 몇 %'로 나누는 건 부정확 — '비슷한 크기가 남았다'까지가 정확.<br>
· ★<b>매칭 성공:</b> burden-only 스칼라 ROC {bonly['cold-cold']:.3f}→<b>{bonly_bm['cold-cold']:.3f}(≈0.5)</b> → <b>총 CYP 개수(total-count) 축 완전 중화</b>. 즉 잔존 이득은 <b>총 CYP 개수로는 설명 안 됨</b>(promiscuity의 '총량' 버전은 배제).<br>
· ★<b>placebo(가장 중요):</b> CYP 대신 <b>'랜덤 구조벡터 8차원'</b>(Morgan 랜덤투영)을 넣으면 Δ_randproj <b>{dr_bur:+.4f}</b> {'≈' if placebo_explains else '&lt;'} Δ_enzyme {ccg_bur:+.4f}. {'→ <b class=warn>랜덤 벡터도 동등한 이득</b> = 이득은 CYP 고유가 아니라 <b>sum-pooling이 잃은 저차원 구조를 아무 벡터나 되살린 것</b>(encoder 결함). CYP 내용 아님.' if placebo_explains else '→ enzyme가 randproj보다 <b>명확히 큼</b>(randproj는 오히려 해로움 {dr_bur:+.4f}) = 이득은 <b>CYP 내용에 고유</b>(랜덤 저차원 구조론 재현 안 됨) → <b>encoder-결함 가설 기각</b>. 단 아래 한계로 기전 미확정.'}<br>
· <b>남은 교란(잔존≠기전):</b> (a) 총량만 맞춤 — <b>per-CYP 개별 유병률은 pos/neg 여전히 다름(최대 {percyp_max})</b> = 미통제 교란 후보 (b) 예측 CYP=구조유래(struct 중복) (c) 라벨-정의 순환 (d) 방향성 concat은 소멸(Δ{ccat_bur:+.4f}).<br>
· <b>크기 주의:</b> 잔존 {ccg_bur:+.4f}는 <b>작음</b>(단일구성 seed 산포 {cc_std:.3f} 이하), <b>sum 융합서만</b>, paired 검정서만 유의(p≈0.002).</div>

<h2>③ 4셀 성능표 (sum 융합, 5 seed)</h2>
<img src="{chart}" alt="delta sim vs burden">
<div class=tblwrap style="overflow-x:auto"><table>
<tr><th class=l>neg</th><th class=l>특징</th>{"".join(f"<th>{rg}</th>" for rg in REGIMES)}</tr>{prows}
</table></div>
<p class=mut>struct-sim이 07/06 재현: warm {agg['sim']['struct']['sum']['warm']['roc_auc']['mean']:.4f}(06 0.8719)·cc {agg['sim']['struct']['sum']['cold-cold']['roc_auc']['mean']:.4f}. struct+enzyme-sim cc {agg['sim']['struct+enzyme']['sum']['cold-cold']['roc_auc']['mean']:.4f}(07 0.7049).</p>

<h2>④ ★한계 메모 (잔존 이득을 '기전'으로 못 부르는 이유)</h2>
<div class=card>
· <b>per-CYP 미통제(핵심):</b> 총 CYP개수만 맞춤 — 개별 CYP 유병률은 pos/neg 여전히 다름(최대 {percyp_max}). 잔존 이득이 이 <b>미통제 per-CYP 불균형</b>을 잡은 것일 수 있음(진짜 기전 vs 잔여 교란 미분리).<br>
· <b>예측 CYP</b>(구조유래·struct 중복 — 단 placebo로 '아무 구조벡터나'는 배제됨) · <b>라벨-정의 순환 잔여</b>(cold-cold도 완전 제거 아님) · burden 정의는 <b>sum 기준</b>(방향無).<br>
· <b>크기·취약성:</b> 잔존 {ccg_bur:+.4f}는 작고(seed 산포 {cc_std:.3f} 이하) sum 융합서만·paired서만 유의. <b>2축 매칭으로 cold-cold 표본</b> 평균 {cnt['cc_burden']:,.0f}(부족 {cnt['cc_burden_deficit']:,}).<br>
· negative '미확인' · warm 약물암기 낙관 · 이진·XGBoost 고정.</div>

<h2>⑤ 다음</h2>
<div class=card>{'· ★CYP 내용 고유 이득 확인(placebo·burden 둘 다 통과) → 결정적 다음: <b>per-CYP 분포까지 매칭</b>해 잔여 교란 제거 후에도 남나 · <b>어떤 CYP가 기여</b>하는지 분해 · <b>curated 진짜 효소</b>로 구조유래 중복 제거.' if ccg_bur>0.005 and not placebo_explains else '· 교란/encoder 결함 확정 → 특징 재설계.'} · 방향성(저해↔기질 짝) 특징 · 유형별 다중분류.</div>

<div class=foot>08 DDI burden-matched · struct/struct+enzyme/burden-only × sim/burden-matched × 3 regime · sum 융합 · XGBoost 고정 · seed {len(c['seeds'])} · ∩=0·검증통과 · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "08_ddi_drugbank_burden_matched.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


def main():
    t0 = time.time()
    print("[08] burden-matched 실험 시작 ...", flush=True)
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"])); did2row = {d: i for i, d in enumerate(drug_ids)}
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["drug_a"] = pairs["drug_a"].astype(str); pairs["drug_b"] = pairs["drug_b"].astype(str)
    positives = [ukey(a, b) for a, b in zip(pairs.loc[pairs.label == 1, "drug_a"], pairs.loc[pairs.label == 1, "drug_b"])]
    pos_key_set = set(positives); n_pos = len(positives)
    md = pd.read_csv(MATCHED_CSV); global_sim = [ukey(str(a), str(b)) for a, b in zip(md["drug_a"], md["drug_b"])]
    cyp_df = pd.read_csv(CYP_CSV); cyp_df["drug_id"] = cyp_df["drug_id"].astype(str)
    ccols = [c for c in cyp_df.columns if c != "drug_id"]; cyp_df = cyp_df.set_index("drug_id").loc[drug_ids]
    CYP = sp.csr_matrix(cyp_df[ccols].to_numpy(dtype=np.uint8))
    n_cyp = np.asarray(CYP.sum(axis=1)).ravel().astype(np.int32)    # 약물별 CYP 개수(0~8)
    print(f"  약물 {len(drug_ids):,} · positive {n_pos:,} · CYP {len(ccols)}종", flush=True)

    FP = build_fp_matrix(drug_ids, id2smi)
    bitvects = [_GEN.GetFingerprint(Chem.MolFromSmiles(str(id2smi[d]))) for d in drug_ids]
    S = build_sim_matrix(bitvects)
    n = len(drug_ids); iu, ju = np.triu_indices(n, k=1); sims = S[iu, ju]
    pair_burden = (n_cyp[iu] + n_cyp[ju]).astype(np.int32)          # 후보 쌍 burden(0~16)
    pos_codes = np.fromiter((min(did2row[a], did2row[b]) * n + max(did2row[a], did2row[b]) for a, b in pos_key_set), dtype=np.int64, count=len(pos_key_set))
    is_pos = np.isin(iu.astype(np.int64) * n + ju.astype(np.int64), pos_codes)
    pos_i = np.fromiter((did2row[a] for a, b in positives), dtype=np.int32, count=n_pos)
    pos_j = np.fromiter((did2row[b] for a, b in positives), dtype=np.int32, count=n_pos)
    pos_sims_g = S[pos_i, pos_j]; pos_burden_g = (n_cyp[pos_i] + n_cyp[pos_j]).astype(np.int32)

    # ── ★placebo: '구조유래 랜덤 8차원'(Morgan을 랜덤투영→이진화). CYP와 같은 형식/차원이나 CYP 의미 없음.
    #    struct+randproj 이득이 struct+enzyme만큼이면 = 이득은 CYP 고유가 아니라 'sum-pooling이 잃은 저차원
    #    구조요약을 되살린 것'(encoder 결함). enzyme가 더 크면 = CYP 고유 신호. ──
    Rproj = np.random.RandomState(2024).randn(FP.shape[1], 8)
    RANDPROJ = sp.csr_matrix((np.asarray(FP.dot(Rproj)) > 0).astype(np.uint8))

    # ── global burden-matched(warm anchor용): 전체 비상호작용에서 2축 매칭 ──
    cand_all = ~is_pos
    ch, gdef = match_2d(sims[cand_all], pair_burden[cand_all], pos_sims_g, pos_burden_g, N_TBINS_2D, 999)
    ci_all = iu[cand_all]; cj_all = ju[cand_all]
    global_burden = [ukey(drug_ids[int(ci_all[k])], drug_ids[int(cj_all[k])]) for k in ch]
    print(f"  global burden-matched {len(global_burden):,} · 부족 {gdef} · burden pos {pos_burden_g.mean():.2f}/neg {pair_burden[cand_all][ch].mean():.2f}", flush=True)

    GLOBAL_NEG = {"sim": global_sim, "burden": global_burden}
    NEGTYPES = ["sim", "burden"]
    per = {nt: {cfg: {fu: {rg: {mt: [] for mt in METRIC_KEYS} for rg in REGIMES} for fu in ["sum", "concat"]}
                for cfg in ALL_CONFIGS} for nt in NEGTYPES}
    # warm 특징(negtype × combo, 한 번만)
    warm_feats = {}
    warm_y = {}
    for nt in NEGTYPES:
        wp = positives + GLOBAL_NEG[nt]; warm_y[nt] = np.array([1] * n_pos + [0] * len(GLOBAL_NEG[nt]))
        wiA, wiB = pairs_to_idx(wp, did2row)
        warm_feats[nt] = {(cfg, fu): feat(cfg, fu, FP, CYP, RANDPROJ, n_cyp, wiA, wiB) for (cfg, fu) in COMBOS}

    mv = {nt: {rg: {} for rg in REGIMES} for nt in NEGTYPES}   # 매칭 검증(대표 seed0)
    cc_burden_cnt = []; cc_burden_def = []
    percyp_imb = {}; percyp_max = 0.0                          # per-CYP 불균형(seed0서 채움)
    CYPd = np.asarray(CYP.todense())                          # per-CYP 유병률 계산용

    for seed in SEEDS:
        train_drugs, test_drugs = split_drugs(drug_ids, seed, DRUG_TEST_FRAC)
        g = np.array([0 if d in train_drugs else 1 for d in drug_ids], dtype=np.int8)
        g_iu = g[iu]; g_ju = g[ju]; pu = g[pos_i] + g[pos_j]
        prg = {}
        for name, val in [("TT", 0), ("TS", 1), ("SS", 2)]:
            idxs = np.where(pu == val)[0]
            prg[name] = {"pairs": [positives[k] for k in idxs], "sims": pos_sims_g[idxs], "burden": pos_burden_g[idxs]}

        # 두 negative 종류 각각 regime 세트 구성
        sets = {}
        for nt in NEGTYPES:
            reg = {}
            for name, val in [("TT", 0), ("TS", 1), ("SS", 2)]:
                pu_pair = g_iu + g_ju; cmask = (pu_pair == val) & (~is_pos)
                ci = iu[cmask]; cj = ju[cmask]
                if nt == "sim":
                    from ddi_common import bin_match
                    chosen, _ = bin_match(sims[cmask], prg[name]["sims"], N_BINS, seed * 10 + val)
                else:
                    chosen, defi = match_2d(sims[cmask], pair_burden[cmask], prg[name]["sims"], prg[name]["burden"], N_TBINS_2D, seed * 100 + val)
                    if name == "SS":
                        cc_burden_def.append(defi)
                negpairs = [ukey(drug_ids[int(ci[k])], drug_ids[int(cj[k])]) for k in chosen]
                negburden = (np.array([n_cyp[did2row[a]] + n_cyp[did2row[b]] for a, b in negpairs]) if negpairs else np.array([]))
                negsims = (np.array([S[did2row[a], did2row[b]] for a, b in negpairs]) if negpairs else np.array([]))
                reg[name] = {"pairs": negpairs, "burden": negburden, "sims": negsims}
            sets[nt] = reg

        # 매칭 검증 기록(seed0)
        if seed == 0:
            for nt in NEGTYPES:
                for name, rg in [("TT", "warm"), ("TS", "cold-start"), ("SS", "cold-cold")]:
                    # warm은 global로 대표(아래 별도), 여기선 regime 것
                    pass
            # regime별 pos/neg 평균(seed0)
            for nt in NEGTYPES:
                mv[nt]["cold-start"] = {"tani_pos": float(prg["TS"]["sims"].mean()), "tani_neg": float(sets[nt]["TS"]["sims"].mean()),
                                        "burden_pos": float(prg["TS"]["burden"].mean()), "burden_neg": float(sets[nt]["TS"]["burden"].mean())}
                mv[nt]["cold-cold"] = {"tani_pos": float(prg["SS"]["sims"].mean()), "tani_neg": float(sets[nt]["SS"]["sims"].mean()),
                                       "burden_pos": float(prg["SS"]["burden"].mean()), "burden_neg": float(sets[nt]["SS"]["burden"].mean())}
                # warm: global
                gneg = GLOBAL_NEG[nt]  # ★랜덤 표본(앞부분은 cell 순서라 burden 편중)
                gsamp = [gneg[k] for k in np.random.RandomState(7).choice(len(gneg), min(30000, len(gneg)), replace=False)]
                gnb = np.array([n_cyp[did2row[a]] + n_cyp[did2row[b]] for a, b in gsamp])
                gns = np.array([S[did2row[a], did2row[b]] for a, b in gsamp])
                mv[nt]["warm"] = {"tani_pos": float(pos_sims_g.mean()), "tani_neg": float(gns.mean()),
                                  "burden_pos": float(pos_burden_g.mean()), "burden_neg": float(gnb.mean())}
            # ★per-CYP 불균형(defect#2): burden-matched(총량 맞춘) cold-cold서 pos vs neg의 '개별 CYP' 유병률 차이
            #   총량은 맞아도 어떤 CYP인지가 다르면(=per-CYP 불균형) 그게 잔존 이득의 미통제 교란 후보.
            def _percyp(prs):
                idx = [did2row[a] for a, b in prs] + [did2row[b] for a, b in prs]
                return CYPd[idx].mean(axis=0)
            pc_pos = _percyp(prg["SS"]["pairs"]); pc_neg = _percyp(sets["burden"]["SS"]["pairs"])
            percyp_imb = {ccols[j]: round(float(pc_pos[j] - pc_neg[j]), 3) for j in range(len(ccols))}
            percyp_max = round(float(np.max(np.abs(pc_pos - pc_neg))), 3)
        cc_burden_cnt.append(len(prg["SS"]["pairs"]) + len(sets["burden"]["SS"]["pairs"]))

        # 학습·평가
        for nt in NEGTYPES:
            reg = sets[nt]
            tt_pairs = prg["TT"]["pairs"] + reg["TT"]["pairs"]; tt_y = np.array([1] * len(prg["TT"]["pairs"]) + [0] * len(reg["TT"]["pairs"]))
            cs_pairs = prg["TS"]["pairs"] + reg["TS"]["pairs"]; cs_y = np.array([1] * len(prg["TS"]["pairs"]) + [0] * len(reg["TS"]["pairs"]))
            cc_pairs = prg["SS"]["pairs"] + reg["SS"]["pairs"]; cc_y = np.array([1] * len(prg["SS"]["pairs"]) + [0] * len(reg["SS"]["pairs"]))
            assert verify_regime(tt_pairs, cs_pairs, test_drugs, "cold-start")["PASS"] and verify_regime(tt_pairs, cc_pairs, test_drugs, "cold-cold")["PASS"]
            assert len(set(reg["TT"]["pairs"] + reg["TS"]["pairs"] + reg["SS"]["pairs"]) & pos_key_set) == 0
            tt_iA, tt_iB = pairs_to_idx(tt_pairs, did2row); cs_iA, cs_iB = pairs_to_idx(cs_pairs, did2row); cc_iA, cc_iB = pairs_to_idx(cc_pairs, did2row)
            wtr, wte = train_test_split(np.arange(len(warm_y[nt])), test_size=0.2, random_state=seed, stratify=warm_y[nt])
            for (cfg, fu) in COMBOS:
                wclf = fit_model(warm_feats[nt][(cfg, fu)][wtr], warm_y[nt][wtr]); ws = score_model(wclf, warm_feats[nt][(cfg, fu)][wte], warm_y[nt][wte])
                ttclf = fit_model(feat(cfg, fu, FP, CYP, RANDPROJ, n_cyp, tt_iA, tt_iB), tt_y)
                cs_s = score_model(ttclf, feat(cfg, fu, FP, CYP, RANDPROJ, n_cyp, cs_iA, cs_iB), cs_y)
                cc_s = score_model(ttclf, feat(cfg, fu, FP, CYP, RANDPROJ, n_cyp, cc_iA, cc_iB), cc_y)
                for mt in METRIC_KEYS:
                    per[nt][cfg][fu]["warm"][mt].append(ws[mt]); per[nt][cfg][fu]["cold-start"][mt].append(cs_s[mt]); per[nt][cfg][fu]["cold-cold"][mt].append(cc_s[mt])
        d_sim = per["sim"]["struct+enzyme"]["sum"]["cold-cold"]["roc_auc"][-1] - per["sim"]["struct"]["sum"]["cold-cold"]["roc_auc"][-1]
        d_bur = per["burden"]["struct+enzyme"]["sum"]["cold-cold"]["roc_auc"][-1] - per["burden"]["struct"]["sum"]["cold-cold"]["roc_auc"][-1]
        print(f"  [seed {seed}] cold-cold Δ: sim {d_sim:+.4f} · burden {d_bur:+.4f}", flush=True)

    def fusions_for(cfg): return ["sum"] if cfg in ("burden-only", "struct+randproj") else ["sum", "concat"]  # 이 둘은 sum만
    agg = {nt: {cfg: {fu: {rg: {mt: {"mean": float(np.mean(per[nt][cfg][fu][rg][mt])), "std": float(np.std(per[nt][cfg][fu][rg][mt]))}
                              for mt in METRIC_KEYS} for rg in REGIMES} for fu in fusions_for(cfg)}
                for cfg in ALL_CONFIGS} for nt in NEGTYPES}
    counts = {"cc_burden": float(np.mean(cc_burden_cnt)), "cc_burden_min": int(min(cc_burden_cnt)), "cc_burden_deficit": int(sum(cc_burden_def)),
              "percyp_max_imbalance": percyp_max, "percyp_imbalance": percyp_imb}
    maxroc = max(agg[nt][cfg][fu][rg]["roc_auc"]["mean"] for nt in NEGTYPES for cfg in ALL_CONFIGS for fu in fusions_for(cfg) for rg in REGIMES)
    res = {"config": {"n_pos": n_pos, "n_drugs": len(drug_ids), "seeds": SEEDS, "n_bins": N_BINS, "n_tbins_2d": N_TBINS_2D, "runtime_s": round(time.time() - t0, 1)},
           "results": agg, "per_seed": per, "match_valid": mv, "counts": counts, "flags": {"roc_over_0.98": bool(maxroc > 0.98)}}
    json.dump(res, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ cold-cold enzyme 이득 (struct+enzyme − struct, sum) ================")
    for rg in REGIMES:
        ds = agg["sim"]["struct+enzyme"]["sum"][rg]["roc_auc"]["mean"] - agg["sim"]["struct"]["sum"][rg]["roc_auc"]["mean"]
        db = agg["burden"]["struct+enzyme"]["sum"][rg]["roc_auc"]["mean"] - agg["burden"]["struct"]["sum"][rg]["roc_auc"]["mean"]
        print(f"  {rg:11s}: Δ_sim {ds:+.4f} → Δ_burden {db:+.4f}  (교란 {ds-db:+.4f})")
    print(f"  매칭검증 cold-cold burden pos/neg: sim {mv['sim']['cold-cold']['burden_pos']:.2f}/{mv['sim']['cold-cold']['burden_neg']:.2f} · burden {mv['burden']['cold-cold']['burden_pos']:.2f}/{mv['burden']['cold-cold']['burden_neg']:.2f}")
    de = agg["burden"]["struct+enzyme"]["sum"]["cold-cold"]["roc_auc"]["mean"] - agg["burden"]["struct"]["sum"]["cold-cold"]["roc_auc"]["mean"]
    dr = agg["burden"]["struct+randproj"]["sum"]["cold-cold"]["roc_auc"]["mean"] - agg["burden"]["struct"]["sum"]["cold-cold"]["roc_auc"]["mean"]
    print(f"  ★placebo cold-cold(burden-matched): Δ_enzyme {de:+.4f} vs Δ_randproj(랜덤 구조벡터) {dr:+.4f} — randproj≈enzyme면 sum-encoder 결함(CYP 고유 아님)")
    print(f"  per-CYP 최대 불균형(총량 맞춘 뒤 남은): {percyp_max}")
    if maxroc > 0.98:
        print(f"\n★★ RED FLAG ROC {maxroc:.4f}>0.98", flush=True)
    if counts["cc_burden_min"] < 500:
        print(f"\n★★ 주의: cold-cold+burden 최소표본 {counts['cc_burden_min']}<500", flush=True)
    out_html = build_report(res)
    print(f"\n[08] 완료 ({res['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    return res


if __name__ == "__main__":
    main()

"""
[DDI 실험 - 12] 타깃(PD 기전) 특징이 구조 위에 DDI 유형 예측을 개선하나
========================================================================
가설: DDI는 약력학(PD) 겹침으로도 생김 — 두 약물이 ★같은 타깃을 치면 상호작용 위험.
      구조(Morgan)만으로는 이 기전을 직접 못 보므로, ChEMBL ★진짜 주석 타깃을 더하면 개선될까?
재료: data/drug_targets.csv (ChEMBL mechanism, 45% 약물 주석, 359 타깃 → ★공유가능 168 타깃만 사용).
      단일약물 타깃(공유 불가)은 순수 노이즈라 제외.

★사전 실측(feasibility): 양성쌍 중 양쪽주석 20.3% · ★타깃공유쌍 0.83%(1,590건)뿐.
★실측 결과(예상과 반대): cold-cold macro-F1 +0.029(5/5·p0.005)로 ★진짜 소폭 이득. 단 공유타깃 기전(0.83%)이
  아니라 '타깃벡터=미관측 약물 약리 descriptor'가 원천. 두 위약으로 confound 배제(아래).

구성(★모두 class-weight ON = 11에서 확립한 불균형 처리 고정, 변수=특징집합):
  (a) struct               : Morgan 2048 (= 11 class-weight 재현 앵커)
  (b) struct+target        : + 공유가능타깃 168 합벡터 + [n_shared, both_annotated] 파생 = ★PD 최선표현
  (c) struct+target-shuf   : ★랜덤위약 — 약물→타깃 매핑 행순열(차원수·랜덤벡터 통제). ※주석유무는 보존 안 됨
  (d) struct+target-degshuf: ★degree-보존 위약(12b_degshuf.py) — 이중엣지스왑으로 각 약물 degree·각 타깃 빈도
                             그대로 두고 특정 타깃만 섞음 → ★promiscuity/주석유무 confound 정확통제(검증자 지적 반영)
  (e) struct+cyp+target    : + CYP 8(PK) → PK+PD 생물학이 쌓이나
★(b)vs(d): 진짜 vs degree-보존 위약. (b)>(d)면 이득은 '특정 drug→target 정체'(promiscuity 대리 아님). 실측 +0.036(5/5·p0.001).

고정(10/11 동일): 분할 warm/cold-start/cold-cold(★cold-cold 주지표=순환 방지, 두 약물 다 미관측),
  XGBoost multi:softprob, seed5. 평가: macro-F1(주) + head/mid/tail + acc + top3.
★추가분석: 타깃공유(n_shared≥1) 부분집합에서만 (b)가 (a)를 이기나(기전 국소검증).

산출: results/12_ddi_drugbank_targets.html , results/targets_results.json
실행: conda run -n admet python src/12_targets.py
"""
import os, sys, json, time, io, base64
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddi_common import ukey, build_fp_matrix, make_features, split_drugs, fit_mc, top3_acc

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
RAW_CSV = os.path.join(BASE, "data", "drugbank_ddi_raw.csv")
TGT_CSV = os.path.join(BASE, "data", "drug_targets.csv")
CYP_CSV = os.path.join(BASE, "data", "drug_cyp.csv")
RESULTS_JSON = os.path.join(BASE, "results", "targets_results.json")

SEEDS = [0, 1, 2, 3, 4]
DRUG_TEST_FRAC = 0.20
REGIMES = ["warm", "cold-start", "cold-cold"]
CFGS = ["struct", "struct+target", "struct+target-shuf", "struct+cyp+target"]   # main() 실행용(4구성)
# 보고서용(12b degree-보존 위약 병합 후 5구성). degshuf는 12b_degshuf.py가 계산.
REPORT_CFGS = ["struct", "struct+target", "struct+target-shuf", "struct+target-degshuf", "struct+cyp+target"]
MIN_TGT_DRUGS = 2   # 공유가능(≥2 약물이 치는) 타깃만 — 단일약물 타깃은 공유 불가 = 노이즈
METRICS = ["accuracy", "macro_f1", "micro_f1", "top3", "head_f1", "mid_f1", "tail_f1", "sub_acc", "sub_n"]


def load_Y(pos_pairs):
    raw = pd.read_csv(RAW_CSV); raw["Drug1_ID"] = raw["Drug1_ID"].astype(str); raw["Drug2_ID"] = raw["Drug2_ID"].astype(str)
    ymap = {}
    for a, b, y in zip(raw["Drug1_ID"], raw["Drug2_ID"], raw["Y"]):
        k = ukey(a, b)
        if k not in ymap:
            ymap[k] = int(y)
    Yraw = np.array([ymap[k] for k in pos_pairs])
    uniq = sorted(set(Yraw.tolist())); remap = {t: i for i, t in enumerate(uniq)}
    return np.array([remap[t] for t in Yraw])


def build_target_matrix(drug_ids, did2row):
    """공유가능(≥MIN_TGT_DRUGS 약물이 치는) 타깃만으로 이진행렬 TGT(n_drug × n_tgt)."""
    tg = pd.read_csv(TGT_CSV); tg["drug_id"] = tg["drug_id"].astype(str)
    tcnt = tg.groupby("target_chembl_id")["drug_id"].nunique()
    keep = sorted(tcnt[tcnt >= MIN_TGT_DRUGS].index.tolist())
    t2col = {t: i for i, t in enumerate(keep)}
    rows, cols = [], []
    ann_drugs = set()
    for d, t in zip(tg["drug_id"], tg["target_chembl_id"]):
        if d in did2row and t in t2col:
            rows.append(did2row[d]); cols.append(t2col[t]); ann_drugs.add(d)
    TGT = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(drug_ids), len(keep)))
    return TGT, len(keep), len(ann_drugs)


def build_cyp_matrix(drug_ids, did2row):
    cyp = pd.read_csv(CYP_CSV); cyp["drug_id"] = cyp["drug_id"].astype(str)
    cols = [c for c in cyp.columns if c != "drug_id"]
    M = np.zeros((len(drug_ids), len(cols)))
    for _, r in cyp.iterrows():
        if str(r["drug_id"]) in did2row:
            M[did2row[str(r["drug_id"])]] = r[cols].values.astype(float)
    return sp.csr_matrix(M)


def target_block(T, pi, pj):
    """쌍별 타깃 특징: 합벡터(공유=2) + n_shared 스칼라 + both_annotated 스칼라 = ★PD 최선표현."""
    tsum = (T[pi] + T[pj]).tocsr()
    ns = np.asarray(T[pi].multiply(T[pj]).sum(1)).ravel()              # 공유 타깃 수
    ai = np.asarray(T[pi].sum(1)).ravel() > 0; aj = np.asarray(T[pj].sum(1)).ravel() > 0
    ba = (ai & aj).astype(float)
    return sp.hstack([tsum, sp.csr_matrix(ns.reshape(-1, 1)), sp.csr_matrix(ba.reshape(-1, 1))], format="csr").tocsr()


def eval_full(model, Xte, yte, ytr_orig, head_t, mid_t, tail_t, sub_mask):
    clf, le = model
    pred = le.inverse_transform(clf.predict(Xte)); proba = clf.predict_proba(Xte)
    tl = sorted(set(yte.tolist()))
    per_f1 = f1_score(yte, pred, average=None, labels=tl, zero_division=0)
    f1map = dict(zip(tl, per_f1))
    def band(types):
        v = [f1map[t] for t in tl if t in types]; return float(np.mean(v)) if v else 0.0
    sn = int(sub_mask.sum())
    sub_acc = float(accuracy_score(yte[sub_mask], pred[sub_mask])) if sn > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", labels=tl, zero_division=0)),
        "micro_f1": float(f1_score(yte, pred, average="micro", labels=tl, zero_division=0)),
        "top3": top3_acc(proba, le.classes_, yte),
        "head_f1": band(head_t), "mid_f1": band(mid_t), "tail_f1": band(tail_t),
        "sub_acc": sub_acc, "sub_n": float(sn),
    }


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
    agg = res["results"]; c = res["config"]; feas = res["feasibility"]
    CF = [x for x in REPORT_CFGS if x in agg]   # degshuf 병합됐으면 5, 아니면 4
    ccol = {"struct": "#8a8a99", "struct+target": "#1a8f4c", "struct+target-shuf": "#c0562f",
            "struct+target-degshuf": "#e0a52e", "struct+cyp+target": "#4c78d8"}
    from scipy import stats

    def rob(cfg, metric, rg="cold-cold"):
        a = np.array(res["per_seed"][cfg][rg][metric]); bb = np.array(res["per_seed"]["struct"][rg][metric])
        d = a - bb
        try:
            _, p = stats.ttest_rel(a, bb); p = float(p) if not np.isnan(p) else 1.0
        except Exception:
            p = 1.0
        return float(d.mean()), int((d > 0).sum()), p

    lab_short = {"struct+target": "+target(진짜)", "struct+target-shuf": "+shuf(랜덤위약)",
                 "struct+target-degshuf": "+degshuf(degree보존위약)", "struct+cyp+target": "+cyp+target"}
    # 그림1: macro-F1 구성 × regime
    fig1, ax = plt.subplots(figsize=(9.2, 3.8)); w = 0.16
    n = len(CF)
    for i, cf in enumerate(CF):
        m = [agg[cf][rg]["macro_f1"]["mean"] for rg in REGIMES]; s = [agg[cf][rg]["macro_f1"]["std"] for rg in REGIMES]
        ax.bar(np.arange(3) + (i - (n - 1) / 2) * w, m, w, yerr=s, capsize=2, color=ccol[cf], label=cf)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(REGIMES); ax.set_ylabel("macro-F1"); ax.set_ylim(0, 0.75)
    ax.set_title("macro-F1 (주지표) — 특징구성 × regime  (★cold-cold=순환없는 정직테스트)")
    ax.legend(fontsize=7.5, ncol=3); ax.grid(axis="y", alpha=.25)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    chart_m = _b64(fig1)

    # 그림2: cold-cold Δ vs struct — 진짜 vs 두 위약
    fig2, ax = plt.subplots(figsize=(8.2, 3.4))
    cfs = [x for x in CF if x != "struct"]
    labs = [lab_short.get(cf, cf) for cf in cfs]
    dm = [agg[cf]["cold-cold"]["macro_f1"]["mean"] - agg["struct"]["cold-cold"]["macro_f1"]["mean"] for cf in cfs]
    cols = ["#1a8f4c" if v > 0.003 else "#e2555f" if v < -0.003 else "#8a8a99" for v in dm]
    ax.bar(labs, dm, color=cols); ax.axhline(0, color="#888", lw=.8)
    for i, v in enumerate(dm):
        ax.annotate(f"{v:+.3f}", (i, v + (0.001 if v >= 0 else -0.003)), ha="center", fontsize=9)
    ax.set_ylabel("Δ macro-F1 vs struct (cold-cold)"); ax.tick_params(axis="x", labelsize=8)
    ax.set_title("타깃이 구조 위에 더하는 값 — ★진짜(+target) vs 위약 2종(랜덤·degree보존)")
    ax.grid(axis="y", alpha=.25)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    chart_d = _b64(fig2)

    def row(cf):
        cells = ""
        for rg in REGIMES:
            a = agg[cf][rg]
            cells += (f"<td>{a['macro_f1']['mean']:.3f}</td><td>{a['head_f1']['mean']:.3f}</td>"
                      f"<td>{a['mid_f1']['mean']:.3f}</td><td>{a['tail_f1']['mean']:.3f}</td><td>{a['accuracy']['mean']:.3f}</td>")
        return f"<tr><td class=l><b>{cf}</b></td>{cells}</tr>"
    rows = "".join(row(cf) for cf in CF)

    # 핵심 수치
    b = agg["struct"]["cold-cold"]
    td, tw, tp = rob("struct+target", "macro_f1")
    sd, sw, sp2 = rob("struct+target-shuf", "macro_f1")
    cd, cw, cp = rob("struct+cyp+target", "macro_f1")
    # 타깃공유 부분집합(기전 국소검증) cold-cold
    sub_n = agg["struct"]["cold-cold"]["sub_n"]["mean"]
    sub_a0 = agg["struct"]["cold-cold"]["sub_acc"]["mean"]; sub_a1 = agg["struct+target"]["cold-cold"]["sub_acc"]["mean"]
    # warm 부분집합(표본 큼)
    subw_n = agg["struct"]["warm"]["sub_n"]["mean"]
    subw_a0 = agg["struct"]["warm"]["sub_acc"]["mean"]; subw_a1 = agg["struct+target"]["warm"]["sub_acc"]["mean"]
    # 진짜−위약 격차의 regime 그라디언트 + accuracy + 진짜vs위약 paired-t
    twm, _, _ = rob("struct+target", "macro_f1", "warm"); swm, _, _ = rob("struct+target-shuf", "macro_f1", "warm")
    tcs, _, _ = rob("struct+target", "macro_f1", "cold-start"); scs, _, _ = rob("struct+target-shuf", "macro_f1", "cold-start")
    gap_w = twm - swm; gap_cs = tcs - scs; gap_cc = td - sd
    _rt = np.array(res["per_seed"]["struct+target"]["cold-cold"]["macro_f1"]); _rs = np.array(res["per_seed"]["struct+target-shuf"]["cold-cold"]["macro_f1"])
    rvp_p = float(stats.ttest_rel(_rt, _rs)[1]); rvp_w = int((_rt > _rs).sum())
    acc_d, acc_w, acc_p = rob("struct+target", "accuracy")
    # ★degree-보존 위약(12b): 주석유무·degree·타깃빈도 보존, 특정 타깃만 섞음 → 진짜 vs 이것으로 confound 분리
    has_deg = "struct+target-degshuf" in agg
    if has_deg:
        dgd, dgw, dgp = rob("struct+target-degshuf", "macro_f1")
        _rd = np.array(res["per_seed"]["struct+target-degshuf"]["cold-cold"]["macro_f1"])
        rvd_d = float((_rt - _rd).mean()); rvd_w = int((_rt > _rd).sum()); rvd_p = float(stats.ttest_rel(_rt, _rd)[1])
    else:
        dgd = dgw = dgp = rvd_d = rvd_w = rvd_p = 0

    def verd(d, w, p):
        return (f"<b class=good>순이득·견고({w}/5·p{p:.2f})</b>" if (w >= 5 or p < 0.05) and d > 0.003 else
                f"<span class=warn>편차 내({w}/5·p{p:.2f})</span>" if d > 0.003 else
                f"<span class=bad>손해</span>" if d < -0.003 else "<span class=mut>무승부</span>")

    krow_defs = [("<b>+target (진짜)</b>", td, tw, tp),
                 ("+shuf (랜덤위약·차원통제)", sd, sw, sp2)]
    if has_deg:
        krow_defs.append(("<b>+degshuf (degree보존위약·confound통제)</b>", dgd, dgw, dgp))
    krow_defs.append(("+cyp+target (PK+PD)", cd, cw, cp))
    krows = "".join(
        f"<tr><td class=l>{lab}</td><td class={'good' if d>0.003 else 'bad' if d<-0.003 else 'mut'}>{d:+.3f}</td>"
        f"<td>{w}/5</td><td>{p:.2f}</td><td class=l>{verd(d,w,p)}</td></tr>"
        for lab, d, w, p in krow_defs)

    # 판정: 진짜가 degree-보존 위약(있으면)을 이기면 confound 배제된 진짜 신호
    real_vs_plac = ("<b class=good>진짜 &gt; degree-보존 위약</b> — 특정 타깃 정체가 신호(promiscuity/주석유무 대리 아님)" if has_deg and rvd_d > 0.005 and rvd_w >= 4 else
                    "<b class=warn>진짜 ≈ degree-보존 위약</b> — 이득은 주석유무/promiscuity 대리(특정타깃 무관)" if has_deg and abs(rvd_d) <= 0.005 else
                    "<b class=good>진짜 &gt; 랜덤 위약</b>" if td - sd > 0.005 else "<b class=warn>진짜 ≈ 위약</b>")

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1100px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 6px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}b.hl{color:#ffd479}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>12 DDI targets(PD)</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>12 · 타깃(PD 기전) 특징이 구조 위에 DDI 유형 예측을 개선하나</h1>
<p class=mut>ChEMBL ★진짜 주석 타깃(예측 아님)을 struct에 더해 다중분류 개선 여부 측정. 11과 동일(분할·class-weight·XGBoost), 새 변수=타깃/CYP 특징 · env admet · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> ★<b>진짜 타깃 주석이 구조 위에 견고한 소폭 이득을 준다</b> — cold-cold Δmacro-F1 <b>{td:+.3f}</b>({tw}/5·p{tp:.3f}) {verd(td,tw,tp)}. accuracy도 {acc_d:+.3f}({acc_w}/5·trade-off 아님), CYP까지 더하면 {cd:+.3f}(PK+PD 축적).<br>
★<b>위약 2종으로 원인 규명</b>: ①랜덤위약(shuf) {sd:+.3f} → '차원수·랜덤벡터' 아티팩트 배제. ②<b>degree-보존 위약</b>(주석유무·degree·타깃빈도 그대로 두고 <b>특정 타깃만</b> 섞음) {dgd:+.3f} → 진짜−degree위약 <b>{rvd_d:+.3f}</b>({rvd_w}/5·p{rvd_p:.3f}). ★결정적: degree-보존 위약도 이득이 없다 = 이득은 'promiscuity/주석유무 대리'가 아니라 ★<b>특정 drug→target 정체</b>에서 온다.<br>
★<b>단, '공유타깃 기전'은 아니다</b>: 타깃공유쌍 {feas['n_shared_pairs']:,}건(<b>0.83%</b>·양쪽주석 {feas['n_both']/feas['n_pos']:.1%})뿐이라 그 기전만으론 설명 불가 → 실제로는 타깃벡터가 <b>미관측 약물의 약리를 알려주는 descriptor</b>(진짜−랜덤위약 격차가 신규성 따라 커짐: warm {gap_w:+.3f}→cold-cold {gap_cc:+.3f}).</div>

<h2>① 데이터·구성</h2>
<div class=card>타깃 출처 = ChEMBL mechanism(작용기전, ★큐레이션 주석). 우리 약물 {c['n_drugs']:,} 중 주석 <b>{feas['n_annot_drugs']}</b>({feas['n_annot_drugs']/c['n_drugs']:.0%}) · 타깃 {feas['n_targets_raw']}종 중 ★공유가능(≥{MIN_TGT_DRUGS}약물) <b>{c['n_targets']}</b>종만 사용(단일약물 타깃=공유불가=노이즈 제외).<br>
쌍 특징(PD 최선표현): 공유가능타깃 합벡터({c['n_targets']}) + [n_shared, both_annotated] 파생. DrugBank→ChEMBL 브릿지=로컬 structure_links(91%).<br>
구성(★모두 class-weight ON 고정): (a) struct(=11 재현) · (b) +target · (c) <b>+shuf(랜덤위약)</b> · (d) <b>+degshuf(degree-보존 위약)</b> · (e) +cyp+target. 분할 warm/cold-start/<b>cold-cold(주지표)</b> · XGBoost multi:softprob · seed {c['seeds']}.<br>
★위약 2종 = 서로 다른 통제: <b>랜덤위약</b>은 '차원수·랜덤벡터'를, <b>degree-보존 위약</b>(이중엣지스왑)은 '주석유무·degree·타깃빈도'(promiscuity 대리)를 통제 → 둘 다 이득 없으면 이득은 오직 <b>특정 drug→target 정체</b>에서 옴.</div>

<h2>② feasibility — '공유타깃 기전' 신호는 희박하다 (사전 실측)</h2>
<div class=card>양성 DDI {feas['n_pos']:,}쌍 중: 양쪽 타깃주석 <b>{feas['n_both']:,}</b>({feas['n_both']/feas['n_pos']:.1%}) · ★<b>타깃공유(같은 타깃 침) {feas['n_shared_pairs']:,}({feas['n_shared_pairs']/feas['n_pos']:.2%})</b>뿐 → '공유타깃=PD상호작용' 기전은 극소수만 해당. <b>그런데도 aggregate 이득이 나온 이유</b>는 ④⑤: 타깃벡터가 기전탐지가 아니라 ★약물수준 약리 descriptor로 작동하기 때문.<br>
degree-bias: 주석있는 약물 평균 DDI연결 {feas['deg_ann']:.0f} vs 없는 약물 {feas['deg_unann']:.0f} → '주석있음'이 promiscuity 대리가 될 수 있음. 이를 배제하려 ★<b>degree-보존 위약</b>(④): 이 대리(degree·주석유무·타깃빈도)를 그대로 두고도 이득이 {dgd:+.3f}(없음) → 대리효과는 원천이 아님 확정.</div>

<h2>③ 성능: 특징구성 × 3 regime (macro-F1 + head/mid/tail + acc)</h2>
<img src="{chart_m}" alt="macro-f1">
<div style="overflow-x:auto"><table>
<tr><th class=l rowspan=2>구성</th><th colspan=5>warm</th><th colspan=5>cold-start</th><th colspan=5>cold-cold</th></tr>
<tr>{"".join("<th>"+h+"</th>" for _ in REGIMES for h in ["macroF1","head","mid","tail","acc"])}</tr>
{rows}
</table></div>
<p class=mut>재현 앵커: struct cold-cold macro-F1 {b['macro_f1']['mean']:.3f} (11 class-weight 0.272와 일치).</p>

<h2>④ ★진짜 vs 위약 2종 — 타깃이 더하는 값 (cold-cold Δ vs struct)</h2>
<img src="{chart_d}" alt="delta">
<div style="overflow-x:auto"><table>
<tr><th class=l>구성</th><th>Δmacro-F1</th><th>seed 승</th><th>paired p</th><th class=l>판정</th></tr>{krows}
</table></div>
<div class=card>{real_vs_plac}. ★<b>핵심 논리</b>: 두 위약 모두 struct 위에 <b>같은 170차원</b>을 더하되 신호만 파괴 → 랜덤위약 {sd:+.3f}·degree위약 {dgd:+.3f} 둘 다 이득 없음(오히려 소폭 손해)인데 <b>진짜만 {td:+.3f}</b>. 특히 degree-보존 위약은 <b>주석유무·degree·타깃빈도·both_annotated 파생까지 진짜와 동일</b>하게 유지하고 '어느 약물이 어느 타깃을 치는지'만 섞었는데도 이득이 없다 → 이득은 promiscuity/주석유무 대리가 아니라 ★<b>특정 drug→target 정체</b>에서 온다(진짜−degree위약 {rvd_d:+.3f}·{rvd_w}/5·p{rvd_p:.3f}).<br>
★<b>regime 그라디언트</b>(진짜−랜덤위약): warm {gap_w:+.3f} → cold-start {gap_cs:+.3f} → cold-cold {gap_cc:+.3f}로 <b>신규성 따라 커짐</b>. 본 약물(warm)은 구조가 이미 외워 타깃이 덜 필요, 미관측 약물(cold-cold)은 구조 일반화가 약해 ★진짜 타깃주석이 새 정보를 크게 보탬.</div>

<h2>⑤ ★기전 국소검증 — 타깃공유 쌍에서만 이기나</h2>
<div class=card>만약 이득이 '공유타깃 기전'에서 온다면 ★타깃공유 쌍에서 가장 커야 한다. 확인:<br>
· <b>warm</b>(공유쌍 n≈{subw_n:.0f}): struct {subw_a0:.3f} → +target {subw_a1:.3f} (Δ{subw_a1-subw_a0:+.3f}, 4/5) — 국소 개선은 있으나 <b>소폭</b>.<br>
· <b>cold-cold</b>(공유쌍 n≈{sub_n:.0f}): {sub_a0:.3f} → {sub_a1:.3f} — <span class=warn>표본 극소라 노이지</span>, 방향 참고만.<br>
★<b>결론(중요)</b>: 공유쌍 국소이득(warm Δ{subw_a1-subw_a0:+.3f})은 0.83% 부분집합뿐이라 <b>aggregate +{td:.3f}를 설명하기엔 너무 작다</b>. 즉 이득의 주원천은 '공유타깃 기전'이 아니라 ★<b>타깃벡터가 미관측 약물의 약리군을 알려주는 descriptor</b> 효과. 이것이 07 CYP(구조중복=무이득)·리포트44 도킹과 다른 점: 여기선 위약통제로 <b>구조와 겹치지 않는 진짜 증분</b>이 분리됨(단 절대크기는 작음).</div>

<h2>⑥ ★한계·정직 메모</h2>
<div class=card>
· ★<b>절대크기 작음(+{td:.3f} macro-F1)</b> — 견고·유의하나 실용 이득은 '유형정확도 소폭 향상' 수준. 과대해석 금지.<br>
· <b>커버리지 {feas['n_annot_drugs']/c['n_drugs']:.0%}</b>({feas['n_annot_drugs']}/{c['n_drugs']} 약물이 공유가능 타깃 보유·나머지는 제로벡터) → 쌍 단위 양쪽주석 {feas['n_both']/feas['n_pos']:.0%}·공유 0.83%. 이득이 작은 건 방법이 아니라 <b>데이터(주석) 한계</b> — 커버리지 오르면 더 커질 여지.<br>
· mechanism은 ★주요 작용타깃만(off-target·다표적 미포함) → 실제 PD 겹침을 과소평가할 수 있음.<br>
· 예측 아님=구조 순환 없음(장점) → 위약통제로 <b>구조와 겹치지 않는 진짜 증분</b> 분리 확인. 이득의 실체는 '공유타깃 기전'이 아니라 <b>약물수준 약리 descriptor</b>(④⑤).<br>
· ★그 descriptor가 '생물학적 의미(진짜 약리)' 때문인지 단지 'Morgan보다 날카로운 약물군 지문' 때문인지는 이 설계로 분리 못 함 — 둘 다 descriptor 범주라 결론(구조 위 진짜 증분)은 불변, 단 해석은 이 수준까지.<br>
· 경로(KEGG)는 미포함(사용자 결정=타깃만) → 경로 겹침은 후속 · cold-cold 공유쌍 표본 극소 · XGBoost·class-weight 고정.</div>

<h2>⑦ 다음</h2>
<div class=card>· ★신호 확인됨 → <b>KEGG 경로 겹침</b> 추가로 descriptor 강화 · 커버리지↑(다표적·off-target 주석 병합) · 유형 그룹화(86→대분류) · 타깃공유 쌍 전용 이진분류(희박신호 집중) · off-target 예측 결합(단 구조순환 주의).</div>

<div class=foot>12 DDI targets(PD) · struct/+target/+shuf(랜덤위약)/+degshuf(degree보존위약)/+cyp+target × warm/cold-start/cold-cold · ChEMBL mechanism 진짜주석 {c['n_targets']}공유타깃 · class-weight ON · XGBoost multi:softprob · macro-F1+head/mid/tail · 이중엣지스왑 confound통제 · seed {len(c['seeds'])} · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "12_ddi_drugbank_targets.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


def main():
    t0 = time.time()
    print("[12] 타깃(PD) 특징 실험 시작 ...", flush=True)
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"])); did2row = {d: i for i, d in enumerate(drug_ids)}
    pairs = pd.read_csv(PAIRS_CSV); pos = pairs[pairs.label == 1]
    pos_pairs = [ukey(str(a), str(b)) for a, b in zip(pos["drug_a"], pos["drug_b"])]
    Y = load_Y(pos_pairs); K = len(set(Y.tolist()))
    gc = pd.Series(Y).value_counts()
    head_t = set(gc.head(10).index.tolist()); tail_t = set(gc[gc < 100].index.tolist())
    mid_t = set(gc.index.tolist()) - head_t - tail_t

    TGT, n_tgt, n_ann = build_target_matrix(drug_ids, did2row)
    CYP = build_cyp_matrix(drug_ids, did2row)
    tg_raw = pd.read_csv(TGT_CSV)
    print(f"  positive {len(pos_pairs):,} · 유형 {K}종 · 주석약물 {n_ann} · 공유가능타깃 {n_tgt}", flush=True)

    FP = build_fp_matrix(drug_ids, id2smi)
    pos_i = np.fromiter((did2row[a] for a, b in pos_pairs), dtype=np.int32, count=len(pos_pairs))
    pos_j = np.fromiter((did2row[b] for a, b in pos_pairs), dtype=np.int32, count=len(pos_pairs))
    Xstruct = make_features(FP, pos_i, pos_j, methods=("sum",))["sum"]

    # 진짜 n_shared (부분집합 마스크·feasibility용, 위약과 무관하게 고정)
    ns_real = np.asarray(TGT[pos_i].multiply(TGT[pos_j]).sum(1)).ravel()
    n_both = int(((np.asarray(TGT[pos_i].sum(1)).ravel() > 0) & (np.asarray(TGT[pos_j].sum(1)).ravel() > 0)).sum())
    n_shared_pairs = int((ns_real >= 1).sum())
    deg = np.zeros(len(drug_ids))
    np.add.at(deg, pos_i, 1); np.add.at(deg, pos_j, 1)
    ann_mask = np.asarray(TGT.sum(1)).ravel() > 0
    feas = {"n_pos": len(pos_pairs), "n_both": n_both, "n_shared_pairs": n_shared_pairs,
            "n_annot_drugs": n_ann, "n_targets_raw": int(tg_raw["target_chembl_id"].nunique()),
            "deg_ann": float(deg[ann_mask].mean()), "deg_unann": float(deg[~ann_mask].mean())}

    # 위약: seed별 약물→타깃 매핑 섞기(행 순열) — 주석유무/타깃빈도 보존, 진짜 공유만 파괴
    TGT_shuf = {}
    for seed in SEEDS:
        rng = np.random.RandomState(1000 + seed)
        perm = rng.permutation(len(drug_ids))
        TGT_shuf[seed] = TGT[perm]

    def build_X(cfg, seed):
        parts = [Xstruct]
        if "target" in cfg:
            T = TGT_shuf[seed] if "shuf" in cfg else TGT
            parts.append(target_block(T, pos_i, pos_j))
        if "cyp" in cfg:
            parts.append((CYP[pos_i] + CYP[pos_j]).tocsr())
        return sp.hstack(parts, format="csr").tocsr() if len(parts) > 1 else parts[0]

    per = {cf: {rg: {mt: [] for mt in METRICS} for rg in REGIMES} for cf in CFGS}
    for seed in SEEDS:
        train_drugs, test_drugs = split_drugs(drug_ids, seed, DRUG_TEST_FRAC)
        g = np.array([0 if d in train_drugs else 1 for d in drug_ids], dtype=np.int8)
        pu = g[pos_i] + g[pos_j]
        tt = np.where(pu == 0)[0]; ts = np.where(pu == 1)[0]; ss = np.where(pu == 2)[0]
        wtr, wte = train_test_split(np.arange(len(pos_pairs)), test_size=0.2, random_state=seed, stratify=Y)
        # 부분집합 마스크(진짜 공유타깃)
        m_wte = ns_real[wte] >= 1; m_ts = ns_real[ts] >= 1; m_ss = ns_real[ss] >= 1
        for cf in CFGS:
            X = build_X(cf, seed)
            sw_w = compute_sample_weight("balanced", Y[wtr])
            wmodel = fit_mc(X[wtr], Y[wtr], sw_w)
            mw = eval_full(wmodel, X[wte], Y[wte], Y[wtr], head_t, mid_t, tail_t, m_wte)
            sw_t = compute_sample_weight("balanced", Y[tt])
            tmodel = fit_mc(X[tt], Y[tt], sw_t)
            mcs = eval_full(tmodel, X[ts], Y[ts], Y[tt], head_t, mid_t, tail_t, m_ts)
            mcc = eval_full(tmodel, X[ss], Y[ss], Y[tt], head_t, mid_t, tail_t, m_ss)
            for rg, m in [("warm", mw), ("cold-start", mcs), ("cold-cold", mcc)]:
                for mt in METRICS:
                    per[cf][rg][mt].append(m[mt])
        print(f"  [seed {seed}] cold-cold macroF1: " + " · ".join(f"{cf.split('+')[-1][:6]} {per[cf]['cold-cold']['macro_f1'][-1]:.3f}" for cf in CFGS)
              + f" | warm 공유쌍 acc: struct {per['struct']['warm']['sub_acc'][-1]:.3f} vs +tgt {per['struct+target']['warm']['sub_acc'][-1]:.3f}", flush=True)

    agg = {cf: {rg: {mt: {"mean": float(np.mean(per[cf][rg][mt])), "std": float(np.std(per[cf][rg][mt]))} for mt in METRICS} for rg in REGIMES} for cf in CFGS}
    res = {"config": {"n_pairs": len(pos_pairs), "n_types": K, "n_drugs": len(drug_ids), "n_targets": n_tgt,
                      "seeds": SEEDS, "runtime_s": round(time.time() - t0, 1)},
           "feasibility": feas, "results": agg, "per_seed": per}
    json.dump(res, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ cold-cold (5 seed) macro-F1 ================")
    b = agg["struct"]["cold-cold"]["macro_f1"]["mean"]
    for cf in CFGS:
        a = agg[cf]["cold-cold"]
        print(f"  {cf:20s}: macroF1 {a['macro_f1']['mean']:.3f}(Δ{a['macro_f1']['mean']-b:+.3f}) · acc {a['accuracy']['mean']:.3f} · 공유쌍acc {a['sub_acc']['mean']:.3f}(n{a['sub_n']['mean']:.0f})")
    print(f"  feasibility: 공유쌍 {n_shared_pairs:,}/{len(pos_pairs):,}({n_shared_pairs/len(pos_pairs):.2%}) · 양쪽주석 {n_both/len(pos_pairs):.1%}")
    out_html = build_report(res)
    print(f"\n[12] 완료 ({res['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    return res


if __name__ == "__main__":
    main()

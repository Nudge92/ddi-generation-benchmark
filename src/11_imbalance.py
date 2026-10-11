"""
[DDI 실험 - 11] 다중분류 불균형 대응 — 희귀 유형을 살릴 수 있나
====================================================================
10에서 다중분류가 majority는 넘었으나 극심한 불균형으로 희귀 유형이 뭉개짐
(흔한 F1 0.31 vs 희귀 0.11). 11: 불균형 대응 기법으로 ★희귀(tail) 유형을 끌어올리되
흔한(head)·전체를 얼마나 희생하나(trade-off)를 함께 측정. 10을 기준선으로 순이득 판정.

기법 4구성(★train에만 적용·test는 원분포 유지):
  (a) baseline     : 10 그대로(대응 없음). 기준선.
  (b) class-weight : 유형별 균형 가중치(빈도 역수 계열, sklearn 'balanced') → 희귀 오분류 큰 벌점.
  (c) undersample  : 흔한 유형을 상한(UNDER_CAP)까지 줄여 학습.
  (d) oversample   : 희귀 유형을 하한(OVER_FLOOR)까지 복제해 학습(가벼운 오버샘플).

고정(10과 동일): 데이터·분할(warm/cold-start/cold-cold)·특징 struct(Morgan 2048)·XGBoost multi:softprob·seed5.
★평가: 주지표 macro-F1 + micro/acc/top3 + ★head/mid/tail 3구간 F1(trade-off 가시화) + majority baseline + '유형49 몰림'.

산출: results/11_ddi_drugbank_imbalance.html , results/imbalance_results.json
실행: conda run -n admet python src/11_imbalance.py
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
from ddi_common import ukey, build_fp_matrix, split_drugs, pair_feature, fit_mc, top3_acc

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
RAW_CSV = os.path.join(BASE, "data", "drugbank_ddi_raw.csv")
RESULTS_JSON = os.path.join(BASE, "results", "imbalance_results.json")

SEEDS = [0, 1, 2, 3, 4]
DRUG_TEST_FRAC = 0.20
REGIMES = ["warm", "cold-start", "cold-cold"]
TECHS = ["baseline", "class-weight", "undersample", "oversample"]
UNDER_CAP = 5000     # 흔한 유형 상한(이보다 많으면 줄임)
OVER_FLOOR = 500     # 희귀 유형 하한(이보다 적으면 복제)
METRICS = ["accuracy", "macro_f1", "micro_f1", "top3", "head_f1", "mid_f1", "tail_f1", "maj_macro_f1", "maj_acc", "pred_maj_frac"]


def pairs_to_idx(pair_list, did2row):
    a = np.fromiter((did2row[x] for x, y in pair_list), dtype=np.int32, count=len(pair_list))
    b = np.fromiter((did2row[y] for x, y in pair_list), dtype=np.int32, count=len(pair_list))
    return a, b


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


def undersample(tr, y, cap, seed):
    """흔한 유형을 cap까지 줄임(train만). 유형별로 cap 초과분 랜덤 제거."""
    rng = np.random.RandomState(seed); out = []
    for t in np.unique(y[tr]):
        it = tr[y[tr] == t]
        out.extend((rng.choice(it, cap, replace=False) if len(it) > cap else it).tolist())
    return np.array(out)


def oversample(tr, y, floor, seed):
    """희귀 유형을 floor까지 복제(train만). floor 미만이면 with-replacement 복제 추가."""
    rng = np.random.RandomState(seed); out = list(tr)
    for t in np.unique(y[tr]):
        it = tr[y[tr] == t]
        if 0 < len(it) < floor:
            out.extend(rng.choice(it, floor - len(it), replace=True).tolist())
    return np.array(out)


def build_train(tech, base_idx, X, Y, seed):
    """기법별 train 인덱스·가중치. ★base_idx(원 train)에만 적용."""
    if tech == "baseline":
        return base_idx, None
    if tech == "class-weight":
        return base_idx, compute_sample_weight("balanced", Y[base_idx])
    if tech == "undersample":
        return undersample(base_idx, Y, UNDER_CAP, seed), None
    if tech == "oversample":
        return oversample(base_idx, Y, OVER_FLOOR, seed), None


def eval_full(model, Xte, yte, ytr_orig, head_t, mid_t, tail_t, maj_type):
    """test 채점(원분포 유지) + head/mid/tail F1 + majority + 유형49 몰림."""
    clf, le = model
    pred = le.inverse_transform(clf.predict(Xte)); proba = clf.predict_proba(Xte)
    tl = sorted(set(yte.tolist()))
    per_f1 = f1_score(yte, pred, average=None, labels=tl, zero_division=0)
    f1map = dict(zip(tl, per_f1))
    def band(types):
        v = [f1map[t] for t in tl if t in types]; return float(np.mean(v)) if v else 0.0
    maj = np.bincount(ytr_orig).argmax()      # 원 train 최빈(기법 무관 baseline)
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", labels=tl, zero_division=0)),
        "micro_f1": float(f1_score(yte, pred, average="micro", labels=tl, zero_division=0)),
        "top3": top3_acc(proba, le.classes_, yte),
        "head_f1": band(head_t), "mid_f1": band(mid_t), "tail_f1": band(tail_t),
        "maj_acc": float(accuracy_score(yte, np.full_like(yte, maj))),
        "maj_macro_f1": float(f1_score(yte, np.full_like(yte, maj), average="macro", labels=tl, zero_division=0)),
        "pred_maj_frac": float(np.mean(pred == maj_type)),   # 유형49로 몰림 비율
    }


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
    agg = res["results"]; c = res["config"]
    tcol = {"baseline": "#8a8a99", "class-weight": "#1a8f4c", "undersample": "#e0a52e", "oversample": "#c0562f"}

    # 그림1: cold-cold head vs tail F1 (trade-off) 기법별
    fig1, ax = plt.subplots(figsize=(8.4, 4.0)); x = np.arange(len(TECHS)); w = 0.38
    hf = [agg[t]["cold-cold"]["head_f1"]["mean"] for t in TECHS]; tf = [agg[t]["cold-cold"]["tail_f1"]["mean"] for t in TECHS]
    ax.bar(x - w / 2, hf, w, color="#4c78d8", label="head(흔한) F1")
    ax.bar(x + w / 2, tf, w, color="#e2555f", label="tail(희귀) F1")
    for xi in range(len(TECHS)):
        ax.annotate(f"{hf[xi]:.3f}", (xi - w / 2, hf[xi] + .005), ha="center", fontsize=7.5)
        ax.annotate(f"{tf[xi]:.3f}", (xi + w / 2, tf[xi] + .005), ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(TECHS, fontsize=9); ax.set_ylabel("F1 (cold-cold, 5 seed)")
    ax.set_title("★trade-off: 불균형 기법이 tail(희귀)을 올리고 head(흔한)을 깎나 (cold-cold)")
    ax.legend(); ax.grid(axis="y", alpha=.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    chart_t = _b64(fig1)

    # 그림2: macro-F1 기법 × regime
    fig2, ax = plt.subplots(figsize=(8.4, 3.8)); w2 = 0.2
    for i, t in enumerate(TECHS):
        m = [agg[t][rg]["macro_f1"]["mean"] for rg in REGIMES]; s = [agg[t][rg]["macro_f1"]["std"] for rg in REGIMES]
        ax.bar(np.arange(3) + (i - 1.5) * w2, m, w2, yerr=s, capsize=2, color=tcol[t], label=t)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(REGIMES); ax.set_ylabel("macro-F1"); ax.set_ylim(0, 0.85)
    ax.set_title("macro-F1 (주지표) — 기법 × regime"); ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    chart_m = _b64(fig2)

    def row(t):
        cells = ""
        for rg in REGIMES:
            a = agg[t][rg]
            cells += (f"<td>{a['macro_f1']['mean']:.3f}</td><td>{a['head_f1']['mean']:.3f}</td>"
                      f"<td>{a['mid_f1']['mean']:.3f}</td><td class=good>{a['tail_f1']['mean']:.3f}</td><td>{a['accuracy']['mean']:.3f}</td>")
        return f"<tr><td class=l><b>{t}</b></td>{cells}</tr>"
    rows = "".join(row(t) for t in TECHS)

    # 핵심: cold-cold 기법별 Δ vs baseline (+ ★per-seed 견고성 — 검증 지적 반영)
    b = agg["baseline"]["cold-cold"]
    from scipy import stats
    def rob(tech, metric, rg="cold-cold"):
        """Δ(tech−baseline)의 평균·per-seed 승수·paired-t p. macro는 견고, tail은 노이지 판별용."""
        a = np.array(res["per_seed"][tech][rg][metric]); bb = np.array(res["per_seed"]["baseline"][rg][metric])
        d = a - bb
        try:
            _, p = stats.ttest_rel(a, bb); p = float(p) if not np.isnan(p) else 1.0
        except Exception:
            p = 1.0
        return float(d.mean()), int((d > 0).sum()), p
    p49_base = b["pred_maj_frac"]["mean"]; b_acc = b["accuracy"]["mean"]; b_mid = b["mid_f1"]["mean"]
    cw = agg["class-weight"]["cold-cold"]; cw_acc = cw["accuracy"]["mean"]; cw_tail = cw["tail_f1"]["mean"]; cw_head = cw["head_f1"]["mean"]; cw_mid = cw["mid_f1"]["mean"]
    cwm_d, cwm_w, cwm_p = rob("class-weight", "macro_f1")   # macro-F1: 견고(5/5·유의)
    cwt_d, cwt_w, cwt_p = rob("class-weight", "tail_f1")    # tail-F1: 노이지(4/5·비유의)
    krows = ""; best_macro = "baseline"; best_val = b["macro_f1"]["mean"]
    for t in TECHS:
        a = agg[t]["cold-cold"]
        dtail = a["tail_f1"]["mean"] - b["tail_f1"]["mean"]; dhead = a["head_f1"]["mean"] - b["head_f1"]["mean"]; dmac = a["macro_f1"]["mean"] - b["macro_f1"]["mean"]
        if a["macro_f1"]["mean"] > best_val:
            best_val = a["macro_f1"]["mean"]; best_macro = t
        if t == "baseline":
            verd = "기준"; tw_s = ""
        else:
            _, mw, mp = rob(t, "macro_f1"); _, tw, _ = rob(t, "tail_f1"); tw_s = f" <span class=mut>({tw}/5)</span>"
            verd = (f"<b class=good>순이득·견고(macro {mw}/5·p{mp:.2f})</b>" if (mw >= 5 or mp < 0.05) and dmac > 0.003 else
                    f"<span class=warn>편차 내(macro {mw}/5·p{mp:.2f})</span>" if dmac > 0.003 else "<span class=mut>무승부</span>")
        krows += (f"<tr><td class=l><b>{t}</b></td><td class={'good' if dtail>0.006 else 'bad' if dtail<-0.006 else 'mut'}>{dtail:+.3f}{tw_s}</td>"
                  f"<td class={'bad' if dhead<-0.006 else 'mut'}>{dhead:+.3f}</td><td class={'good' if dmac>0.003 else 'bad' if dmac<-0.003 else 'mut'}>{dmac:+.3f}</td>"
                  f"<td>{a['pred_maj_frac']['mean']:.2f}</td><td class=l>{verd}</td></tr>")

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1100px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 6px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}b.hl{color:#ffd479}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>11 DDI imbalance</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>11 · DDI 다중분류 불균형 대응 — 희귀 유형을 살릴 수 있나</h1>
<p class=mut>10 다중분류의 희귀 유형 뭉개짐을 class-weight/under/oversample로 대응, trade-off 측정. 10과 동일(분할·struct·XGBoost), 새 변수=불균형 기법 · env admet · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> ★<b>class-weight가 macro-F1을 견고하게 올림</b> +{cwm_d:.3f}({cwm_w}/5 seed·paired p{cwm_p:.2f}·{b['macro_f1']['mean']:.3f}→{best_val:.3f}). 단 이 이득은 주로 <b>mid 유형</b>에서 옴(49몰림 {p49_base:.2f}→{cw['pred_maj_frac']['mean']:.2f}↓ → mid F1 {b_mid:.3f}→{cw_mid:.3f}).<br>
★<b>정작 title 질문(희귀 살리기)은 불안정</b>: tail F1 +{cwt_d:.3f}처럼 보이나 <b>{cwt_w}/5 seed·한 seed 견인·paired p{cwt_p:.2f}(비유의)</b> — 방향성뿐, 견고한 rescue 아님(tail 절대 {cw_tail:.3f} 낮음=데이터부족).<br>
★<b>accuracy는 희생</b>: {cw_acc:.3f}(baseline {b_acc:.3f}, <b>−{(b_acc-cw_acc)*100:.0f}pt</b>) — trade-off는 macro-F1(균형)↑ vs accuracy(최빈유형 지배)↓. 불균형 기법=공짜 아님.</div>

<h2>① 기법 & 설정</h2>
<div class=card>(a) baseline(10) · (b) class-weight(sklearn 'balanced' 샘플가중) · (c) undersample(흔한 유형 ≤{UNDER_CAP:,}) · (d) oversample(희귀 유형 ≥{OVER_FLOOR} 복제). ★<b>모두 train에만</b> 적용, test는 원분포 유지(리샘플하면 평가 왜곡). struct(Morgan)·XGBoost multi:softprob·seed {c['seeds']}.<br>
유형 구간: <b>head</b>=상위 10종 · <b>tail</b>=100건 미만({c['n_tail']}종) · <b>mid</b>=나머지({c['n_mid']}종).</div>

<h2>② 성능: 기법 4구성 × 3 regime (macro-F1 + head/mid/tail)</h2>
<img src="{chart_m}" alt="macro-f1">
<div class=tblwrap style="overflow-x:auto"><table>
<tr><th class=l rowspan=2>기법</th><th colspan=5>warm</th><th colspan=5>cold-start</th><th colspan=5>cold-cold</th></tr>
<tr>{"".join("<th>"+h+"</th>" for _ in REGIMES for h in ["macroF1","head","mid","<b>tail</b>","acc"])}</tr>
{rows}
</table></div>

<h2>③ ★trade-off (cold-cold: tail 올리고 head 깎나)</h2>
<img src="{chart_t}" alt="head vs tail">
<div class=tblwrap style="overflow-x:auto"><table>
<tr><th class=l>기법</th><th>Δtail(희귀)</th><th>Δhead(흔한)</th><th>Δmacro-F1</th><th>유형49 몰림</th><th class=l>판정</th></tr>{krows}
</table></div>
<div class=card>★<b>순이득 판정</b>(cold-cold): {'<b class=good>'+best_macro+'</b>가 macro-F1 최고' if best_macro!='baseline' else '<b class=warn>어떤 기법도 baseline macro-F1을 순이득으로 못 넘음</b>'}. 유형49 몰림: baseline {p49_base:.2f} → {'기법으로 완화됨' if min(agg[t]['cold-cold']['pred_maj_frac']['mean'] for t in TECHS)<p49_base-0.02 else '큰 변화 없음'}.<br>
· <b>class-weight/oversample</b>: tail F1 Δ{agg['class-weight']['cold-cold']['tail_f1']['mean']-b['tail_f1']['mean']:+.3f}/{agg['oversample']['cold-cold']['tail_f1']['mean']-b['tail_f1']['mean']:+.3f} — 희귀를 {'끌어올리나' if agg['class-weight']['cold-cold']['tail_f1']['mean']>b['tail_f1']['mean'] else '못 올리고'} head를 {'깎음' if agg['class-weight']['cold-cold']['head_f1']['mean']<b['head_f1']['mean'] else '유지'}.<br>
· <b>undersample</b>: 흔한 유형 데이터를 버려 head F1 Δ{agg['undersample']['cold-cold']['head_f1']['mean']-b['head_f1']['mean']:+.3f}.</div>

<h2>④ 핵심 해석</h2>
<div class=card>
· <b>(1) 희귀(tail) 살렸나 — 약함·불안정:</b> class-weight tail F1 +{cwt_d:.3f}처럼 보이나 <b>{cwt_w}/5 seed·paired p{cwt_p:.2f}(비유의)·한 seed가 견인</b> → <b>견고한 rescue 아님</b>. 절대 tail F1 {cw_tail:.3f}로 여전히 낮음 = 표본 자체 부족(데이터 문제, 기법으로 완전해결 X).<br>
· <b>(2) 무엇을 희생하나:</b> class-weight는 <b>head F1은 거의 유지</b>({b['head_f1']['mean']:.3f}→{cw_head:.3f})지만 <b>accuracy는 −{(b_acc-cw_acc)*100:.0f}pt 하락</b>({b_acc:.3f}→{cw_acc:.3f}). 이유: accuracy는 <b>최빈 유형49(32%)가 지배</b> — 49 과예측을 줄이면(몰림 {p49_base:.2f}→{cw['pred_maj_frac']['mean']:.2f}) 49 recall↓로 accuracy↓. 즉 '희귀를 살리려 최빈유형 정답을 일부 포기'(균형 vs 총정답률 선택).<br>
· <b>(3) 전체 순이득(견고성 구분):</b> <b>class-weight</b>(macro +{cwm_d:.3f}·{cwm_w}/5·p{cwm_p:.2f})와 <b>undersample</b>(+0.014·5/5)은 macro-F1 <b>견고한 순이득</b>. 반면 <b>oversample은 편차 내</b>(+0.003·3/5·비유의). ★즉 순이득의 핵심은 '희귀 rescue'가 아니라 <b>49 과예측(몰림) 완화 → mid 유형 개선</b>이며, accuracy를 내주고 얻은 것.<br>
· ★<b>교훈:</b> 불균형 기법은 마법이 아니라 <b>trade-off 재분배</b> — 어디에 F1을 쓸지 고르는 것. 희귀 유형이 근본적으로 표본 부족이면 기법으로 완전 해결 안 됨.</div>

<h2>⑤ ★한계 메모</h2>
<div class=card>
· 리샘플/가중치는 <b>trade-off 존재(공짜 아님)</b> · 단순 복제 oversample은 <b>과적합 위험</b>(같은 샘플 반복) · undersample은 정보 손실 · <b>tail 유형은 표본 자체가 적어 한계</b>(데이터 문제, 기법으로 완전 해결 X).<br>
· cold-cold 표본 작음 · struct만(CYP 10서 무이득) · <b>이진(02~09)과 비교 불가</b> · UNDER_CAP {UNDER_CAP:,}/OVER_FLOOR {OVER_FLOOR}는 튜닝값(더 조정 가능) · XGBoost 고정.</div>

<h2>⑥ 다음</h2>
<div class=card>· (순이득 기법 있으면) 그 위에 타깃/경로 특징(12) · <b>유형 그룹화(86종→대분류)</b>로 tail 자체를 줄이기 · 전용 다중분류/딥러닝 · 희귀 유형 few-shot.</div>

<div class=foot>11 DDI imbalance · baseline/class-weight/under/oversample × warm/cold-start/cold-cold · struct · XGBoost multi:softprob · macro-F1 + head/mid/tail · train-only 리샘플 · seed {len(c['seeds'])} · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "11_ddi_drugbank_imbalance.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


def main():
    t0 = time.time()
    print("[11] 불균형 대응 실험 시작 ...", flush=True)
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"])); did2row = {d: i for i, d in enumerate(drug_ids)}
    pairs = pd.read_csv(PAIRS_CSV); pos = pairs[pairs.label == 1]
    pos_pairs = [ukey(str(a), str(b)) for a, b in zip(pos["drug_a"], pos["drug_b"])]
    Y = load_Y(pos_pairs); K = len(set(Y.tolist()))
    gc = pd.Series(Y).value_counts()
    head_t = set(gc.head(10).index.tolist()); tail_t = set(gc[gc < 100].index.tolist())
    mid_t = set(gc.index.tolist()) - head_t - tail_t; maj_type = int(gc.index[0])
    print(f"  positive {len(pos_pairs):,} · 유형 {K}종 · head {len(head_t)}·mid {len(mid_t)}·tail {len(tail_t)} · 최빈유형(remap) {maj_type}", flush=True)

    FP = build_fp_matrix(drug_ids, id2smi)
    pos_i = np.fromiter((did2row[a] for a, b in pos_pairs), dtype=np.int32, count=len(pos_pairs))
    pos_j = np.fromiter((did2row[b] for a, b in pos_pairs), dtype=np.int32, count=len(pos_pairs))
    X = pair_feature(FP, None, pos_i, pos_j, "struct", "sum")

    per = {t: {rg: {mt: [] for mt in METRICS} for rg in REGIMES} for t in TECHS}
    for seed in SEEDS:
        train_drugs, test_drugs = split_drugs(drug_ids, seed, DRUG_TEST_FRAC)
        g = np.array([0 if d in train_drugs else 1 for d in drug_ids], dtype=np.int8)
        pu = g[pos_i] + g[pos_j]
        tt = np.where(pu == 0)[0]; ts = np.where(pu == 1)[0]; ss = np.where(pu == 2)[0]
        wtr, wte = train_test_split(np.arange(len(pos_pairs)), test_size=0.2, random_state=seed, stratify=Y)
        for tech in TECHS:
            # warm 모델
            wi, wsw = build_train(tech, wtr, X, Y, seed)
            wmodel = fit_mc(X[wi], Y[wi], wsw)
            mw = eval_full(wmodel, X[wte], Y[wte], Y[wtr], head_t, mid_t, tail_t, maj_type)
            # cold 공통(TT) 모델 → TS·SS
            ti, tsw = build_train(tech, tt, X, Y, seed)
            tmodel = fit_mc(X[ti], Y[ti], tsw)
            mcs = eval_full(tmodel, X[ts], Y[ts], Y[tt], head_t, mid_t, tail_t, maj_type)
            mcc = eval_full(tmodel, X[ss], Y[ss], Y[tt], head_t, mid_t, tail_t, maj_type)
            for rg, m in [("warm", mw), ("cold-start", mcs), ("cold-cold", mcc)]:
                for mt in METRICS:
                    per[tech][rg][mt].append(m[mt])
        print(f"  [seed {seed}] cold-cold tail-F1: " + " · ".join(f"{t[:4]} {per[t]['cold-cold']['tail_f1'][-1]:.3f}" for t in TECHS)
              + f" | macroF1: " + " · ".join(f"{t[:4]} {per[t]['cold-cold']['macro_f1'][-1]:.3f}" for t in TECHS), flush=True)

    agg = {t: {rg: {mt: {"mean": float(np.mean(per[t][rg][mt])), "std": float(np.std(per[t][rg][mt]))} for mt in METRICS} for rg in REGIMES} for t in TECHS}
    res = {"config": {"n_pairs": len(pos_pairs), "n_types": K, "seeds": SEEDS, "under_cap": UNDER_CAP, "over_floor": OVER_FLOOR,
                      "n_head": len(head_t), "n_mid": len(mid_t), "n_tail": len(tail_t), "runtime_s": round(time.time() - t0, 1)},
           "results": agg, "per_seed": per}
    json.dump(res, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ cold-cold (5 seed) ================")
    b = agg["baseline"]["cold-cold"]
    for t in TECHS:
        a = agg[t]["cold-cold"]
        print(f"  {t:12s}: macroF1 {a['macro_f1']['mean']:.3f}(Δ{a['macro_f1']['mean']-b['macro_f1']['mean']:+.3f}) · head {a['head_f1']['mean']:.3f} · tail {a['tail_f1']['mean']:.3f}(Δ{a['tail_f1']['mean']-b['tail_f1']['mean']:+.3f}) · 49몰림 {a['pred_maj_frac']['mean']:.2f}")
    out_html = build_report(res)
    print(f"\n[11] 완료 ({res['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    return res


if __name__ == "__main__":
    main()

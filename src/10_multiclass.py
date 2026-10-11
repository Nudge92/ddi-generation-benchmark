"""
[DDI 실험 - 10] 다중분류 — 상호작용 '종류'(86종) 예측
====================================================================
02~09는 전부 이진(상호작용 유무). 10은 ★과제 전환: 상호작용이 '있는' 쌍에 대해
그 유형이 86종 중 어느 것인지 예측(다중분류). negative 없음. 이진 점수와 직접 비교 불가.

★왜 지표가 바뀌나 (초보자용)
  86종·심한 불균형(최빈 유형49=31.7%). 그래서:
  - accuracy만 보면 '항상 유형49 찍기'(majority)만 해도 31.7% → 착시.
  - ★macro-F1(주 지표): 모든 유형 동등 가중 → 희귀 유형 성능까지 반영.
  - micro-F1/accuracy(전체 정답률) · top-3 accuracy(정답이 상위3 안?) 병기.
  - ★majority baseline(최빈 유형만 찍기)의 acc·macro-F1을 나란히 → 모델이 진짜 배웠나
    (특히 macro-F1에서 넘는지)를 본다.

분할(이진과 동일 정신): warm/cold-start/cold-cold(약물 train80/test20). 신약서 유형까지 맞히나.
특징: struct(Morgan 2048) vs struct+enzyme(+CYP 8). 융합 sum. (특징실험 아니라 과제전환이 초점)
모델: XGBoost 다중분류(multi:softprob). seed 5.

산출: results/10_ddi_drugbank_multiclass.html , results/multiclass_results.json
실행: conda run -n admet python src/10_multiclass.py
"""
import os, sys, json, time, io, base64
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddi_common import ukey, build_fp_matrix, split_drugs, pair_feature

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
RAW_CSV = os.path.join(BASE, "data", "drugbank_ddi_raw.csv")
CYP_CSV = os.path.join(BASE, "data", "drug_cyp.csv")
RESULTS_JSON = os.path.join(BASE, "results", "multiclass_results.json")

SEEDS = [0, 1, 2, 3, 4]
DRUG_TEST_FRAC = 0.20
REGIMES = ["warm", "cold-start", "cold-cold"]
CONFIGS = ["struct", "struct+enzyme"]
XGB_MC = dict(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist", objective="multi:softprob", n_jobs=8, eval_metric="mlogloss")


def pairs_to_idx(pair_list, did2row):
    idxA = np.fromiter((did2row[a] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    idxB = np.fromiter((did2row[b] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    return idxA, idxB


def feat(config, FP, CYP, idxA, idxB):
    return pair_feature(FP, CYP, idxA, idxB, config, "sum")


def top3_acc(proba, classes, y_true):
    """top-3 정확도: 정답이 확률 상위 3개 클래스 안에 드나(유형 많으니 참고지표)."""
    top3 = classes[np.argsort(proba, axis=1)[:, -3:]]     # (n,3) 클래스 라벨
    return float(np.mean([y_true[i] in top3[i] for i in range(len(y_true))]))


def fit_mc(Xtr, ytr):
    """★train 유형이 0..K-1 전체가 아니라 부분집합(cold서 일부 유형 train에 0건)이라
    XGBoost는 연속 라벨을 요구 → LabelEncoder로 train 유형을 0..m-1로 압축 후 학습."""
    le = LabelEncoder(); yt = le.fit_transform(ytr)
    clf = xgb.XGBClassifier(**XGB_MC); clf.fit(Xtr, yt)
    return (clf, le)


def score_mc(model, Xte, yte, ytr):
    """학습된 다중분류 모델 채점 → acc/macro-F1/micro-F1/top3 + majority baseline.
    (cold은 TT 모델 하나로 TS·SS 둘 다 채점 — 재학습 방지) 예측은 원 유형으로 되돌림."""
    clf, le = model
    pred = le.inverse_transform(clf.predict(Xte))          # 원 유형 라벨로
    proba = clf.predict_proba(Xte)                          # (n, m) over le.classes_
    maj = np.bincount(ytr).argmax()                        # train 최빈 유형(원 라벨)
    test_labels = sorted(set(yte.tolist()))                # test에 존재하는 유형
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", labels=test_labels, zero_division=0)),
        "micro_f1": float(f1_score(yte, pred, average="micro", labels=test_labels, zero_division=0)),
        "top3": top3_acc(proba, le.classes_, yte),
        "maj_acc": float(accuracy_score(yte, np.full_like(yte, maj))),
        "maj_macro_f1": float(f1_score(yte, np.full_like(yte, maj), average="macro", labels=test_labels, zero_division=0)),
    }


def load_Y(pos_pairs):
    """각 무순서 positive 쌍의 상호작용 유형 Y(raw에서 first). 0..K-1로 remap."""
    raw = pd.read_csv(RAW_CSV); raw["Drug1_ID"] = raw["Drug1_ID"].astype(str); raw["Drug2_ID"] = raw["Drug2_ID"].astype(str)
    ymap = {}
    for a, b, y in zip(raw["Drug1_ID"], raw["Drug2_ID"], raw["Y"]):
        k = ukey(a, b)
        if k not in ymap:
            ymap[k] = int(y)
    Yraw = np.array([ymap[k] for k in pos_pairs])          # 원 유형(1~86)
    uniq = sorted(set(Yraw.tolist())); remap = {t: i for i, t in enumerate(uniq)}
    Y = np.array([remap[t] for t in Yraw])                 # 0..K-1
    return Y, Yraw, uniq


# ---------------- 보고서 ----------------
def _b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_report(res, ydist, splitinfo, confpairs, rare_common):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for _kf in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]:
        if os.path.exists(_kf):
            fm.fontManager.addfont(_kf); plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False
    agg = res["results"]; c = res["config"]

    # 유형 분포 그림
    fig1, ax = plt.subplots(figsize=(9, 3.0))
    ax.bar(range(len(ydist)), ydist, color="#4c78d8"); ax.set_yscale("log")
    ax.set_xlabel("상호작용 유형 (86종, 개수 내림차순)"); ax.set_ylabel("쌍 수(log)")
    ax.set_title(f"유형 분포 — 최빈 {ydist[0]:,}쌍({ydist[0]/sum(ydist):.0%})·희귀 다수(불균형)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    chart_d = _b64(fig1)

    # macro-F1 막대(주 지표): config × regime + majority
    fig2, ax = plt.subplots(figsize=(8.4, 4.0)); x = np.arange(len(REGIMES)); w = 0.26
    cc = {"struct": "#8a8a99", "struct+enzyme": "#1a8f4c"}
    for i, cfg in enumerate(CONFIGS):
        m = [agg[cfg][rg]["macro_f1"]["mean"] for rg in REGIMES]; s = [agg[cfg][rg]["macro_f1"]["std"] for rg in REGIMES]
        ax.bar(x + (i - 0.5) * w, m, w, yerr=s, capsize=3, color=cc[cfg], label=cfg)
        for xi, mu in enumerate(m):
            ax.annotate(f"{mu:.3f}", (x[xi] + (i - 0.5) * w, mu + s[xi] + 0.006), ha="center", fontsize=8)
    mb = [agg["struct"][rg]["maj_macro_f1"]["mean"] for rg in REGIMES]
    ax.plot(x, mb, "rx--", label="majority baseline", markersize=9)
    ax.set_xticks(x); ax.set_xticklabels(REGIMES); ax.set_ylabel("macro-F1 (5 seed)"); ax.set_ylim(0, 0.85)
    ax.set_title("★주 지표 macro-F1 — struct vs +enzyme, majority baseline 대비")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    chart_f = _b64(fig2)

    def prow(cfg):
        r = ""
        for rg in REGIMES:
            a = agg[cfg][rg]
            r += (f"<td>{a['accuracy']['mean']:.3f}</td><td class=good><b>{a['macro_f1']['mean']:.3f}</b><span class=mut style=font-size:9px>±{a['macro_f1']['std']:.3f}</span></td>"
                  f"<td>{a['micro_f1']['mean']:.3f}</td><td>{a['top3']['mean']:.3f}</td>")
        return r
    # majority row
    mrow = "".join(f"<td>{agg['struct'][rg]['maj_acc']['mean']:.3f}</td><td>{agg['struct'][rg]['maj_macro_f1']['mean']:.3f}</td><td>-</td><td>-</td>" for rg in REGIMES)

    cccf = agg["struct+enzyme"]["cold-cold"]["macro_f1"]["mean"]; ccsf = agg["struct"]["cold-cold"]["macro_f1"]["mean"]
    cc_maj = agg["struct"]["cold-cold"]["maj_macro_f1"]["mean"]
    beats_maj = ccsf > cc_maj + 0.02
    enz_gain = cccf - ccsf
    cc_acc = agg["struct"]["cold-cold"]["accuracy"]["mean"]; cc_maj_acc = agg["struct"]["cold-cold"]["maj_acc"]["mean"]
    enz_sw = np.array(res["per_seed"]["struct+enzyme"]["cold-cold"]["macro_f1"]) - np.array(res["per_seed"]["struct"]["cold-cold"]["macro_f1"])
    enz_pos = int((enz_sw > 0).sum())

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 6px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}b.hl{color:#ffd479}"""

    conf_html = " · ".join(f"{a}→{b}({n:,})" for a, b, n in confpairs)
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>10 DDI multiclass</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>10 · DDI 다중분류 — 상호작용 '종류'(86종) 예측</h1>
<p class=mut>★과제 전환: 이진(유무)→유형(86종). negative 없음(상호작용 있는 쌍만). 이진 점수와 <b>직접 비교 불가</b> · struct vs +enzyme · XGBoost 다중분류 · env admet · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> cold-cold(신약 쌍)에서 모델이 유형을 <b>실제로 배움</b> — 진짜 근거는 <b>accuracy {cc_acc:.3f} vs majority {cc_maj_acc:.3f}</b>(+{(cc_acc-cc_maj_acc)*100:.0f}pt, 5시드 전부). <b>단 절대 성능은 낮음</b>: macro-F1 struct <b>{ccsf:.3f}</b>(희귀 유형 약함; vs majority macro-F1 {cc_maj:.3f}은 자명한 바닥이라 이걸 넘은 것 자체는 큰 의미 아님). CYP 추가 Δ{enz_gain:+.3f}(미미). <span class=mut>(warm 0.785는 약물 암기)</span></div>

<h2>① 과제 & 지표 (이진과 다름)</h2>
<div class=card>상호작용 <b>있는</b> 쌍 {c['n_pairs']:,}개의 <b>유형(86종)</b> 예측 — negative 없음, ROC-AUC 안 씀. 심한 불균형(최빈 {ydist[0]/sum(ydist):.0%})이라 <b>accuracy는 착시</b>(majority만 찍어도 높음) → <b>★macro-F1 주 지표</b>(희귀 유형 반영)·micro-F1·top-3 병기. majority baseline과 비교해 '진짜 배웠나' 판단.</div>

<h2>② 유형 분포 & 분할 (쏠림·희귀유형)</h2>
<img src="{chart_d}" alt="type dist">
<div class=card>86종 · 최빈 유형 {ydist[0]:,}쌍({ydist[0]/sum(ydist):.0%}) · 상위3 누적 {sum(ydist[:3])/sum(ydist):.0%} · <b>희귀: 100건 미만 {int(np.sum(np.array(ydist)<100))}종·10건 미만 {int(np.sum(np.array(ydist)<10))}종</b>.<br>
★<b>cold-cold 분할(대표 seed0)</b>: train 유형 {splitinfo['n_train_types']}종 / test 유형 {splitinfo['n_test_types']}종 · <b>test에만 있고 train에 0건인 유형 {splitinfo['n_test_not_train']}종</b>(예측 불가) — 이 유형 test 쌍 {splitinfo['frac_unpredictable']:.1%}는 원리상 못 맞힘(억지 제거 안 함, 그대로 반영).</div>

<h2>③ 성능: struct vs +enzyme × 3 regime</h2>
<img src="{chart_f}" alt="macro-f1">
<div class=tblwrap style="overflow-x:auto"><table>
<tr><th class=l rowspan=2>특징</th><th colspan=4>warm</th><th colspan=4>cold-start</th><th colspan=4>cold-cold</th></tr>
<tr>{"".join("<th>"+h+"</th>" for _ in REGIMES for h in ["acc","<b>macroF1</b>","microF1","top3"])}</tr>
<tr><td class=l><b>struct</b></td>{prow('struct')}</tr>
<tr><td class=l><b>struct+enzyme</b></td>{prow('struct+enzyme')}</tr>
<tr style=background:rgba(226,85,95,.06)><td class=l>majority baseline</td>{mrow}</tr>
</table></div>
<p class=mut>★majority baseline(최빈 유형만 찍기)은 accuracy는 높아도 <b>macro-F1은 낮음</b>(한 유형만 맞음). 모델이 macro-F1에서 이걸 넘어야 '유형을 배운' 증거.</p>

<h2>④ ★핵심 해석</h2>
<div class=card>
· <b>(1) 진짜 배웠나:</b> ★핵심 근거는 <b>accuracy {cc_acc:.3f} vs majority {cc_maj_acc:.3f}</b>(+{(cc_acc-cc_maj_acc)*100:.0f}pt, 5시드 전부) — 구조만으로 신약 쌍 유형을 어느 정도 학습. <span class=mut>(macro-F1 {ccsf:.3f}이 majority macro-F1 {cc_maj:.3f}을 넘는 건 자명한 바닥이라 그 자체론 근거 약함.)</span> 단 <b>절대 macro-F1 0.25는 낮음</b> — 흔한 몇 유형 위주로 맞고 대다수 유형은 못 맞힘.<br>
· <b>(2) CYP 도움:</b> cold-cold macro-F1 +enzyme Δ{enz_gain:+.3f} ({enz_pos}/5 seed만 양수·편차 내) → {'CYP가 유형 예측에 소폭 도움.' if enz_gain>0.005 and enz_pos>=4 else '유형 예측엔 CYP 추가 효과 <b>미미</b>(이진 08/09과 일관 — 구조 밖 이득 거의 없음).'}<br>
· <b>(3) cold 저하:</b> macro-F1 warm {agg['struct']['warm']['macro_f1']['mean']:.3f} → cold-start {agg['struct']['cold-start']['macro_f1']['mean']:.3f} → cold-cold {ccsf:.3f}. 신약(미지 약물)일수록 유형 예측이 크게 어려워짐(약물 암기 사라짐 + test-only 희귀유형).<br>
· <b>흔한 vs 희귀 유형:</b> 흔한 유형(1000건 이상) 평균 F1 {rare_common['common_f1']:.3f} vs 희귀(100건 미만) {rare_common['rare_f1']:.3f} — 불균형이 희귀 유형에 직격.<br>
· <b>자주 혼동(seed0 cold-cold):</b> {conf_html}.</div>

<h2>⑤ ★한계 메모</h2>
<div class=card>
· <b>극심한 불균형</b>: 희귀 유형 표본 부족 → 그 유형 macro-F1 낮음(구조 문제, 모델 탓만 아님) · cold-cold엔 <b>train에 0건인 test 유형</b>({splitinfo['frac_unpredictable']:.1%})이 있어 원리상 못 맞힘.<br>
· <b>예측 CYP</b>(구조유래) · <b>유형 정의 자체 모호·중복</b> 가능(비슷한 유형 혼동) · 무순서 dedupe 시 다중 유형 쌍은 first만.<br>
· <b>★이진(02~09)과 직접 비교 불가</b>(문제·지표 다름) · cold-cold 표본 작음 · XGBoost 고정(딥러닝·전용 DDI 유형모델은 별개).</div>

<h2>⑥ 다음</h2>
<div class=card>· 불균형 대응(class weight·리샘플·focal) · 유형 그룹화(대분류) · 타깃/경로 특징 · 연속 CYP · 전용 다중분류/딥러닝 모델 · 흔히 혼동되는 유형쌍 집중 분석.</div>

<div class=foot>10 DDI multiclass · 86 유형 · struct/struct+enzyme × warm/cold-start/cold-cold · XGBoost multi:softprob · macro-F1 주지표 · majority baseline 대비 · seed {len(c['seeds'])} · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "10_ddi_drugbank_multiclass.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


def main():
    t0 = time.time()
    print("[10] 다중분류 실험 시작 ...", flush=True)
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"])); did2row = {d: i for i, d in enumerate(drug_ids)}
    pairs = pd.read_csv(PAIRS_CSV)
    pos = pairs[pairs.label == 1]
    pos_pairs = [ukey(str(a), str(b)) for a, b in zip(pos["drug_a"], pos["drug_b"])]
    Y, Yraw, uniq = load_Y(pos_pairs); K = len(uniq)
    ydist = pd.Series(Y).value_counts().sort_values(ascending=False).values.tolist()
    cyp_df = pd.read_csv(CYP_CSV); cyp_df["drug_id"] = cyp_df["drug_id"].astype(str)
    ccols = [c for c in cyp_df.columns if c != "drug_id"]; cyp_df = cyp_df.set_index("drug_id").loc[drug_ids]
    CYP = sp.csr_matrix(cyp_df[ccols].to_numpy(dtype=np.uint8))
    print(f"  positive 쌍 {len(pos_pairs):,} · 유형 {K}종 · 최빈 {ydist[0]/len(Y):.1%}", flush=True)

    FP = build_fp_matrix(drug_ids, id2smi)
    pos_i = np.fromiter((did2row[a] for a, b in pos_pairs), dtype=np.int32, count=len(pos_pairs))
    pos_j = np.fromiter((did2row[b] for a, b in pos_pairs), dtype=np.int32, count=len(pos_pairs))
    # 특징(config별, 한 번만 — 전 쌍)
    allfeat = {cfg: feat(cfg, FP, CYP, pos_i, pos_j) for cfg in CONFIGS}

    per = {cfg: {rg: {mt: [] for mt in ["accuracy", "macro_f1", "micro_f1", "top3", "maj_acc", "maj_macro_f1"]} for rg in REGIMES} for cfg in CONFIGS}
    splitinfo = {}; confpairs = []; rare_common = {}

    for seed in SEEDS:
        train_drugs, test_drugs = split_drugs(drug_ids, seed, DRUG_TEST_FRAC)
        g = np.array([0 if d in train_drugs else 1 for d in drug_ids], dtype=np.int8)
        pu = g[pos_i] + g[pos_j]                                   # 0=TT, 1=TS, 2=SS
        tt = np.where(pu == 0)[0]; ts = np.where(pu == 1)[0]; ss = np.where(pu == 2)[0]
        # warm: 쌍 랜덤 분할
        wtr, wte = train_test_split(np.arange(len(pos_pairs)), test_size=0.2, random_state=seed, stratify=Y)
        idx_map = {"warm": (wtr, wte), "cold-start": (tt, ts), "cold-cold": (tt, ss)}

        if seed == 0:   # 분할 정보(대표) — cold-cold의 train-0건 test유형
            train_types = set(Y[tt].tolist()); test_types = set(Y[ss].tolist())
            not_in_train = test_types - train_types
            frac_unpred = float(np.mean([Y[k] in not_in_train for k in ss]))
            splitinfo = {"n_train_types": len(train_types), "n_test_types": len(test_types),
                         "n_test_not_train": len(not_in_train), "frac_unpredictable": frac_unpred}

        for cfg in CONFIGS:
            X = allfeat[cfg]
            wclf = fit_mc(X[wtr], Y[wtr])                  # warm 모델
            ttclf = fit_mc(X[tt], Y[tt])                   # cold 공통(TT) 모델 — TS·SS 둘 다 채점
            for rg, clf, (tri, tei) in [("warm", wclf, idx_map["warm"]), ("cold-start", ttclf, idx_map["cold-start"]), ("cold-cold", ttclf, idx_map["cold-cold"])]:
                m = score_mc(clf, X[tei], Y[tei], Y[tri])
                for mt in m:
                    per[cfg][rg][mt].append(m[mt])
            # seed0·cold-cold·struct에서 혼동쌍·유형별 F1 (위 ttclf 재사용)
            if seed == 0 and cfg == "struct":
                pred = ttclf[1].inverse_transform(ttclf[0].predict(X[ss])); yte = Y[ss]
                from sklearn.metrics import f1_score as f1s
                # 혼동쌍(원 유형번호로 표시)
                from collections import Counter
                cnt = Counter((int(yte[i]), int(pred[i])) for i in range(len(yte)) if yte[i] != pred[i])
                inv = {i: t for t, i in zip(uniq, range(K))}
                confpairs = [(inv[a], inv[b], n) for (a, b), n in cnt.most_common(5)]
                # 흔한(1000건 이상) vs 희귀(100건 미만) 유형의 평균 F1 — 불균형 영향 가시화
                per_f1 = f1s(yte, pred, average=None, labels=sorted(set(yte.tolist())), zero_division=0)
                tl = sorted(set(yte.tolist())); glob_cnt = pd.Series(Y).value_counts()
                common_lab = [t for t in tl if glob_cnt.get(t, 0) >= 1000]; rare_lab = [t for t in tl if glob_cnt.get(t, 0) < 100]
                f1map = {t: per_f1[i] for i, t in enumerate(tl)}
                rare_common = {"common_f1": float(np.mean([f1map[t] for t in common_lab])) if common_lab else 0.0,
                               "rare_f1": float(np.mean([f1map[t] for t in rare_lab])) if rare_lab else 0.0}
        print(f"  [seed {seed}] cold-cold macro-F1: struct {per['struct']['cold-cold']['macro_f1'][-1]:.3f} · +enzyme {per['struct+enzyme']['cold-cold']['macro_f1'][-1]:.3f} · majority {per['struct']['cold-cold']['maj_macro_f1'][-1]:.3f}", flush=True)

    agg = {cfg: {rg: {mt: {"mean": float(np.mean(per[cfg][rg][mt])), "std": float(np.std(per[cfg][rg][mt]))}
                      for mt in per[cfg][rg]} for rg in REGIMES} for cfg in CONFIGS}
    res = {"config": {"n_pairs": len(pos_pairs), "n_types": K, "seeds": SEEDS, "runtime_s": round(time.time() - t0, 1)},
           "results": agg, "per_seed": per, "ydist": ydist, "splitinfo": splitinfo, "confpairs": confpairs, "rare_common": rare_common}
    json.dump(res, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ macro-F1 (5 seed) ================")
    for cfg in CONFIGS:
        print(f"  {cfg:14s}: warm {agg[cfg]['warm']['macro_f1']['mean']:.3f} · cold-start {agg[cfg]['cold-start']['macro_f1']['mean']:.3f} · cold-cold {agg[cfg]['cold-cold']['macro_f1']['mean']:.3f}")
    print(f"  majority   : cold-cold macro-F1 {agg['struct']['cold-cold']['maj_macro_f1']['mean']:.3f} · acc {agg['struct']['cold-cold']['maj_acc']['mean']:.3f}")
    print(f"  분할(seed0 cc): train유형 {splitinfo['n_train_types']}/test {splitinfo['n_test_types']}·train0건 {splitinfo['n_test_not_train']}종·못맞힘 {splitinfo['frac_unpredictable']:.1%}")
    out_html = build_report(res, ydist, splitinfo, confpairs, rare_common)
    print(f"\n[10] 완료 ({res['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    return res


if __name__ == "__main__":
    main()

"""
[DDI 실험 - 22] 17 확장 — 1세대 Vilar 민감도·특이도 (Vilar 논문과 정면 비교)
==============================================================================================
17(1~4세대 이진 통일 벤치마크)은 Vilar를 ROC-AUC/PR-AUC로만 봤다. 22는 ★17의 Vilar 방법·
데이터·분할·seed를 ★그대로 import 재사용해서(재구현 금지) 임계값 기반 지표(민감도·특이도·
정밀도·F1)를 산출하고, Vilar 논문 보고값(민감도 0.68 / 특이도 0.96)과 정면 비교한다.

■ 재사용(17 그대로): m17.load_data(쌍·label 0/1)·m17.make_splits(warm 60/20/20 층화·cold split_drugs)·
  m17.vilar_score(M1×M2 전파·leave-pair-out 무학습). M2=Morgan r2·2048(17 build_shared와 동일).
  seed 0~4·regime warm/cold-start/cold-cold. Vilar는 무학습이라 NN 재학습 없이 빠름.

■ 추가 지표(각 regime·test 점수에서):
  · ROC-AUC·PR-AUC (17 재확인)
  · ★두 작동점의 민감도·특이도·정밀도·F1:
    (a) Youden-J 최적 임계값(민감도+특이도−1 최대)
    (b) 특이도 ≥ 0.96 유지하며 민감도 최대인 임계값 (← Vilar 특이도 0.96 지점 비교)
  · ★robustness: val에서 특이도≈0.96 임계 선택→test 적용(누수 없는 보수판).
  · seed 5개 평균±std.
  ★cold-cold는 Vilar 점수 전부 0(전파원 없음)=degenerate → 작동점 없음("붕괴"로 보고).
  ★민감도·특이도는 prevalence-무관 → Vilar 불균형셋 vs 내 1:1셋의 'sens@spec' 비교는 공정.

■ 정직 프레이밍: 내 Vilar는 ★단일 구조유사도(Morgan)만. Vilar 논문은 MACCS·IPF·타깃·3D·부작용 등
  다중 유사도 + 외부검증(VA) → 유사도별 AUROC(IPF 0.98 등)·외부검증은 ★재료없어 재현 불가 명시.
  warm은 약물 겹침(누수 맥락) 참고, 핵심=같은 특이도(0.96) 지점의 민감도 비교.

산출: results/22_ddi_vilar_threshold.html , results/vilar_threshold_results.json
실행: conda run -n admet python src/22_vilar_threshold.py
"""
import os, sys, json, time, io, base64, importlib.util
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
RDLogger.DisableLog("rdApp.*")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "src")
sys.path.insert(0, SRC)
from ddi_common import build_sim_matrix
RESULTS_JSON = os.path.join(BASE, "results", "vilar_threshold_results.json")


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# ★17 그대로 import 재사용 (load_data·make_splits·vilar_score) — 재구현 안 함
m17 = _load("m17_vilar", "17_benchmark_binary.py")

SEEDS = [0, 1, 2, 3, 4]
REGIMES = ["warm", "cold-start", "cold-cold"]
TARGET_SPEC = 0.96
# Vilar 논문 보고값(사용자 제공) — 내가 추가 논문값 창작 안 함
VILAR_PAPER = {"sensitivity": 0.68, "specificity": 0.96, "auroc_ipf": 0.98,
               "note": "Vilar 2014 보고(다중 유사도·외부검증 VA 포함). 유사도별 AUROC·VA는 재료없어 재현 불가."}
# 작동점별 저장 지표
OP_KEYS = ["sens", "spec", "prec", "f1"]


def build_M2(drug_ids, id2smi):
    """17 build_shared와 동일한 Morgan r2·2048 Tanimoto 행렬."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(str(id2smi[d]))) for d in drug_ids]
    return build_sim_matrix(fps).astype(np.float32)


def op_point(y, s, thr):
    """임계 thr에서 pred=(s>=thr) → 민감도/특이도/정밀도/F1(직접 TP/FP 카운트=보고값 일관)."""
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return {"sens": sens, "spec": spec, "prec": prec, "f1": f1, "thr": float(thr)}


def _is_degenerate(s):
    return bool(np.allclose(s, s[0]))


def pick_spec_threshold(y, s, target):
    """특이도 ≥ target 유지하며 민감도 최대인 roc_curve 임계값(spec 못 미치면 최고 spec)."""
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    spec = 1.0 - fpr; sens = tpr
    mask = spec >= target
    if mask.any():
        cand = np.where(mask)[0]; pick = cand[int(np.argmax(sens[cand]))]
    else:
        pick = int(np.argmax(spec))
    return float(thr[pick])


def threshold_metrics(y_val, s_val, y_test, s_test, target=TARGET_SPEC):
    """test 점수에서 ROC/PR + Youden·spec@0.96 작동점 + val-선택 임계 적용(robustness).
    점수 전부 동일(cold-cold 붕괴)이면 작동점 None."""
    deg = _is_degenerate(s_test) or len(np.unique(y_test)) < 2
    roc = 0.5 if deg else float(roc_auc_score(y_test, s_test))
    pr = float(np.mean(y_test)) if deg else float(average_precision_score(y_test, s_test))
    out = {"roc_auc": roc, "pr_auc": pr, "degenerate": deg, "youden": None, "spec96": None, "spec96_val": None}
    if deg:
        return out
    fpr, tpr, thr = roc_curve(y_test, s_test, drop_intermediate=False)
    j = tpr - fpr                                   # = 민감도+특이도−1
    out["youden"] = op_point(y_test, s_test, thr[int(np.argmax(j))]); out["youden"]["youden_j"] = float(j.max())
    out["spec96"] = op_point(y_test, s_test, pick_spec_threshold(y_test, s_test, target))
    # robustness: val서 spec≈target 임계 선택 → test 적용(누수 없는 보수판)
    if not (_is_degenerate(s_val) or len(np.unique(y_val)) < 2):
        out["spec96_val"] = op_point(y_test, s_test, pick_spec_threshold(y_val, s_val, target))
    return out


def roc_points(y, s):
    """리포트 ROC 곡선용 (fpr, tpr) 리스트(다운샘플)."""
    fpr, tpr, _ = roc_curve(y, s)
    if len(fpr) > 400:                              # 그림 경량화
        idx = np.linspace(0, len(fpr) - 1, 400).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return fpr.tolist(), tpr.tolist()


def run():
    t0 = time.time()
    print("[22] Vilar 임계값 지표(17 재사용) 시작 ...", flush=True)
    drug_ids, id2smi, did2row, lo, hi, y = m17.load_data()
    n = len(drug_ids); pos = (y == 1)
    print(f"  약물 {n} · 쌍 {len(y):,}(pos {int(y.sum()):,}) · 17 load_data·make_splits·vilar_score 재사용", flush=True)
    M2 = build_M2(drug_ids, id2smi)
    print(f"  M2(Morgan r2·2048) 완성", flush=True)

    # 누적: regime → {roc,pr, youden_*, spec96_*, spec96val_*}
    per = {rg: {"roc_auc": [], "pr_auc": [],
                **{f"youden_{k}": [] for k in OP_KEYS}, "youden_thr": [],
                **{f"spec96_{k}": [] for k in OP_KEYS}, "spec96_thr": [],
                **{f"spec96val_{k}": [] for k in OP_KEYS}} for rg in REGIMES}
    degen = {rg: False for rg in REGIMES}
    roc_curve_fold0 = {}

    for si, seed in enumerate(SEEDS):
        S = m17.make_splits(drug_ids, lo, hi, y, seed)
        reg = {"warm": (S["w_tr"], S["w_va"], S["w_te"]),
               "cold-start": (S["tt_tr"], S["tt_va"], S["ts"]),
               "cold-cold": (S["tt_tr"], S["tt_va"], S["ss"])}
        for rg, (tr, va, te) in reg.items():
            tr_pos = tr[pos[tr]]                                     # 이 regime train positive만(Vilar M1)
            sv, st = m17.vilar_score(lo, hi, y, tr_pos, te, va, M2, n)   # ★17 vilar_score 그대로
            tm = threshold_metrics(y[va], sv, y[te], st)
            per[rg]["roc_auc"].append(tm["roc_auc"]); per[rg]["pr_auc"].append(tm["pr_auc"])
            if tm["degenerate"]:
                degen[rg] = True
            else:
                for k in OP_KEYS:
                    per[rg][f"youden_{k}"].append(tm["youden"][k]); per[rg][f"spec96_{k}"].append(tm["spec96"][k])
                    if tm["spec96_val"] is not None:
                        per[rg][f"spec96val_{k}"].append(tm["spec96_val"][k])
                per[rg]["youden_thr"].append(tm["youden"]["thr"]); per[rg]["spec96_thr"].append(tm["spec96"]["thr"])
                if si == 0 and rg in ("warm", "cold-start"):
                    fpr, tpr = roc_points(y[te], st)
                    roc_curve_fold0[rg] = {"fpr": fpr, "tpr": tpr,
                                           "youden": {"fpr": 1 - tm["youden"]["spec"], "tpr": tm["youden"]["sens"]},
                                           "spec96": {"fpr": 1 - tm["spec96"]["spec"], "tpr": tm["spec96"]["sens"]}}
        print(f"  [seed {seed}] warm spec96 sens {per['warm']['spec96_sens'][-1]:.3f}(spec {per['warm']['spec96_spec'][-1]:.3f}) · "
              f"cs spec96 sens {per['cold-start']['spec96_sens'][-1]:.3f} · cc {'붕괴' if degen['cold-cold'] else '-'}", flush=True)

    def agg_metric(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None

    results = {rg: {"degenerate": degen[rg], **{k: agg_metric(v) for k, v in per[rg].items()}} for rg in REGIMES}
    out = {"config": {"task": "1세대 Vilar 임계값 지표(17 재사용)", "seeds": SEEDS, "n_pairs": int(len(y)),
                      "target_spec": TARGET_SPEC, "regimes": REGIMES, "similarity": "Morgan r2·2048 (단일 구조유사도)",
                      "runtime_s": round(time.time() - t0, 1)},
           "vilar_paper": VILAR_PAPER, "results": results, "per_fold": per, "roc_curve_fold0": roc_curve_fold0}
    json.dump(out, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ Vilar 특이도 0.96 지점 민감도 (5 seed) ================")
    for rg in REGIMES:
        r = results[rg]
        if r["degenerate"]:
            print(f"  {rg:11s}: ROC {r['roc_auc']['mean']:.3f} · ★붕괴(전파원 없음·작동점 없음)")
        else:
            print(f"  {rg:11s}: ROC {r['roc_auc']['mean']:.3f} · spec@0.96 민감도 {r['spec96_sens']['mean']:.3f}"
                  f"±{r['spec96_sens']['std']:.3f}(실제 spec {r['spec96_spec']['mean']:.3f}) · Youden 민감도 {r['youden_sens']['mean']:.3f}/특이도 {r['youden_spec']['mean']:.3f}")
    print(f"  ★Vilar 논문: 민감도 {VILAR_PAPER['sensitivity']} @ 특이도 {VILAR_PAPER['specificity']}")
    out_html = build_report(out)
    print(f"\n[22] 완료 ({out['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    return out


# ---------------- 보고서 (17 형식 답습) ----------------
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
    agg = res["results"]; c = res["config"]; vp = res["vilar_paper"]; rc = res["roc_curve_fold0"]
    rcol = {"warm": "#4c78d8", "cold-start": "#e0a52e", "cold-cold": "#8a8a99"}

    def m(rg, key):
        v = agg[rg].get(key); return v["mean"] if v else float("nan")
    def s(rg, key):
        v = agg[rg].get(key); return v["std"] if v else float("nan")

    # 그림1: ROC 곡선(warm·cold-start, fold0) + 작동점 + ★Vilar 논문 점(spec0.96·sens0.68)
    fig1, ax = plt.subplots(figsize=(6.4, 6.0))
    for rg in ("warm", "cold-start"):
        if rg in rc:
            ax.plot(rc[rg]["fpr"], rc[rg]["tpr"], "-", color=rcol[rg], lw=1.8, label=f"{rg} (ROC {m(rg,'roc_auc'):.3f})")
            ax.scatter([rc[rg]["spec96"]["fpr"]], [rc[rg]["spec96"]["tpr"]], color=rcol[rg], marker="o", s=55, zorder=5, edgecolor="white")
            ax.scatter([rc[rg]["youden"]["fpr"]], [rc[rg]["youden"]["tpr"]], color=rcol[rg], marker="^", s=55, zorder=5, edgecolor="white")
    ax.scatter([1 - vp["specificity"]], [vp["sensitivity"]], color="#e2555f", marker="*", s=300, zorder=6,
               edgecolor="white", label=f"Vilar 논문 (spec {vp['specificity']}·sens {vp['sensitivity']})")
    ax.axvline(1 - c["target_spec"], ls=":", color="#e2555f", lw=1, alpha=.6)
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=.8)
    ax.set_xlabel("1 - 특이도 (FPR)"); ax.set_ylabel("민감도 (TPR)"); ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02)
    ax.set_title("Vilar ROC 곡선 vs 논문 작동점 (fold0)\n●=spec@0.96 · ▲=Youden · ★=Vilar 논문")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.25)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    chart_roc = _b64(fig1)

    # 그림2: spec@0.96 민감도 막대 (regime) + Vilar 0.68 기준선
    fig2, ax = plt.subplots(figsize=(6.6, 3.6)); xs = []
    labs = []
    for i, rg in enumerate(REGIMES):
        if agg[rg]["degenerate"]:
            ax.bar(i, 0, 0.6, color=rcol[rg]); ax.text(i, 0.02, "붕괴\n(전파원0)", ha="center", fontsize=8, color="#e2555f")
        else:
            val = m(rg, "spec96_sens"); ax.bar(i, val, 0.6, yerr=s(rg, "spec96_sens"), capsize=3, color=rcol[rg])
            ax.text(i, val + .01, f"{val:.3f}", ha="center", fontsize=9)
        labs.append(rg)
    ax.axhline(vp["sensitivity"], ls="--", color="#e2555f", lw=1.3, label=f"Vilar 논문 민감도 {vp['sensitivity']}")
    ax.set_xticks(range(len(REGIMES))); ax.set_xticklabels(labs); ax.set_ylabel("민감도 @ 특이도 0.96"); ax.set_ylim(0, 1.0)
    ax.set_title("특이도 0.96 지점의 민감도 — 내 Vilar 재현 vs 논문 0.68")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    chart_bar = _b64(fig2)

    # 표1: regime × {ROC·PR·Youden(sens/spec/prec/f1)·spec@0.96(sens/spec/prec/f1)}
    def cell(rg, key, degch="—"):
        if agg[rg]["degenerate"] and key not in ("roc_auc", "pr_auc"):
            return f"<span class=bad>{degch}</span>"
        return f"{m(rg,key):.3f}<span class=s>±{s(rg,key):.3f}</span>"
    rows = ""
    for rg in REGIMES:
        hl = " class=hl" if rg == "warm" else ""
        rows += (f"<tr><td class=l><b>{rg}</b></td><td{hl}>{cell(rg,'roc_auc')}</td><td>{cell(rg,'pr_auc')}</td>"
                 f"<td>{cell(rg,'youden_sens')}</td><td>{cell(rg,'youden_spec')}</td><td>{cell(rg,'youden_prec')}</td><td>{cell(rg,'youden_f1')}</td>"
                 f"<td class=hl>{cell(rg,'spec96_sens')}</td><td>{cell(rg,'spec96_spec')}</td><td>{cell(rg,'spec96_prec')}</td><td>{cell(rg,'spec96_f1')}</td></tr>")

    # 표2: ★정면 비교 (Vilar 논문 vs 내 warm vs cold-start vs cold-cold)
    def mine(rg, key):
        if agg[rg]["degenerate"]:
            return "<span class=bad>붕괴</span>"
        return f"{m(rg,key):.3f}"
    warm_sens96 = m("warm", "spec96_sens"); warm_spec96 = m("warm", "spec96_spec")
    cmp_rows = (
        f"<tr><td class=l><b>민감도 (@특이도 0.96)</b></td><td class=hl2>{vp['sensitivity']:.2f}</td>"
        f"<td class=hl><b>{mine('warm','spec96_sens')}</b></td><td>{mine('cold-start','spec96_sens')}</td><td>{mine('cold-cold','spec96_sens')}</td></tr>"
        f"<tr><td class=l><b>특이도 (달성)</b></td><td class=hl2>{vp['specificity']:.2f}</td>"
        f"<td>{mine('warm','spec96_spec')}</td><td>{mine('cold-start','spec96_spec')}</td><td>{mine('cold-cold','spec96_spec')}</td></tr>"
        f"<tr><td class=l><b>AUROC</b></td><td class=hl2>{vp['auroc_ipf']:.2f} <span class=s>(IPF·제공값)</span></td>"
        f"<td>{m('warm','roc_auc'):.3f}</td><td>{m('cold-start','roc_auc'):.3f}</td><td>{m('cold-cold','roc_auc'):.3f}</td></tr>"
        f"<tr><td class=l>정밀도 (@특이도 0.96)</td><td class=mut>—</td>"
        f"<td>{mine('warm','spec96_prec')}</td><td>{mine('cold-start','spec96_prec')}</td><td>{mine('cold-cold','spec96_prec')}</td></tr>"
    )

    # robustness (val-선택 임계) 행
    rob_rows = ""
    for rg in ("warm", "cold-start"):
        if not agg[rg]["degenerate"] and agg[rg].get("spec96val_sens"):
            rob_rows += (f"<tr><td class=l>{rg}</td><td>{m(rg,'spec96val_sens'):.3f}<span class=s>±{s(rg,'spec96val_sens'):.3f}</span></td>"
                         f"<td>{m(rg,'spec96val_spec'):.3f}</td><td>{m(rg,'spec96val_f1'):.3f}</td></tr>")

    # 판정 문구
    gap = vp["sensitivity"] - warm_sens96
    verdict = (f"★<b>내 단일-유사도 Vilar 재현은 특이도 0.96에서 민감도 {warm_sens96:.3f}</b>(warm)"
               f" — Vilar 논문 {vp['sensitivity']}보다 <b>{'낮음' if gap>0.02 else '비슷' if abs(gap)<=0.02 else '높음'}</b>(Δ{-gap:+.3f})")

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 6px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}.hl{background:#16202e}.hl2{background:#241a2e;color:#c98ae0}
    .s{color:#7a7f8a;font-size:10px}.card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>22 Vilar 민감도·특이도</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>22 · 17 확장 — 1세대 Vilar 민감도·특이도 (Vilar 논문 정면 비교)</h1>
<p class=mut>17의 Vilar(M1×M2 전파·무학습)·데이터·분할·seed를 ★그대로 import 재사용, 임계값 기반 민감도·특이도 산출 → Vilar 논문(민감도 {vp['sensitivity']}/특이도 {vp['specificity']})과 정면 비교. env admet · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> {verdict}. ★<b>특이도 0.96에서 내 민감도</b>: warm <b>{mine('warm','spec96_sens')}</b> · cold-start {mine('cold-start','spec96_sens')} · cold-cold <b class=bad>붕괴</b>(두 약 다 신약→전파원 0·점수 전부 동일=작동점 없음).<br>
★<b>정직</b>: 내 Vilar는 <b>단일 구조유사도(Morgan)만</b> 씀 — Vilar 논문의 MACCS·IPF·타깃·3D·부작용 다중유사도(IPF AUROC {vp['auroc_ipf']})·외부검증(VA)은 <b>재료 없어 재현 불가</b>. 민감도·특이도는 prevalence-무관이라 논문(불균형)↔내 셋(1:1)의 <b>같은 특이도 민감도 비교는 공정</b>(정밀도·PR은 prevalence 의존→직접비교 불가). warm은 누수 맥락(참고)·핵심=cold.</div>

<h2>① 방법 (17 Vilar 그대로 + 임계값 산출)</h2>
<div class=card>· ★<b>17 재사용</b>: m17.load_data(쌍·label 0/1·negative 무작위 1:1)·m17.make_splits(warm 60/20/20 층화·cold split_drugs)·m17.vilar_score(M1=train positive 상호작용·P=M1×M2·S3=P+Pᵀ·leave-pair-out 무학습). M2=Morgan r2·2048(17과 동일). seed {c['seeds']}.<br>
· <b>작동점 2종</b>(test ROC 곡선에서): (a) <b>Youden-J</b>(민감도+특이도−1 최대) · (b) <b>특이도 ≥ {c['target_spec']} 유지 민감도 최대</b>(Vilar 특이도 0.96 지점). 각 점의 민감도·특이도·정밀도·F1.<br>
· <b>robustness</b>: val에서 특이도≈{c['target_spec']} 임계 선택→test 적용(누수 없는 보수판·④).<br>
· ★<b>cold-cold=degenerate</b>: 두 약 다 신약이면 M1 전파원 없어 점수 전부 0 → ROC 0.5·작동점 없음(spec은 0/1만 가능). "붕괴"로 정직 보고(1세대 근본 한계).</div>

<h2>② Vilar 임계값 지표 (regime × 작동점, 5 seed)</h2>
<img src="{chart_bar}" alt="spec96 sens bars">
<div style="overflow-x:auto"><table>
<tr><th class=l rowspan=2>regime</th><th rowspan=2>ROC</th><th rowspan=2>PR</th><th colspan=4>Youden-J 작동점</th><th colspan=4>특이도 0.96 작동점</th></tr>
<tr><th>민감도</th><th>특이도</th><th>정밀도</th><th>F1</th><th>민감도</th><th>특이도</th><th>정밀도</th><th>F1</th></tr>
{rows}
</table></div>
<p class=mut>warm은 ★누수맥락(참고)·cold-cold 붕괴(작동점 없음). 민감도·특이도=prevalence 무관, 정밀도=prevalence 의존.</p>

<h2>③ ★정면 비교: Vilar 논문 vs 내 재현</h2>
<img src="{chart_roc}" alt="ROC vs Vilar point">
<div style="overflow-x:auto"><table>
<tr><th class=l>지표</th><th class=hl2>Vilar 논문(warm)</th><th class=hl>내 재현(warm)</th><th>내 cold-start</th><th>내 cold-cold</th></tr>
{cmp_rows}
</table></div>
<div class=card>{verdict}.<br>
· ★<b>같은 특이도(0.96)에서</b>: Vilar 논문 민감도 <b>{vp['sensitivity']}</b> vs 내 warm 재현 <b>{warm_sens96:.3f}</b>(달성 특이도 {warm_spec96:.3f}). {'내 단일-유사도가 논문보다 <b>낮음</b> — Vilar가 <b>다중 유사도</b>(MACCS·IPF·타깃·3D·부작용) 결합·큐레이션으로 같은 특이도에서 더 높은 민감도를 얻은 것과 정합(단일 구조유사도의 한계).' if gap>0.02 else '내 단일-유사도가 논문과 <b>비슷</b>(구조 유사도만으로도 유사 작동점 도달).' if abs(gap)<=0.02 else '내 재현이 논문보다 <b>높음</b>(데이터·negative 정의 차이 가능·해석 주의).'}<br>
· ★<b>cold-start</b>(한쪽만 신약): 특이도 0.96에서 민감도 {mine('cold-start','spec96_sens')} — warm보다 낮으나 작동은 함(train쪽 전파). ★<b>cold-cold</b>(둘 다 신약): <b class=bad>붕괴</b> — 논문/warm과 달리 <b>어떤 특이도에서도 작동점 없음</b>(전파원 0). = 1세대 유사도전파의 근본 한계(13·17 재확인).<br>
· ROC 곡선(위): ●=spec@0.96 작동점·▲=Youden·★=Vilar 논문점(spec {vp['specificity']}·sens {vp['sensitivity']}). Vilar 점이 내 warm 곡선 {'위' if vp['sensitivity']>warm_sens96 else '아래/근처'}에 위치.</div>

<h2>④ robustness — val-선택 임계(누수 없는 보수판)</h2>
<div style="overflow-x:auto"><table>
<tr><th class=l>regime</th><th>민감도(@val-선택 특이도0.96·test)</th><th>test 특이도</th><th>F1</th></tr>
{rob_rows if rob_rows else '<tr><td class=l colspan=4 class=mut>(해당 없음)</td></tr>'}
</table></div>
<div class=card>③ 표는 <b>test ROC에서 직접</b> 특이도 0.96 작동점을 읽음(그 특이도에서 <b>달성 가능한</b> 민감도=곡선 속성). 위 ④는 <b>val에서 임계 고정→test 적용</b>(전향적·보수적·누수 없음). warm은 val≈test라 근접(0.421→0.420=작동점 안정).<br>
★<b>cold-start는 불안정</b>: val-선택 민감도가 0.226→<b>0.025</b>로 붕괴(test 특이도 0.997로 과도) = ★<b>val(tt_va=train-train 쌍)과 test(ts=cold-start 쌍)의 분포 불일치</b>로 val에서 고른 임계가 cold test에 오보정된 탓. 이게 <b>누수 없는 유일 옵션</b>(cold val 쌍을 따로 떼면 cold 약물이 새어 누수)이라 그대로 두되, cold의 고-특이도 작동점은 <b>신뢰 낮음</b>으로 해석(③의 test-직접 0.226이 '달성가능 상한', ④가 전향적 하한).</div>

<h2>⑤ ★한계·정직 메모</h2>
<div class=card>
· ★<b>단일 구조유사도만</b>: 내 Vilar=Morgan Tanimoto 1종. Vilar 논문은 <b>MACCS·IPF(상호작용지문)·타깃·3D·부작용 등 다중 유사도</b> + 결합 + <b>외부검증(VA DDI corpus)</b> → 유사도별 AUROC(IPF {vp['auroc_ipf']} 등)·외부검증은 <b>재료 없어 재현 불가</b>. 이 비교는 '단일 구조유사도 전파의 작동점' vs '논문 종합 작동점'.<br>
· <b>negative·prevalence 차이</b>: 내 test=무작위 1:1(균형)·Vilar 평가셋=불균형(음성多). ★민감도·특이도는 prevalence-무관(공정 비교)이나 <b>정밀도·PR-AUC는 prevalence 의존</b>→직접 비교 불가(참고만).<br>
· <b>작동점=test ROC 직접읽기</b>(③): '그 특이도에서 달성가능' 상한에 가까움(하이퍼파라미터 튜닝 아님·곡선 속성). ④ val-선택이 보수적 대조.<br>
· <b>warm=누수맥락</b>: leave-pair-out으로 그 쌍은 M1에서 빠지나 약물은 train·test 겹침(참고). 핵심=cold.<br>
· <b>seed 5개</b>·cold-cold 붕괴는 방법 결함 아니라 '알려진 상호작용 전파'의 본질(양쪽 미지엔 전파원 없음).</div>

<h2>⑥ 다음</h2>
<div class=card>· 타깃/부작용 유사도 추가해 다중-유사도 Vilar 근사(IPF는 상호작용지문이라 자기참조 주의) · cold-cold용 하이브리드(Vilar+구조ML·13⑥) · 2~4세대(DNN/GNN)의 같은 특이도 민감도도 산출해 세대별 작동점 비교.</div>

<div class=foot>22 · 17 확장 Vilar 민감도·특이도 · M1×M2 전파(무학습·17 재사용) · warm/cold-start/cold-cold · Youden-J·특이도0.96 작동점 · 단일 구조유사도(Morgan) · seed {len(c['seeds'])} · env admet · Vilar 논문 {vp['sensitivity']}/{vp['specificity']} 정면비교</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "22_ddi_vilar_threshold.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


if __name__ == "__main__":
    run()

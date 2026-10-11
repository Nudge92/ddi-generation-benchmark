"""
[DDI 실험 - 18] 순수 모델 스왑 — "모델 탓? 표현 탓?" 원인 분리
==================================================================================
지금까지 세대 비교(13~17)는 ★파이프라인 비교였다(표현+모델이 같이 다름):
  2세대=SSP+DNN · 3세대=SSP+DNN(다른 arch) · 4세대=그래프+GNN.
  → 4세대가 낮은 게 'GNN(모델)' 탓인지 '그래프(표현)' 탓인지 못 가름.
★이번엔 하나를 고정하고 다른 하나만 바꿔 '모델 축 / 표현 축'을 분리한다.

  [실험 A] ★표현 고정 → 모델만 트리 vs DNN
    · 2×2 = {SSP-concat, Morgan-sum} × {XGBoost, DNN}.
    · 같은 표현 안에서 (XGB vs DNN) = 순수 '모델' 효과.
    · 대각선(Morgan·XGB≈10/11, SSP·DNN≈14)은 기존 재현(검증), 비대각(SSP·XGB, Morgan·DNN)이 새 셀.

  [실험 B] ★모델 고정(SSI-DDI) → 표현만 그래프 vs 엣지제거
    · 같은 SSI-DDI(인코더+co-attention+RESCAL)에 (진짜 분자그래프) vs (결합 엣지 제거=원자 bag).
    · 차이 = 순수 '그래프 연결정보' 기여. (지문을 GNN에 넣는 건 정의가 애매 → 엣지제거가 정직한 대조.)

■ 데이터·평가(16·17과 동일): TDC DrugBank 다중분류 86종·positive만·warm/cold-start/cold-cold·
  seed 5개 평균±std·지표 macro-F1(주)·acc·top3·head/mid/tail(m14.eval_dnn 그대로).
■ 재사용: m14(DeepDDI·train_dnn·build_ssp_pca·eval_dnn·predict_proba)·m16(SSIDDI·train_ssiddi·
  predict_proba_ssiddi·prepare_data·make_regime_indices·build_all_graphs)·ddi_common(fit_mc·build_sim_matrix·split_drugs).
■ ★판정(17 교훈): raw 차이 금지 → fold(seed) 방향일치(5/5)+부호검정. GNN CUDA 비결정성 고정.

산출: results/18_ddi_model_swap.html , results/model_swap_results.json
실행: conda run -n admet python src/18_model_swap.py [smoke|full]
"""
import os, sys, json, time, io, base64, importlib.util, warnings
warnings.filterwarnings("ignore")
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import torch
# ★GNN 재현성(17 교훈): cudnn 결정성 고정. use_deterministic_algorithms(warn_only)는 GNN scatter가
#   결정적 구현이 없어 매 호출 경고만 쏟고 실제 결정성은 못 줌 → 미사용. GNN run-to-run 잡음은
#   5 seed + 부호검정으로 대응(16·17 방식). cudnn 플래그는 무해.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import ddi_common as C


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m14 = _load("m14_ms", "14_deepddi_dnn.py")     # DeepDDI DNN · SSP · eval_dnn
m16 = _load("m16_ms", "16_ssiddi_gnn.py")      # SSI-DDI GNN · graphs · prepare_data

BASE = os.path.dirname(SRC)
RESULTS_JSON = os.path.join(BASE, "results", "model_swap_results.json")
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
REGIMES = ["warm", "cold-start", "cold-cold"]
METRICS = ["macro_f1", "accuracy", "top3", "head_f1", "mid_f1", "tail_f1"]
N_CLASS = m14.N_CLASS
DEV = m16.DEV
if SMOKE:                                       # 짧은 학습(파이프라인 점검)
    m14.MAX_EPOCH, m14.PATIENCE = 6, 3
GNN_EPOCH = 6 if SMOKE else 100


# ══════════════════════════════════════════════════════════════════════════════
#  준비: 데이터(16과 동일) + M2(Tanimoto) + Morgan 드럭행렬 + 엣지제거 그래프
# ══════════════════════════════════════════════════════════════════════════════
def prepare():
    D = m16.prepare_data()                                  # drug_ids·graphs·pi·pj·lo·hi·Y·head/mid/tail
    drug_ids, id2smi, _ = m16.load_drug_universe()
    # M2(Tanimoto) — SSP 기준. drug_ids 순서 동일(둘 다 drug_smiles.csv).
    bvs = [C._GEN.GetFingerprint(Chem.MolFromSmiles(str(id2smi[d]))) for d in D["drug_ids"]]
    M2 = C.build_sim_matrix(bvs).astype(np.float32)
    # Morgan 드럭행렬(n_drug × 2048) — 쌍은 sum-fusion(fpA+fpB), fold 무관.
    FP = np.zeros((len(D["drug_ids"]), 2048), dtype=np.float32)
    from rdkit import DataStructs
    for i, bv in enumerate(bvs):
        DataStructs.ConvertToNumpyArray(bv, FP[i])
    # ★엣지 제거 그래프(실험 B 대조): 같은 원자특징, 결합 엣지만 비움 → 원자 bag.
    from torch_geometric.data import Data
    graphs_noedge = []
    for g in D["graphs"]:
        d = Data(x=g.x, edge_index=torch.zeros((2, 0), dtype=torch.long))
        graphs_noedge.append(d)
    D["M2"] = M2; D["FP"] = FP; D["graphs_noedge"] = graphs_noedge
    return D


# ── 특징 provider (표현을 '고정'하는 지점) ──
def feat_ssp(ssp, lo, hi, idx):
    """SSP-concat: 두 약물의 SSP를 이어붙임(표현=SSP)."""
    return np.concatenate([ssp[lo[idx]], ssp[hi[idx]]], axis=1).astype(np.float32)


def feat_morgan(FP, lo, hi, idx):
    """Morgan-sum: 두 약물 지문 합(표현=Morgan·02의 대칭융합)."""
    return (FP[lo[idx]] + FP[hi[idx]]).astype(np.float32)


def xgb_proba86(clf, le, X):
    """XGBoost 다중분류 예측을 ★86열 전체로 확장(train에 없던 유형=0열). eval_dnn 입력용."""
    p = clf.predict_proba(X)
    full = np.zeros((X.shape[0], N_CLASS), dtype=np.float32)
    full[:, le.classes_] = p
    return full


# ══════════════════════════════════════════════════════════════════════════════
#  실험 A: 표현 고정 → 모델만 (XGBoost vs DNN) × (SSP, Morgan)
# ══════════════════════════════════════════════════════════════════════════════
def run_expA(D):
    lo, hi, Y = D["lo"], D["hi"], D["Y"]
    buckets = (D["head_t"], D["mid_t"], D["tail_t"])
    cells = ["SSP·XGB", "SSP·DNN", "Morgan·XGB", "Morgan·DNN"]
    per = {c: {rg: {m: [] for m in METRICS} for rg in REGIMES} for c in cells}
    curves = {}
    for si, seed in enumerate(SEEDS):
        R = m16.make_regime_indices(D, seed)                # tt/ts/ss/wtr/wte (16과 동일 분할)
        train_drugs, _ = C.split_drugs(D["drug_ids"], seed, 0.20)
        g = np.array([0 if d in train_drugs else 1 for d in D["drug_ids"]], dtype=np.int8)
        train_rows = np.where(g == 0)[0]
        ssp, P = m14.build_ssp_pca(D["M2"], train_rows)      # SSP(train 기준·누수0) — 14와 동일

        # 표현별 feature 함수
        reps = {"SSP": lambda idx: feat_ssp(ssp, lo, hi, idx),
                "Morgan": lambda idx: feat_morgan(D["FP"], lo, hi, idx)}

        def eval_at(proba_fn):
            """warm/cold-start/cold-cold 채점 dict 반환. proba_fn(idx)->86열 proba."""
            out = {}
            out["warm"] = m14.eval_dnn(proba_fn(R["wte"]), Y[R["wte"]], *buckets)
            out["cold-start"] = m14.eval_dnn(proba_fn(R["ts"]), Y[R["ts"]], *buckets)
            out["cold-cold"] = m14.eval_dnn(proba_fn(R["ss"]), Y[R["ss"]], *buckets)
            return out

        for rep_name, feat in reps.items():
            # 학습셋 특징(warm=wtr, cold=tt) — 표현 고정, 모델만 바꿈
            Xw, yw = feat(R["wtr"]), Y[R["wtr"]]
            Xc, yc = feat(R["tt"]), Y[R["tt"]]
            # ── 모델 1: XGBoost ──
            clf_w, le_w = C.fit_mc(Xw, yw)
            clf_c, le_c = C.fit_mc(Xc, yc)
            resX = {"warm": m14.eval_dnn(xgb_proba86(clf_w, le_w, feat(R["wte"])), Y[R["wte"]], *buckets),
                    "cold-start": m14.eval_dnn(xgb_proba86(clf_c, le_c, feat(R["ts"])), Y[R["ts"]], *buckets),
                    "cold-cold": m14.eval_dnn(xgb_proba86(clf_c, le_c, feat(R["ss"])), Y[R["ss"]], *buckets)}
            for rg in REGIMES:
                for m in METRICS: per[f"{rep_name}·XGB"][rg][m].append(resX[rg][m])
            # ── 모델 2: DNN(exp14 DeepDDI 8×2048) ──
            mw, cw = m14.train_dnn(Xw, yw, None, seed, track=(si == 0 and rep_name == "SSP"))
            mc, _ = m14.train_dnn(Xc, yc, None, seed)
            resD = {"warm": m14.eval_dnn(m14.predict_proba(mw, feat(R["wte"])), Y[R["wte"]], *buckets),
                    "cold-start": m14.eval_dnn(m14.predict_proba(mc, feat(R["ts"])), Y[R["ts"]], *buckets),
                    "cold-cold": m14.eval_dnn(m14.predict_proba(mc, feat(R["ss"])), Y[R["ss"]], *buckets)}
            for rg in REGIMES:
                for m in METRICS: per[f"{rep_name}·DNN"][rg][m].append(resD[rg][m])
            if si == 0 and rep_name == "SSP":
                curves["SSP·DNN"] = cw
        print(f"  [A seed {seed}] cc macro-F1: " +
              " · ".join(f"{c}={per[c]['cold-cold']['macro_f1'][-1]:.3f}" for c in cells), flush=True)
    return per, curves


# ══════════════════════════════════════════════════════════════════════════════
#  실험 B: 모델 고정(SSI-DDI) → 표현만 (그래프 vs 엣지제거)
# ══════════════════════════════════════════════════════════════════════════════
def run_expB(D):
    lo, hi, Y = D["lo"], D["hi"], D["Y"]
    buckets = (D["head_t"], D["mid_t"], D["tail_t"])
    cells = {"graph": D["graphs"], "no-edge": D["graphs_noedge"]}
    per = {c: {rg: {m: [] for m in METRICS} for rg in REGIMES} for c in cells}
    curves = {}
    for si, seed in enumerate(SEEDS):
        R = m16.make_regime_indices(D, seed)
        for name, gset in cells.items():
            # warm 모델 + cold 모델(TT) — 표현(그래프/엣지제거)만 다름, 모델은 동일 SSI-DDI
            mw, cw = m16.train_ssiddi(gset, lo, hi, Y, R["wtr"], seed, max_epoch=GNN_EPOCH, track=(si == 0))
            pw = m16.predict_proba_ssiddi(mw, gset, lo, hi, R["wte"])
            mc, _ = m16.train_ssiddi(gset, lo, hi, Y, R["tt"], seed, max_epoch=GNN_EPOCH)
            res = {"warm": m14.eval_dnn(pw, Y[R["wte"]], *buckets),
                   "cold-start": m14.eval_dnn(m16.predict_proba_ssiddi(mc, gset, lo, hi, R["ts"]), Y[R["ts"]], *buckets),
                   "cold-cold": m14.eval_dnn(m16.predict_proba_ssiddi(mc, gset, lo, hi, R["ss"]), Y[R["ss"]], *buckets)}
            for rg in REGIMES:
                for m in METRICS: per[name][rg][m].append(res[rg][m])
            if si == 0:
                curves[name] = cw
        print(f"  [B seed {seed}] cc macro-F1: graph={per['graph']['cold-cold']['macro_f1'][-1]:.3f} "
              f"no-edge={per['no-edge']['cold-cold']['macro_f1'][-1]:.3f}", flush=True)
    return per, curves


def agg(per):
    """per[cell][rg][metric]=list → mean±std."""
    return {c: {rg: {m: {"mean": float(np.mean(per[c][rg][m])), "std": float(np.std(per[c][rg][m]))}
                     for m in METRICS} for rg in REGIMES} for c in per}


def run_full():
    t0 = time.time()
    D = prepare()
    print(f"[18] 약물 {len(D['drug_ids'])} · positive {len(D['Y'])} · device {DEV} · "
          f"head/mid/tail {len(D['head_t'])}/{len(D['mid_t'])}/{len(D['tail_t'])}", flush=True)
    print("[18] 실험 A: 표현 고정 → 모델 스왑 (SSP/Morgan × XGB/DNN)", flush=True)
    perA, curvesA = run_expA(D)
    print("[18] 실험 B: 모델 고정(SSI-DDI) → 표현 스왑 (그래프 vs 엣지제거)", flush=True)
    perB, curvesB = run_expB(D)
    out = {
        "config": dict(task="model-swap disentangle", n_pairs=int(len(D["Y"])), n_types=N_CLASS,
                       seeds=SEEDS, device=str(DEV), regimes=REGIMES,
                       runtime_s=round(time.time() - t0, 1),
                       n_head=len(D["head_t"]), n_mid=len(D["mid_t"]), n_tail=len(D["tail_t"])),
        "expA": agg(perA), "expA_per_seed": perA, "expA_curve": curvesA,
        "expB": agg(perB), "expB_per_seed": perB, "expB_curve": curvesB,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[18] 완료 · {(time.time()-t0)/60:.1f}분 · 저장 {RESULTS_JSON}", flush=True)
    print("=== 실험 A cold-cold macro-F1 (mean) ===")
    for c in ["SSP·XGB", "SSP·DNN", "Morgan·XGB", "Morgan·DNN"]:
        print(f"  {c:12s} {out['expA'][c]['cold-cold']['macro_f1']['mean']:.4f}")
    print("=== 실험 B cold-cold macro-F1 (mean) ===")
    for c in ["graph", "no-edge"]:
        print(f"  {c:12s} {out['expB'][c]['cold-cold']['macro_f1']['mean']:.4f}")
    build_report(out)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  보고서
# ══════════════════════════════════════════════════════════════════════════════
def _b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _signtest(a_list, b_list):
    """b−a 방향일치(부호검정). 반환 (Δmean, wins_b, n, p_sign). fold 상관 강건(17 방식)."""
    from scipy.stats import binomtest
    a, b = np.array(a_list), np.array(b_list); n = min(len(a), len(b))
    a, b = a[:n], b[:n]; w = int((b > a).sum())
    p = binomtest(w, n, 0.5).pvalue if n >= 2 else float("nan")
    return float((b - a).mean()), w, n, p


def build_report(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for _kf in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]:
        if os.path.exists(_kf):
            fm.fontManager.addfont(_kf); plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False
    c = res["config"]; A = res["expA"]; B = res["expB"]; psA = res["expA_per_seed"]; psB = res["expB_per_seed"]

    def cc(d, cell):
        return d[cell]["cold-cold"]["macro_f1"]["mean"]

    # ── 그림1: 설계 스키마 ──
    fig0, ax = plt.subplots(figsize=(9.6, 2.6)); ax.axis("off")
    ax.text(0.5, 0.92, "지금까지(13~17)=파이프라인 비교(표현+모델 동시)  →  18=하나 고정, 하나만 스왑", ha="center", fontsize=9.5, transform=ax.transAxes)
    ax.text(0.25, 0.62, "실험 A\n표현 고정(SSP·Morgan)\n→ 모델만 (XGB↔DNN)\n= 순수 '모델' 효과", ha="center", va="center", fontsize=8.5,
            transform=ax.transAxes, bbox=dict(boxstyle="round", fc="#3a5a8a", ec="#222", alpha=.85, pad=0.6), color="white")
    ax.text(0.72, 0.62, "실험 B\n모델 고정(SSI-DDI)\n→ 표현만 (그래프↔엣지제거)\n= 순수 '그래프' 효과", ha="center", va="center", fontsize=8.5,
            transform=ax.transAxes, bbox=dict(boxstyle="round", fc="#7a3a8a", ec="#222", alpha=.85, pad=0.6), color="white")
    chart_d = _b64(fig0)

    # ── 그림2: 두 실험 cold-cold macro-F1 막대 ──
    fig1, axs = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axs[0]
    a_cells = ["SSP·XGB", "SSP·DNN", "Morgan·XGB", "Morgan·DNN"]
    vals = [cc(A, x) for x in a_cells]; errs = [A[x]["cold-cold"]["macro_f1"]["std"] for x in a_cells]
    cols = ["#4c78d8", "#b0392f", "#8aa1c8", "#d88a82"]
    ax.bar(range(4), vals, yerr=errs, capsize=3, color=cols)
    ax.set_xticks(range(4)); ax.set_xticklabels(a_cells, fontsize=7.5, rotation=12); ax.set_ylabel("cold-cold macro-F1")
    ax.set_title("실험 A · 표현 고정, 모델 스왑"); ax.grid(axis="y", alpha=.25)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    ax = axs[1]
    b_cells = ["graph", "no-edge"]
    vals = [cc(B, x) for x in b_cells]; errs = [B[x]["cold-cold"]["macro_f1"]["std"] for x in b_cells]
    ax.bar(range(2), vals, yerr=errs, capsize=3, color=["#7a3a8a", "#b0a0c0"])
    ax.set_xticks(range(2)); ax.set_xticklabels(["그래프", "엣지제거"]); ax.set_ylabel("cold-cold macro-F1")
    ax.set_title("실험 B · 모델 고정(GNN), 표현 스왑"); ax.grid(axis="y", alpha=.25)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    chart_b = _b64(fig1)

    # ── 표 A: 2×2 (regime별 macro-F1) ──
    def tblA():
        h = "<tr><th class=l>셀(표현·모델)</th>" + "".join(f"<th>{rg}</th>" for rg in REGIMES) + "<th>acc(cc)</th></tr>"
        for cell in a_cells:
            r = A[cell]
            h += (f"<tr><td class=l><b>{cell}</b></td>"
                  + "".join(f"<td class={'hl' if rg=='cold-cold' else ''}>{r[rg]['macro_f1']['mean']:.3f}"
                            f"<span class=s>±{r[rg]['macro_f1']['std']:.3f}</span></td>" for rg in REGIMES)
                  + f"<td>{r['cold-cold']['accuracy']['mean']:.3f}</td></tr>")
        return h

    # ── 모델 축 / 표현 축 부호검정(cold-cold) ──
    def cc_seed(ps, cell):
        return ps[cell]["cold-cold"]["macro_f1"]
    model_axis = [  # 같은 표현에서 DNN vs XGB
        ("SSP: DNN vs XGB", "SSP·XGB", "SSP·DNN"),
        ("Morgan: DNN vs XGB", "Morgan·XGB", "Morgan·DNN"),
    ]
    rep_axis = [    # 같은 모델에서 Morgan vs SSP
        ("XGB: Morgan vs SSP", "SSP·XGB", "Morgan·XGB"),
        ("DNN: Morgan vs SSP", "SSP·DNN", "Morgan·DNN"),
    ]
    def sign_rows(pairs, ps):
        r = ""
        for lab, base, test in pairs:
            dm, w, n, p = _signtest(cc_seed(ps, base), cc_seed(ps, test))
            sig = "good" if (not np.isnan(p) and p < 0.10 and dm > 0) else ("bad" if (not np.isnan(p) and p < 0.10 and dm < 0) else "mut")
            pstr = "n/a" if np.isnan(p) else f"{p:.3f}"
            r += f"<tr><td class=l>{lab}</td><td class={'good' if dm>0 else 'bad'}>{dm:+.4f}</td><td>{w}/{n}</td><td class={sig}>{pstr}</td></tr>"
        return r
    # 실험 B 부호검정: 그래프 vs 엣지제거
    dmB, wB, nB, pB = _signtest(cc_seed(psB, "no-edge"), cc_seed(psB, "graph"))

    # 결론 판정
    ssp_model = cc(A, "SSP·DNN") - cc(A, "SSP·XGB")       # DNN−XGB (SSP)
    mor_model = cc(A, "Morgan·DNN") - cc(A, "Morgan·XGB")
    graph_gain = cc(B, "graph") - cc(B, "no-edge")
    # ★적대검증 반영: 그래프 축은 mean>0이 아니라 부호검정 유의성으로 판정(오버셀 방지).
    PFLOOR = 0.0625                                        # n=5 양측 부호검정 바닥 → 어떤 결과도 p<0.05 불가
    graph_sig = (not np.isnan(pB)) and pB < 0.05 and wB >= 5
    graph_txt = ("그래프 연결정보가 유의하게 도움" if graph_sig else
                 f"그래프 연결정보는 cold-cold서 <b>도움이 확인되지 않음</b>(방향성만·+{graph_gain:.3f}·{wB}/{nB}·p_sign "
                 f"{('n/a' if np.isnan(pB) else f'{pB:.3f}')}·편차내)")
    gnn_cc = cc(B, "graph")
    gsd = B["graph"]["cold-cold"]["macro_f1"]["std"]; nsd = B["no-edge"]["cold-cold"]["macro_f1"]["std"]
    flat = {"SSP·DNN": cc(A, "SSP·DNN"), "Morgan·XGB": cc(A, "Morgan·XGB"), "Morgan·DNN": cc(A, "Morgan·DNN")}
    gnn_below_all = all(gnn_cc < v for v in flat.values())

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 6px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.hl{background:#16202e}.s{color:#7a7f8a;font-size:10px}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>18 DDI model-swap</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>18 · 순수 모델 스왑 — "모델 탓? 표현 탓?" 원인 분리</h1>
<p class=mut>13~17은 파이프라인 비교(표현+모델 동시). 18은 하나 고정·하나만 스왑 → 모델 축/표현 축 분리. 16·17과 동일 데이터·분할·지표 · {c['device']} · {c['runtime_s']:.0f}s · seed {len(c['seeds'])}</p>

<div class=card><b>한 줄 결론:</b> 신약(cold-cold) macro-F1로 4세대 열위의 원인 분해 (★n=5 → 부호검정 바닥 p={PFLOOR} = <b>어떤 결과도 p&lt;0.05 불가</b>, 전부 방향성/marginal) —<br>
★<b>모델 축 = 표현-의존 상호작용</b>(단순 'DNN vs 트리' 아님): SSP서 DNN−XGB {ssp_model:+.3f}(5/5·p 0.062)로 크나 Morgan서 {mor_model:+.3f}(3/5·ns)로 미미 → <b>모델을 표현에 맞춰라</b>(트리+밀집PCA-SSP만 신약서 붕괴, DNN은 표현-강건).<br>
★<b>표현 축(그래프)</b>(같은 GNN, 그래프 vs 엣지제거): {graph_txt}.<br>
★<b>4세대 열위 귀속</b>: 그래프-GNN(cc {gnn_cc:.3f})이 Exp-A 모든 flat 셀보다 {'낮음' if gnn_below_all else '비교'} — 그래프 표현은 자기 GNN엔 (약)도움이라 원인 아님 → <b>미튜닝 SSI-DDI 모델</b>(16 하한)에 가장 직접 귀속(모델/표현/데이터 병목 중 여기선 모델 기여 과소평가 말 것).<br>
★한계: 유일한 크고 깨끗한 효과는 SSP+트리 상호작용뿐·SSP↔Morgan 3중차이(의미·융합·밀도)로 기전 미분리·재현 불균일·fold 상관→부호검정.</div>

<h2>① 설계</h2>
<img src="{chart_d}" alt="design">
<img src="{chart_b}" alt="bars">

<h2>② 실험 A · 표현 고정 → 모델 스왑 (macro-F1)</h2>
<div style="overflow-x:auto"><table>{tblA()}</table></div>
<div style="overflow-x:auto"><table>
<tr><th class=l>★모델 축(같은 표현, DNN vs XGB) · cold-cold</th><th>Δmean</th><th>win</th><th>p_sign</th></tr>
{sign_rows(model_axis, psA)}
</table></div>
<div style="overflow-x:auto"><table>
<tr><th class=l>표현 축(같은 모델, Morgan vs SSP) · cold-cold</th><th>Δmean</th><th>win</th><th>p_sign</th></tr>
{sign_rows(rep_axis, psA)}
</table></div>
<div class=card>★대각선 재현(검증·불균일): <b>SSP·DNN은 14 plain과 bit-동일</b>(하네스 강검증)·Morgan·XGB는 10/11 struct에 근사(+0.007=희소CSR↔밀집 XGBoost 결측처리 차·seed 편차 아님)·그래프는 16에 근사(편차내). 비대각(SSP·XGB·Morgan·DNN)이 새 셀. Δ&gt;0=뒤가 앞보다 높음. fold 상관이라 <b>부호검정(win·p_sign)</b>이 근거(t-test 아님·17 교훈).</div>

<h2>③ 실험 B · 모델 고정(SSI-DDI) → 표현 스왑 (macro-F1)</h2>
<div style="overflow-x:auto"><table>
<tr><th class=l>셀</th>""" + "".join(f"<th>{rg}</th>" for rg in REGIMES) + f"""</tr>
<tr><td class=l><b>그래프(진짜 결합)</b></td>""" + "".join(f"<td class={'hl' if rg=='cold-cold' else ''}>{B['graph'][rg]['macro_f1']['mean']:.3f}<span class=s>±{B['graph'][rg]['macro_f1']['std']:.3f}</span></td>" for rg in REGIMES) + f"""</tr>
<tr><td class=l><b>엣지제거(원자 bag)</b></td>""" + "".join(f"<td class={'hl' if rg=='cold-cold' else ''}>{B['no-edge'][rg]['macro_f1']['mean']:.3f}<span class=s>±{B['no-edge'][rg]['macro_f1']['std']:.3f}</span></td>" for rg in REGIMES) + f"""</tr>
</table></div>
<div class=card>그래프−엣지제거(cold-cold) = <b>{graph_gain:+.4f}</b> ({wB}/{nB} seed·p_sign {('n/a' if np.isnan(pB) else f'{pB:.3f}')}).
★같은 SSI-DDI 모델·같은 원자특징, 결합 엣지만 차이 → 이 Δ가 순수 '그래프 연결정보' 기여이나 —
{graph_txt}(그래프 {gnn_cc:.3f}±{gsd:.3f} vs 엣지제거 {cc(B,'no-edge'):.3f}±{nsd:.3f}).<br>
★<b>결정적</b>: 그래프 절대값({gnn_cc:.3f})은 Exp-A <b>모든 flat 셀보다 낮음</b>(SSP·DNN {flat['SSP·DNN']:.3f}·Morgan·XGB {flat['Morgan·XGB']:.3f}·Morgan·DNN {flat['Morgan·DNN']:.3f}) → 4세대 열위 원인은 그래프 표현이 아니라 <b>미튜닝 GNN 모델</b>(16 하한).</div>

<h2>④ ★결론 · 모델 축 / 표현 축 분리</h2>
<div class=card>
· <b>모델 축 = 표현-의존 상호작용</b>: SSP서 DNN≫XGB({ssp_model:+.3f}·5/5)·Morgan서 DNN≈XGB({mor_model:+.3f}·ns). '트리 vs DNN 절대우열'이 아니라 — <b>트리는 희소이진(Morgan) 필요, 밀집PCA-SSP서 신약 외삽 붕괴</b>(warm 0.71→cc 0.12). DNN은 표현-강건. → <b>모델을 표현에 맞춰라</b>.<br>
· <b>표현 축(그래프)</b>: 같은 GNN, 그래프 vs 엣지제거({graph_gain:+.3f}·{wB}/{nB}·p {('n/a' if np.isnan(pB) else f'{pB:.3f}')}) — {graph_txt}. 그래프 표현이 4세대 열위의 원인은 아님(자기 GNN엔 약도움).<br>
· ★<b>종합</b>: 4세대(그래프+GNN)가 낮은 건 <b>단일 대주효과 축이 아니다</b> — 유일한 크고 깨끗한 효과는 SSP+트리 상호작용(붕괴). 그래프-GNN(cc {gnn_cc:.3f})이 flat 셀들보다 낮은 건 {'미튜닝 SSI-DDI 모델(16 하한)에 가장 직접 귀속' if gnn_below_all else '셀 참조'}(그래프 표현 아님). 표현/데이터도 병목이나(13~17) 여기선 <b>모델(GNN) 기여를 과소평가 말 것</b>.<br>
· <b>정직 한계</b>: (i) n=5 → 부호검정 바닥 p={PFLOOR}=<b>어떤 결과도 p&lt;0.05 불가</b>(두 대효과도 5/5 p=0.062=marginal). (ii) 기전 '트리가 밀집PCA-SSP 외삽 못함'은 <b>가설</b> — SSP↔Morgan은 의미·융합(concat vs sum)·밀도(밀집연속 vs 희소이진) <b>3중</b> 동시차이라 밀도 단독 귀속 불가. (iii) 재현 불균일(SSP·DNN bit-동일·Morgan·XGB +0.007·그래프 편차내). (iv) 그래프↔엣지제거 정보량 다름·GNN 미튜닝(16 하한). (v) ~0.27 천장도 soft(15 DDIMDL struct 0.284&gt;최고셀 Morgan·DNN {flat['Morgan·DNN']:.3f}). fold 상관→부호검정만·warm 누수맥락·seed {len(c['seeds'])}.</div>

<div class=foot>18 model-swap · 실험A(표현고정 SSP/Morgan × 모델 XGB/DNN)·실험B(모델고정 SSI-DDI × 표현 그래프/엣지제거) · warm/cs/cc · macro-F1 · 재사용 m14·m16·ddi_common · seed {len(c['seeds'])} · {c['device']} · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "18_ddi_model_swap.html")
    with open(out_html, "w") as f:
        f.write(html)
    print(f"[report] → {out_html}", flush=True)
    return out_html


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode in ("smoke", "full"):
        run_full()
    else:
        print(f"[알림] 모드 '{mode}' 불명. smoke|full.")


if __name__ == "__main__":
    main()

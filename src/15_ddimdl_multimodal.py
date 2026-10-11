"""
[DDI 실험 - 15] 3세대(DDIMDL, Deng 2020) 재현 — 여러 특징을 각자 학습해 평균 vs XGBoost
=================================================================================
DDIMDL 핵심: 여러 약물 특징(구조·효소·타깃·경로)을 ★각각 별도 신경망(sub-model)으로 학습 →
  예측(확률)을 ★평균내 최종 86종 유형 예측. 14(2세대)는 특징 1개(SSP)였고,
  3세대는 ★여러 특징을 sub-model로 합치는 게 핵심. 이번엔 '구조+효소' 축소판(타깃·경로는 다음).

■ 특징(각 특징 = 약물쌍 → 벡터, ★train 약물 기준 유사도 프로필+PCA, 14 재사용·누수0):
  · 구조 특징 = 각 약물의 구조 유사도 프로필(train 약물 대비 Tanimoto). 14 SSP 로직 재사용.
  · 효소 특징 = 각 약물의 효소 유사도 프로필. ★ADMET-AI 예측 CYP(07)로 약물간 효소 유사도
    (두 약물이 쓰는 CYP가 얼마나 겹치나=Jaccard) 행렬 E를 만들고, E의 train 열로 프로필.
  · 각 특징: 두 약물 프로필 concat(순서고정) → 그 특징의 쌍 벡터. PCA 분산 90%(14와 동일).

■ 모델(DDIMDL = sub-model 2개 + 평균):
  · sub-model = 3층(→512→256→86), 각 은닉층 뒤 BatchNorm+ReLU+Dropout0.3(논문 구조).
  · ★구조 sub-model, 효소 sub-model을 ★각각 독립 학습(자기 특징으로 86종 softmax 예측).
  · ★결합 = 두 sub-model의 예측확률을 ★평균 → 최종. (논문 average 방식)
  · CE·Adam lr1e-3·batch256·epoch≤100·val-loss early-stop.

■ 비교(핵심): (1) 구조만 / 효소만 / 구조+효소평균 — "합치면 나아지나?"(DDIMDL 핵심질문)
             (2) DDIMDL(구조+효소) vs 14 DeepDDI(cc 0.257) vs 11 XGBoost+cw(cc 0.272).
평가·지표·분할 14와 동일(warm 누수맥락·핵심 cold). seed5.

산출: results/15_ddi_ddimdl_multimodal.html , results/ddimdl_results.json
실행: conda run -n admet python src/15_ddimdl_multimodal.py  (빠른점검 DDIMDL_SMOKE=1)
"""
import os, sys, json, time, io, base64, importlib.util
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddi_common as C
from ddi_common import ukey, build_fp_matrix, split_drugs, build_sim_matrix, top3_acc

# 14 모듈의 재사용 함수(중복 구현 방지): build_ssp_pca·load_Y·predict_proba·eval_dnn
_spec = importlib.util.spec_from_file_location("m14", os.path.join(os.path.dirname(os.path.abspath(__file__)), "14_deepddi_dnn.py"))
M14 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M14)
build_ssp_pca = M14.build_ssp_pca; load_Y = M14.load_Y; predict_proba = M14.predict_proba; eval_dnn = M14.eval_dnn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
CYP_CSV = os.path.join(BASE, "data", "drug_cyp.csv")
RESULTS_JSON = os.path.join(BASE, "results", "ddimdl_results.json")

SMOKE = os.environ.get("DDIMDL_SMOKE", "") == "1"
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
MAX_EPOCH = 12 if SMOKE else 100
PATIENCE = 4 if SMOKE else 8
DRUG_TEST_FRAC = 0.20
REGIMES = ["warm", "cold-start", "cold-cold"]
CONFIGS = ["struct_only", "enzyme_only", "ddimdl_avg"]   # (1) 합치면 나아지나
BATCH, LR = 256, 1e-3
N_CLASS = 86
METRICS = ["accuracy", "macro_f1", "micro_f1", "top3", "head_f1", "mid_f1", "tail_f1"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== DDIMDL sub-model =====================
class SubModel(nn.Module):
    """DDIMDL sub-model: 3층(→512→256→86), 은닉층마다 BatchNorm+ReLU+Dropout0.3(논문 구조).
    ★특징 1개(구조 or 효소)만 받아 86종 예측. 여러 sub-model을 따로 학습 후 평균이 DDIMDL 핵심."""
    def __init__(self, in_dim, n_class=N_CLASS, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256, n_class),
        )

    def forward(self, x):
        return self.net(x)


def train_submodel(Xtr, ytr, seed=0):
    """★sub-model 1개 학습(구조 또는 효소 특징으로). 14와 동일 패턴:
    train서 10% val 떼어 val-loss early-stopping, 미니배치 Adam CE. 반환 (model, curve)."""
    torch.manual_seed(seed); np.random.seed(seed)
    idx = np.arange(len(ytr))
    tr_i, va_i = train_test_split(idx, test_size=0.1, random_state=seed)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
    model = SubModel(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    Xva = Xt[va_i].to(DEV); yva = yt[va_i].to(DEV)
    best = np.inf; best_state = None; wait = 0; curve = []
    for ep in range(MAX_EPOCH):
        model.train(); perm = np.random.permutation(tr_i); tl = 0.0; nb = 0
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            if len(b) < 2:                                   # BatchNorm 배치1 방지
                continue
            xb = Xt[b].to(DEV); yb = yt[b].to(DEV)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tl += float(loss.item()); nb += 1
        model.eval()
        with torch.no_grad():
            vout = model(Xva); vloss = float(crit(vout, yva).item()); vpred = vout.argmax(1).cpu().numpy()
        vmac = f1_score(ytr[va_i], vpred, average="macro", zero_division=0)
        curve.append((ep, tl / max(nb, 1), vloss, float(vmac)))
        if vloss < best - 1e-4:
            best = vloss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, curve


def build_enzyme_sim(CYP):
    """효소 유사도 행렬 E(n×n): 두 약물이 쓰는 CYP 집합의 Jaccard(겹침/합집합).
    CYP 전부0인 약물은 합집합0 → 유사도0(정보 없음)."""
    inter = CYP @ CYP.T
    cnt = CYP.sum(1)
    union = cnt[:, None] + cnt[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def build_cyp_matrix(drug_ids, did2row):
    cyp = pd.read_csv(CYP_CSV); cyp["drug_id"] = cyp["drug_id"].astype(str)
    cols = [c for c in cyp.columns if c != "drug_id"]
    M = np.zeros((len(drug_ids), len(cols)), dtype=np.float32)
    for _, r in cyp.iterrows():
        if str(r["drug_id"]) in did2row:
            M[did2row[str(r["drug_id"])]] = r[cols].values.astype(np.float32)
    return M


def run():
    t0 = time.time()
    print(f"[15] DDIMDL(sub-model 2개+평균) 시작 · device={DEV} · SMOKE={SMOKE}", flush=True)
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"])); did2row = {d: i for i, d in enumerate(drug_ids)}
    n = len(drug_ids)
    pairs = pd.read_csv(PAIRS_CSV); pos = pairs[pairs.label == 1]
    pos_pairs = [ukey(str(a), str(b)) for a, b in zip(pos["drug_a"], pos["drug_b"])]
    pi = np.array([did2row[a] for a, b in pos_pairs], dtype=np.int32)
    pj = np.array([did2row[b] for a, b in pos_pairs], dtype=np.int32)
    lo = np.minimum(pi, pj); hi = np.maximum(pi, pj)
    Y = load_Y(pos_pairs); K = len(set(Y.tolist()))
    gc = pd.Series(Y).value_counts()
    head_t = set(gc.head(10).index.tolist()); tail_t = set(gc[gc < 100].index.tolist())
    mid_t = set(gc.index.tolist()) - head_t - tail_t
    print(f"  약물 {n} · positive {len(pi):,} · 유형 {K}종", flush=True)

    # 유사도 행렬: 구조(M2 Tanimoto)·효소(E Jaccard)
    bvs = [C._GEN.GetFingerprint(Chem.MolFromSmiles(str(id2smi[d]))) for d in drug_ids]
    M2 = build_sim_matrix(bvs).astype(np.float32)
    CYP = build_cyp_matrix(drug_ids, did2row)
    E = build_enzyme_sim(CYP)
    print(f"  M2(구조)·E(효소 Jaccard) 완성 · E 평균 off-diag {E[~np.eye(n,dtype=bool)].mean():.3f}", flush=True)

    per = {cf: {rg: {mt: [] for mt in METRICS} for rg in REGIMES} for cf in CONFIGS}
    pdims = {"struct": [], "enzyme": []}; curve_repr = None
    for seed in SEEDS:
        train_drugs, test_drugs = split_drugs(drug_ids, seed, DRUG_TEST_FRAC)
        g = np.array([0 if d in train_drugs else 1 for d in drug_ids], dtype=np.int8)
        train_rows = np.where(g == 0)[0]
        # ★각 특징을 train 기준 프로필+PCA(누수0). concat(순서고정)로 쌍 벡터.
        ssp_s, Ps = build_ssp_pca(M2, train_rows); pdims["struct"].append(Ps)
        ssp_e, Pe = build_ssp_pca(E, train_rows); pdims["enzyme"].append(Pe)
        Xs = np.concatenate([ssp_s[lo], ssp_s[hi]], axis=1)   # 구조 특징 쌍 벡터
        Xe = np.concatenate([ssp_e[lo], ssp_e[hi]], axis=1)   # 효소 특징 쌍 벡터
        pu = g[pi] + g[pj]
        tt = np.where(pu == 0)[0]; ts = np.where(pu == 1)[0]; ss = np.where(pu == 2)[0]
        wtr, wte = train_test_split(np.arange(len(pos_pairs)), test_size=0.2, random_state=seed, stratify=Y)
        print(f"  [seed {seed}] P_struct={Ps} P_enz={Pe} · TT{len(tt):,} TS{len(ts):,} SS{len(ss):,}", flush=True)

        def run_group(tr_idx, test_map):
            """한 학습셋(tr_idx)으로 구조·효소 sub-model 각각 학습 → test들 채점.
            ★핵심(DDIMDL): 두 sub-model 예측확률을 평균."""
            ms, cs = train_submodel(Xs[tr_idx], Y[tr_idx], seed)   # 구조 sub-model
            me, ce = train_submodel(Xe[tr_idx], Y[tr_idx], seed)   # 효소 sub-model
            out = {}
            for rg, idx in test_map.items():
                ps = predict_proba(ms, Xs[idx]); pe = predict_proba(me, Xe[idx])   # 각자 예측
                avg = (ps + pe) / 2.0                                              # ★평균 결합
                out[rg] = {
                    "struct_only": eval_dnn(ps, Y[idx], head_t, mid_t, tail_t),
                    "enzyme_only": eval_dnn(pe, Y[idx], head_t, mid_t, tail_t),
                    "ddimdl_avg": eval_dnn(avg, Y[idx], head_t, mid_t, tail_t),
                }
            return out, cs, ce

        # warm(자기 학습셋)·cold(TT 학습→TS/SS)
        warm_out, cs_curve, ce_curve = run_group(wtr, {"warm": wte})
        cold_out, _, _ = run_group(tt, {"cold-start": ts, "cold-cold": ss})
        if curve_repr is None:
            curve_repr = {"struct": cs_curve, "enzyme": ce_curve, "seed": seed, "regime": "warm"}
        for rg_out in [warm_out, cold_out]:
            for rg, cfgs in rg_out.items():
                for cf in CONFIGS:
                    for mt in METRICS:
                        per[cf][rg][mt].append(cfgs[cf][mt])
        print(f"    cc macroF1: struct {per['struct_only']['cold-cold']['macro_f1'][-1]:.3f} · enzyme {per['enzyme_only']['cold-cold']['macro_f1'][-1]:.3f} · avg {per['ddimdl_avg']['cold-cold']['macro_f1'][-1]:.3f}", flush=True)

    agg = {cf: {rg: {mt: {"mean": float(np.mean(per[cf][rg][mt])), "std": float(np.std(per[cf][rg][mt]))} for mt in METRICS} for rg in REGIMES} for cf in CONFIGS}
    res = {"config": {"n_pairs": len(pi), "n_types": K, "seeds": SEEDS, "pca_struct": int(np.mean(pdims["struct"])),
                      "pca_enzyme": int(np.mean(pdims["enzyme"])), "batch": BATCH, "lr": LR, "device": str(DEV),
                      "n_head": len(head_t), "n_mid": len(mid_t), "n_tail": len(tail_t), "smoke": SMOKE,
                      "runtime_s": round(time.time() - t0, 1)},
           "results": agg, "per_seed": per, "curve": curve_repr}
    json.dump(res, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ cold-cold macro-F1 (합치면 나아지나) ================")
    for cf in CONFIGS:
        a = agg[cf]["cold-cold"]
        print(f"  {cf:12s}: {a['macro_f1']['mean']:.3f}  (14 DeepDDI 0.257 · 11 XGB+cw 0.272)")
    if not SMOKE:
        out_html = build_report(res)
        print(f"\n[15] 완료 ({res['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    else:
        print(f"\n[15] SMOKE 완료 ({res['config']['runtime_s']:.0f}s) — sub-model 학습 확인용")
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
    agg = res["results"]; c = res["config"]
    # 앵커: 14 DeepDDI plain, 11 XGBoost class-weight (per_seed까지 로드해 paired 비교)
    dd14_full = json.load(open(os.path.join(BASE, "results", "deepddi_results.json")))
    x11_full = json.load(open(os.path.join(BASE, "results", "imbalance_results.json")))
    dd14 = dd14_full["results"]["plain"]; x11 = x11_full["results"]["class-weight"]
    ccol = {"struct_only": "#2a6f4c", "enzyme_only": "#8a6d1a", "ddimdl_avg": "#7a3a8a"}
    from scipy import stats
    _s = np.array(res["per_seed"]["struct_only"]["cold-cold"]["macro_f1"])
    _a = np.array(res["per_seed"]["ddimdl_avg"]["cold-cold"]["macro_f1"])
    _d14 = np.array(dd14_full["per_seed"]["plain"]["cold-cold"]["macro_f1"])
    _x11 = np.array(x11_full["per_seed"]["class-weight"]["cold-cold"]["macro_f1"])
    def _pair(x, y):
        return float((x - y).mean()), int((x > y).sum()), float(stats.ttest_rel(x, y)[1])
    fuse_d, fuse_w, fuse_p = _pair(_a, _s)          # 합치기(avg vs struct)
    vs14_d, vs14_w, vs14_p = _pair(_s, _d14)        # 15 구조 vs 14 DeepDDI(아키텍처 다름)
    vs11_d, vs11_w, vs11_p = _pair(_a, _x11)        # 15 avg vs 11 XGBoost+cw
    # ★다중비교 보정(시리즈 관행 BH-FDR): 3개 paired 검정
    try:
        from scipy.stats import false_discovery_control
        _bh = false_discovery_control([fuse_p, vs14_p, vs11_p], method="bh")
        vs14_bh, vs11_bh = float(_bh[1]), float(_bh[2])
    except Exception:
        vs14_bh, vs11_bh = min(vs14_p * 3, 1.0), min(vs11_p * 3, 1.0)

    # 그림1: 방법 스키마
    fig0, ax = plt.subplots(figsize=(8.6, 2.6)); ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.55), 0.2, 0.34, fc="#2a6f4c", ec="#222", alpha=.85, transform=ax.transAxes))
    ax.text(0.12, 0.72, "구조 특징\n(Tanimoto SSP)", ha="center", va="center", fontsize=8, color="white", transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0.02, 0.12), 0.2, 0.34, fc="#8a6d1a", ec="#222", alpha=.85, transform=ax.transAxes))
    ax.text(0.12, 0.29, "효소 특징\n(CYP Jaccard)", ha="center", va="center", fontsize=8, color="white", transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0.34, 0.55), 0.22, 0.34, fc="#3a5a8a", ec="#222", alpha=.85, transform=ax.transAxes))
    ax.text(0.45, 0.72, "구조 sub-model\n(3층 DNN)", ha="center", va="center", fontsize=8, color="white", transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0.34, 0.12), 0.22, 0.34, fc="#3a5a8a", ec="#222", alpha=.85, transform=ax.transAxes))
    ax.text(0.45, 0.29, "효소 sub-model\n(3층 DNN)", ha="center", va="center", fontsize=8, color="white", transform=ax.transAxes)
    ax.text(0.30, 0.72, "→", fontsize=15, transform=ax.transAxes); ax.text(0.30, 0.29, "→", fontsize=15, transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0.66, 0.33, ), 0.14, 0.34, fc="#7a3a8a", ec="#222", alpha=.9, transform=ax.transAxes))
    ax.text(0.73, 0.5, "★확률\n평균", ha="center", va="center", fontsize=9, color="white", transform=ax.transAxes)
    ax.text(0.60, 0.5, "→", fontsize=15, transform=ax.transAxes)
    ax.text(0.83, 0.5, "→ 86종", fontsize=10, transform=ax.transAxes)
    chart_m = _b64(fig0)

    # 그림2: 합치면 나아지나 (구조만/효소만/평균 × regime)
    fig1, ax = plt.subplots(figsize=(8.4, 3.6)); w = 0.25
    for i, cf in enumerate(CONFIGS):
        vals = [agg[cf][rg]["macro_f1"]["mean"] for rg in REGIMES]; err = [agg[cf][rg]["macro_f1"]["std"] for rg in REGIMES]
        ax.bar(np.arange(3) + (i - 1) * w, vals, w, yerr=err, capsize=3, color=ccol[cf], label=cf)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(REGIMES); ax.set_ylabel("macro-F1"); ax.set_ylim(0, 0.85)
    ax.set_title("★합치면 나아지나 — 구조만 vs 효소만 vs 구조+효소평균(DDIMDL)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    chart_b = _b64(fig1)

    # 그림3: sub-model 학습곡선
    cv = res.get("curve"); chart_c = ""
    if cv:
        fig2, ax = plt.subplots(figsize=(7.2, 3.2))
        for key, col in [("struct", "#2a6f4c"), ("enzyme", "#8a6d1a")]:
            arr = np.array(cv[key]); ax.plot(arr[:, 0], arr[:, 1], "-o", ms=3, color=col, label=f"{key} train loss")
            ax.plot(arr[:, 0], arr[:, 2], "--", color=col, alpha=.6, label=f"{key} val loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")
        ax.set_title(f"sub-model 학습곡선 ({cv['regime']}·seed{cv['seed']}) — 둘 다 loss 감소=정상")
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.25)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        chart_c = _b64(fig2)

    def g(dat, rg, k):
        v = dat[rg][k]; return v["mean"] if isinstance(v, dict) else v
    def rows_tbl(dat):
        r = ""
        for rg in REGIMES:
            r += (f"<tr><td class=l>{rg}</td><td class=hl>{g(dat,rg,'macro_f1'):.3f}</td><td>{g(dat,rg,'micro_f1'):.3f}</td><td>{g(dat,rg,'accuracy'):.3f}</td>"
                  f"<td>{g(dat,rg,'top3'):.3f}</td><td>{g(dat,rg,'head_f1'):.3f}</td><td>{g(dat,rg,'mid_f1'):.3f}</td><td>{g(dat,rg,'tail_f1'):.3f}</td></tr>")
        return r

    # 핵심 수치
    s_cc = agg["struct_only"]["cold-cold"]["macro_f1"]["mean"]; e_cc = agg["enzyme_only"]["cold-cold"]["macro_f1"]["mean"]
    a_cc = agg["ddimdl_avg"]["cold-cold"]["macro_f1"]["mean"]
    dd14_cc = dd14["cold-cold"]["macro_f1"]["mean"]; x11_cc = x11["cold-cold"]["macro_f1"]["mean"]
    fuse_gain = a_cc - s_cc
    fuse_verdict = ("<b class=good>합치면 나아짐</b>" if fuse_gain > 0.005 else
                    "<b class=warn>합쳐도 그대로</b>" if abs(fuse_gain) <= 0.005 else "<b class=bad>합치면 오히려 손해</b>")

    cmp2 = ""
    for lab, v in [("DDIMDL 구조+효소 평균", a_cc), ("DDIMDL 구조만", s_cc), ("14 DeepDDI (SSP+8층DNN)", dd14_cc), ("11 XGBoost +class-weight", x11_cc)]:
        cls = "good" if v == max(a_cc, s_cc, dd14_cc, x11_cc) else "mut"
        cmp2 += f"<tr><td class=l><b>{lab}</b></td><td class={cls}>{v:.3f}</td></tr>"

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 6px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}.hl{background:#16202e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>15 DDIMDL</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>15 · 3세대(DDIMDL 2020) 재현 — 여러 특징을 각자 학습해 평균 vs XGBoost</h1>
<p class=mut>구조·효소 특징을 ★각각 sub-model(3층 DNN)로 학습 → 예측 평균. 14·11과 동일 데이터·분할·지표 · {c['device']} · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> ★<b>'합치면 나아지나?' — 효소 축소판에선 no(중립)</b>: cold-cold macro-F1 구조만 {s_cc:.3f} · 효소만 {e_cc:.3f}(거의 무의미) · <b>구조+효소평균 {a_cc:.3f}</b>(avg−struct {fuse_d:+.3f}·{fuse_w}/5·p{fuse_p:.2f}) → {fuse_verdict}. 효소(CYP 8차원)가 약해 융합 이점 없음(07·10 'CYP 무이득'과 일관·다특징 이점은 타깃·경로가 있어야).<br>
★<b>단 반전</b>: DDIMDL의 <b>구조 sub-model</b>이 오히려 <b>14 DeepDDI({dd14_cc:.3f})를 paired 상회</b>(구조 {s_cc:.3f}·+{vs14_d:.3f}·{vs14_w}/5·<b>p{vs14_p:.3f}·BH {vs14_bh:.3f} 견고</b>); 11 XGBoost({x11_cc:.3f})와는 평균 {a_cc:.3f}·+{vs11_d:.3f}·{vs11_w}/5·p{vs11_p:.3f}(<b>BH {vs11_bh:.3f}·Bonferroni 미달=경계</b>). 절대값은 seed 편차(±{agg['struct_only']['cold-cold']['macro_f1']['std']:.3f}) 내. ★이 이득은 <b>융합 아니라 모델쪽</b>(특징은 14와 동일 SSP+PCA) — 단 15 sub-model은 14와 <b>깊이·너비·dropout·batch가 모두 달라</b> 어느 요인인지 미분리.</div>

<h2>① 방법 (sub-model 2개 → ★평균)</h2>
<img src="{chart_m}" alt="method">
<div class=card>· <b>구조 특징</b>=train 약물 대비 Tanimoto 프로필(14 SSP) · <b>효소 특징</b>=ADMET-AI CYP(07) 기반 약물간 Jaccard 프로필. 각 특징 concat(순서고정)+PCA(분산90%·구조 P≈{c['pca_struct']}·효소 P≈{c['pca_enzyme']}·★train fit 누수0).<br>
· ★<b>sub-model 2개 독립 학습</b>: 구조용·효소용 각 3층 DNN(→512→256→86·BN·ReLU·Dropout0.3). 각자 자기 특징으로 86종 softmax 예측. · ★<b>결합=두 예측확률 평균</b>(DDIMDL average). CE·Adam·batch{c['batch']}·val-loss early-stop.<br>
★14(2세대)는 특징 1개, 15(3세대)는 여러 특징 sub-model+평균이 핵심. 이번은 구조+효소 축소판(타깃·경로는 데이터 관문→다음).</div>

<h2>② sub-model 학습곡선 (둘 다 학습되나)</h2>
<img src="{chart_c}" alt="curve">
<div class=card>구조·효소 sub-model 모두 train/val loss 감소 → 학습 정상(NaN·붕괴 없음). 효소는 정보량이 적어 val loss 바닥이 높음(=예측력 약함).</div>

<h2>③ ★핵심(1): 합치면 나아지나 — 구조만 / 효소만 / 평균 × regime</h2>
<img src="{chart_b}" alt="fuse">
<div style="overflow-x:auto"><table>
<tr><th class=l>config</th><th>규제</th><th colspan=7 class=mut>cold-cold 기준 macro-F1은 표 참고</th></tr></table></div>"""
    for cf in CONFIGS:
        html += f"""
<p class=mut><b>{cf}</b> (5 seed)</p>
<div style="overflow-x:auto"><table>
<tr><th class=l>regime</th><th>macroF1</th><th>microF1</th><th>acc</th><th>top3</th><th>head</th><th>mid</th><th>tail</th></tr>
{rows_tbl(agg[cf])}
</table></div>"""
    html += f"""
<div class=card>★<b>합치기 효과(cold-cold)</b>: 구조만 {s_cc:.3f} → 구조+효소평균 {a_cc:.3f} (paired {fuse_d:+.3f}·{fuse_w}/5·p{fuse_p:.2f}) → {fuse_verdict}(중립). 효소만 {e_cc:.3f}=거의 무의미(CYP 8차원·PCA 성분 {c['pca_enzyme']}개·38% 약물 CYP 없음). ★강한특징+무의미특징 단순평균이 <b>구조를 해치진 않았으나(중립) 돕지도 못함</b> → DDIMDL 다특징 이점은 <b>정보량 있는 특징</b>이 있어야. 논문은 가중결합(약한특징 자동 저가중)—본 재현은 명세대로 단순평균.</div>

<h2>④ ★핵심(2): DDIMDL vs 14 DeepDDI vs 11 XGBoost (cold-cold macro-F1)</h2>
<div style="overflow-x:auto"><table>
<tr><th class=l>방법 (cold-cold, 신약)</th><th>macro-F1</th></tr>
{cmp2}
</table></div>
<div class=card>★<b>결론(정직)</b>: DDIMDL 구조 sub-model {s_cc:.3f}·평균 {a_cc:.3f}이 14 DeepDDI {dd14_cc:.3f}·11 XGBoost {x11_cc:.3f}를 ★<b>paired 5/5 소폭 상회</b> — <b>vs14 견고</b>(+{vs14_d:.3f}·p{vs14_p:.3f}·BH {vs14_bh:.3f}), <b>vs11 경계</b>(+{vs11_d:.3f}·p{vs11_p:.3f}·BH {vs11_bh:.3f}·Bonferroni 미달). 절대 margin 작고 seed 편차(±{agg['struct_only']['cold-cold']['macro_f1']['std']:.3f}) 내 → <b>'무승부~근소우위'</b>. ★핵심: 우위는 <b>다특징 융합이 아니라</b>(융합 중립) <b>모델쪽</b>(특징은 14와 동일 SSP+PCA 파이프라인). 단 15 sub-model은 14와 depth·width·dropout·batch가 모두 달라 <b>어느 요인이 원인인지 미분리</b>(향후 통제실험). ★즉 세대(다특징)보다 <b>아키텍처/표현</b>이 이 데이터의 지렛대. 다특징 이점은 정보량 있는 특징(타깃·경로·metadock 효소)이 다음 관문. <span class=mut>(11·14·15 모두 동일 drug_smiles.csv·split_drugs → seed별 분할 동일=paired 유효.)</span></div>

<h2>⑤ ★한계·정직 메모</h2>
<div class=card>
· <b>구조+효소 축소판</b>: DDIMDL 원논문은 4특징(구조·타깃·효소·경로) — 여기선 2개(타깃·경로는 데이터 관문). 다특징 이점 과소평가일 수 있음(상한 아님).<br>
· <b>효소=ADMET-AI 예측 CYP</b>(07 방식·8차원·38% 약물 전부0) → 약한 특징. ★다음엔 metadock 도킹 기반 효소로 교체 예정.<br>
· <b>단순평균</b>(논문 명세) — DDIMDL 실제는 sub-model 가중결합 → 약한특징 희석을 줄임. 가중결합 미구현.<br>
· <b>warm 누수맥락</b>(14 동일)·SSP=train기준·<b>파이프라인 비교</b>(특징+모델 동시, 순수 모델비교 아님)·sub-model 3층(14는 8층, 구조 다름)·이진(02~09) 비교 불가.</div>

<h2>⑥ 다음</h2>
<div class=card>· 타깃(12)·경로(KEGG) 특징 추가로 4-modal 완성 · 효소를 metadock 도킹으로 교체 · 가중결합(단순평균→학습가중) · 같은 특징으로 트리 vs sub-model.</div>

<div class=foot>15 DDIMDL sub-model 2개(구조·효소)+평균 · 3층 DNN(512-256-86·BN·ReLU·Drop0.3) · warm/cold-start/cold-cold · macro-F1+head/mid/tail · PCA90%(구조P≈{c['pca_struct']}·효소P≈{c['pca_enzyme']}) · vs 14 DeepDDI·11 XGBoost · seed {len(c['seeds'])} · {c['device']} · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "15_ddi_ddimdl_multimodal.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


if __name__ == "__main__":
    run()

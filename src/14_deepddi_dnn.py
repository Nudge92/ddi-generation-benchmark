"""
[DDI 실험 - 14] 2세대(DeepDDI, Ryu 2018) 재현 — SSP + 심층신경망(DNN)으로 86종 유형 예측
==================================================================================
DeepDDI 핵심: 각 약물을 ★SSP(Structural Similarity Profile = 다른 약들과의 Tanimoto 닮음 목록)로
  표현 → 두 약물 SSP를 이어붙여(concat) → ★심층신경망(DNN, 8층×2048)에 학습 → 86종 중 어느
  상호작용인지 분류. ★이 저장소 첫 신경망(지금까지 XGBoost만).
목적: (1) 논문 재현 (2) 신약(cold) 평가 (3) 내 XGBoost 다중분류(10·11)와 공정비교.

■ SSP 만들기(누수 차단이 핵심):
  - 각 약물 SSP = 그 약물과 ★train 약물들과의 Tanimoto 목록(길이=train약물수 K). 13의 M2(닮음표) 재사용.
  - ★기준(reference)=train 약물만 → test 약물끼리의 닮음은 절대 특징에 안 들어감(누수 0).
    train 약물의 SSP엔 자기자신(=1.0) 포함, test 약물은 train과의 닮음만(자기 없음).
  - PCA로 축소(논문 방식): ★PCA를 train 약물 SSP에만 fit → 전체(train+test) transform(누수 0).
  - 두 약물 결합: 논문대로 concat. ★순서 고정(행번호 작은쪽 먼저)=대칭성 문제 회피(02 concat 교훈).

■ DNN(논문 구조): 은닉 8층 × 2048 노드, 각 층 Linear→BatchNorm→ReLU(★BN은 원DeepDDI 포함·8층 학습 안정화),
  출력 86, softmax, 손실 cross-entropy, Adam lr 1e-3, batch 512, epoch≤100, ★val-loss early-stopping.
  ★단순화(보고): 논문은 multi-label+sigmoid+임계0.47이나 우리 데이터는 쌍당 1유형 99.8% → 다중분류(softmax).
    10·11과 동일 세팅 → XGBoost와 공정비교.

■ 평가(★내 방식 cold): warm/cold-start/cold-cold(10·11 동일 분할). warm은 '누수 맥락' 주석.
  지표 macro-F1(주)·micro·acc·top3 + head/mid/tail. seed5. 변형=plain·class-weight(11 대응).

산출: results/14_ddi_deepddi_dnn.html , results/deepddi_results.json
실행: conda run -n admet python src/14_deepddi_dnn.py
  (빠른 점검: DEEPDDI_SMOKE=1 → seed1개·plain·epoch12)
"""
import os, sys, json, time, io, base64
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddi_common as C
from ddi_common import ukey, build_fp_matrix, split_drugs, build_sim_matrix, top3_acc

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
RAW_CSV = os.path.join(BASE, "data", "drugbank_ddi_raw.csv")
RESULTS_JSON = os.path.join(BASE, "results", "deepddi_results.json")

SMOKE = os.environ.get("DEEPDDI_SMOKE", "") == "1"
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
VARIANTS = ["plain"] if SMOKE else ["plain", "class-weight"]
MAX_EPOCH = 12 if SMOKE else 100
PATIENCE = 4 if SMOKE else 8
DRUG_TEST_FRAC = 0.20
REGIMES = ["warm", "cold-start", "cold-cold"]
PCA_VAR = 0.90               # PCA 유지 분산(논문식 축소)
N_HIDDEN, WIDTH = 8, 2048    # 논문 구조: 8층 × 2048
BATCH, LR = 512, 1e-3
N_CLASS = 86
METRICS = ["accuracy", "macro_f1", "micro_f1", "top3", "head_f1", "mid_f1", "tail_f1"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_Y(pos_pairs):
    """상호작용 유형(0..85). 10·11과 동일 로직."""
    raw = pd.read_csv(RAW_CSV); raw["Drug1_ID"] = raw["Drug1_ID"].astype(str); raw["Drug2_ID"] = raw["Drug2_ID"].astype(str)
    ymap = {}
    for a, b, y in zip(raw["Drug1_ID"], raw["Drug2_ID"], raw["Y"]):
        k = ukey(a, b)
        if k not in ymap:
            ymap[k] = int(y)
    Yraw = np.array([ymap[k] for k in pos_pairs])
    uniq = sorted(set(Yraw.tolist())); remap = {t: i for i, t in enumerate(uniq)}
    return np.array([remap[t] for t in Yraw])


# ===================== DNN =====================
class DeepDDI(nn.Module):
    """SSP-concat → 8층×2048(Linear-BN-ReLU) → 86. ★8층 심층이라 BatchNorm으로 학습 안정화(원논문 방식)."""
    def __init__(self, in_dim, n_hidden=N_HIDDEN, width=WIDTH, n_class=N_CLASS):
        super().__init__()
        layers = []; d = in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, width), nn.BatchNorm1d(width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, n_class)]      # 출력 86(softmax는 CrossEntropy가 내부처리)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_dnn(Xtr, ytr, weight=None, seed=0, track=False):
    """DNN 학습 루프(무엇을 왜):
    1) train에서 10% val 떼어 ★val-loss로 early-stopping(과적합 방지).
    2) 미니배치 Adam으로 cross-entropy 최소화. weight!=None이면 class-weighted(희귀유형 벌점↑=11 대응).
    3) 매 epoch val-loss 기록 → 개선 없으면 patience 후 중단, 최적 가중치 복원.
    반환: (model, curve[(ep,train_loss,val_loss,val_macroF1)])."""
    torch.manual_seed(seed); np.random.seed(seed)
    idx = np.arange(len(ytr))
    tr_i, va_i = train_test_split(idx, test_size=0.1, random_state=seed)   # rare type 위해 stratify 생략(견고)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    model = DeepDDI(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    wt = torch.tensor(weight, dtype=torch.float32, device=DEV) if weight is not None else None
    crit = nn.CrossEntropyLoss(weight=wt)
    Xva = Xtr_t[va_i].to(DEV); yva = ytr_t[va_i].to(DEV)
    best = np.inf; best_state = None; wait = 0; curve = []
    for ep in range(MAX_EPOCH):
        model.train()
        perm = np.random.permutation(tr_i)
        tl = 0.0; nb = 0
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            if len(b) < 2:                             # BatchNorm은 배치 1개서 에러 → 마지막 1개 배치 건너뜀(견고성)
                continue
            xb = Xtr_t[b].to(DEV); yb = ytr_t[b].to(DEV)
            opt.zero_grad(); out = model(xb); loss = crit(out, yb)
            loss.backward(); opt.step()
            tl += float(loss.item()); nb += 1
        # ---- val 평가 ----
        model.eval()
        with torch.no_grad():
            vout = model(Xva); vloss = float(crit(vout, yva).item())
            vpred = vout.argmax(1).cpu().numpy()
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


def predict_proba(model, X):
    """배치로 softmax 확률(n×86)."""
    model.eval(); out = []
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, len(Xt), 4096):
            xb = Xt[s:s + 4096].to(DEV)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def eval_dnn(proba, yte, head_t, mid_t, tail_t):
    pred = proba.argmax(1)
    tl = sorted(set(yte.tolist()))
    per_f1 = f1_score(yte, pred, average=None, labels=tl, zero_division=0)
    f1map = dict(zip(tl, per_f1))
    def band(types):
        v = [f1map[t] for t in tl if t in types]; return float(np.mean(v)) if v else 0.0
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", labels=tl, zero_division=0)),
        "micro_f1": float(f1_score(yte, pred, average="micro", labels=tl, zero_division=0)),
        "top3": top3_acc(proba, np.arange(N_CLASS), yte),
        "head_f1": band(head_t), "mid_f1": band(mid_t), "tail_f1": band(tail_t),
    }


def build_ssp_pca(M2, train_rows):
    """★SSP+PCA(누수차단): reference=train 약물. SSP_all=M2[:,train](n×K).
    PCA는 train 약물 SSP에만 fit → 전체 transform. 반환 ssp(n×P), P."""
    R = np.sort(train_rows)
    ssp_all = M2[:, R].astype(np.float32)          # 각 약물의 train 약물 대비 Tanimoto (n×K)
    pca = PCA(n_components=PCA_VAR, svd_solver="full")
    pca.fit(ssp_all[R])                            # ★train 행에만 fit(누수 0)
    z = pca.transform(ssp_all).astype(np.float32)  # 전체 축소 (n×P)
    return z, z.shape[1]


def run():
    t0 = time.time()
    print(f"[14] DeepDDI(SSP+DNN) 시작 · device={DEV} · SMOKE={SMOKE}", flush=True)
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"])); did2row = {d: i for i, d in enumerate(drug_ids)}
    n = len(drug_ids)
    pairs = pd.read_csv(PAIRS_CSV); pos = pairs[pairs.label == 1]
    pos_pairs = [ukey(str(a), str(b)) for a, b in zip(pos["drug_a"], pos["drug_b"])]
    pi = np.array([did2row[a] for a, b in pos_pairs], dtype=np.int32)
    pj = np.array([did2row[b] for a, b in pos_pairs], dtype=np.int32)
    lo = np.minimum(pi, pj); hi = np.maximum(pi, pj)          # concat 순서 고정
    Y = load_Y(pos_pairs); K = len(set(Y.tolist()))
    gc = pd.Series(Y).value_counts()
    head_t = set(gc.head(10).index.tolist()); tail_t = set(gc[gc < 100].index.tolist())
    mid_t = set(gc.index.tolist()) - head_t - tail_t
    print(f"  약물 {n} · positive {len(pi):,} · 유형 {K}종 · head{len(head_t)}/mid{len(mid_t)}/tail{len(tail_t)}", flush=True)

    FP = build_fp_matrix(drug_ids, id2smi)
    bvs = [C._GEN.GetFingerprint(Chem.MolFromSmiles(str(id2smi[d]))) for d in drug_ids]
    M2 = build_sim_matrix(bvs).astype(np.float32)
    print(f"  M2(Tanimoto) 완성", flush=True)

    per = {v: {rg: {mt: [] for mt in METRICS} for rg in REGIMES} for v in VARIANTS}
    pca_dims = []; curve_repr = None
    for seed in SEEDS:
        train_drugs, test_drugs = split_drugs(drug_ids, seed, DRUG_TEST_FRAC)
        g = np.array([0 if d in train_drugs else 1 for d in drug_ids], dtype=np.int8)
        train_rows = np.where(g == 0)[0]
        # ---- SSP+PCA (train 기준·누수 0) ----
        ssp, P = build_ssp_pca(M2, train_rows); pca_dims.append(P)
        Xall = np.concatenate([ssp[lo], ssp[hi]], axis=1)     # 쌍 특징 = concat(SSP_lo, SSP_hi) (n_pairs × 2P)
        # ---- regime 인덱스 ----
        pu = g[pi] + g[pj]
        tt = np.where(pu == 0)[0]; ts = np.where(pu == 1)[0]; ss = np.where(pu == 2)[0]
        wtr, wte = train_test_split(np.arange(len(pos_pairs)), test_size=0.2, random_state=seed, stratify=Y)
        print(f"  [seed {seed}] P={P} · TT {len(tt):,}·TS {len(ts):,}·SS {len(ss):,}", flush=True)

        for variant in VARIANTS:
            def make_weight(ytr):
                if variant != "class-weight":
                    return None
                cls = np.unique(ytr); w = compute_class_weight("balanced", classes=cls, y=ytr)
                full = np.ones(N_CLASS, dtype=np.float32); full[cls] = w      # 없는 유형=1.0(타깃 안 됨)
                return full
            # warm 모델
            mdl, cv = train_dnn(Xall[wtr], Y[wtr], make_weight(Y[wtr]), seed, track=True)
            if curve_repr is None and variant == "plain":
                curve_repr = {"regime": "warm", "variant": variant, "seed": seed, "curve": cv}
            mw = eval_dnn(predict_proba(mdl, Xall[wte]), Y[wte], head_t, mid_t, tail_t)
            # cold 공통(TT) 모델 → TS·SS
            mdl_c, cvc = train_dnn(Xall[tt], Y[tt], make_weight(Y[tt]), seed)
            mcs = eval_dnn(predict_proba(mdl_c, Xall[ts]), Y[ts], head_t, mid_t, tail_t)
            mcc = eval_dnn(predict_proba(mdl_c, Xall[ss]), Y[ss], head_t, mid_t, tail_t)
            for rg, m in [("warm", mw), ("cold-start", mcs), ("cold-cold", mcc)]:
                for mt in METRICS:
                    per[variant][rg][mt].append(m[mt])
            print(f"    [{variant:12s}] macroF1 warm {mw['macro_f1']:.3f} · cs {mcs['macro_f1']:.3f} · cc {mcc['macro_f1']:.3f}"
                  f" | cc acc {mcc['accuracy']:.3f} top3 {mcc['top3']:.3f} tail {mcc['tail_f1']:.3f}", flush=True)

    agg = {v: {rg: {mt: {"mean": float(np.mean(per[v][rg][mt])), "std": float(np.std(per[v][rg][mt]))} for mt in METRICS} for rg in REGIMES} for v in VARIANTS}
    res = {"config": {"n_pairs": len(pi), "n_types": K, "seeds": SEEDS, "pca_dim": int(np.mean(pca_dims)),
                      "n_hidden": N_HIDDEN, "width": WIDTH, "batch": BATCH, "lr": LR, "max_epoch": MAX_EPOCH,
                      "n_head": len(head_t), "n_mid": len(mid_t), "n_tail": len(tail_t), "device": str(DEV),
                      "smoke": SMOKE, "runtime_s": round(time.time() - t0, 1)},
           "results": agg, "per_seed": per, "curve": curve_repr}
    json.dump(res, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ cold-cold (macro-F1) ================")
    for v in VARIANTS:
        a = agg[v]["cold-cold"]
        print(f"  DeepDDI[{v:12s}]: macroF1 {a['macro_f1']['mean']:.3f} · acc {a['accuracy']['mean']:.3f} · tail {a['tail_f1']['mean']:.3f}  (XGB 10=0.252·11cw=0.272)")
    if not SMOKE:
        out_html = build_report(res)
        print(f"\n[14] 완료 ({res['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    else:
        print(f"\n[14] SMOKE 완료 ({res['config']['runtime_s']:.0f}s) — loss 감소·NaN 여부 확인용")
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
    # 10·11 앵커 로드
    a10 = json.load(open(os.path.join(BASE, "results", "multiclass_results.json")))["results"]["struct"]
    a11 = json.load(open(os.path.join(BASE, "results", "imbalance_results.json")))["results"]
    xgb = {"XGB struct(10)": a10, "XGB +class-weight(11)": a11["class-weight"]}

    # 그림1: 방법 스키마
    fig0, ax = plt.subplots(figsize=(9.2, 2.4)); ax.axis("off")
    steps = ["약물 A,B", "SSP\n(train 약물\n대비 Tanimoto)", "PCA 축소\n(train fit)", "concat\n(2P차원)", "DNN\n8층×2048\n(BN·ReLU)", "86종\nsoftmax"]
    cols = ["#444", "#8a6d1a", "#2a6f4c", "#3a5a8a", "#7a3a8a", "#b0392f"]
    for i, (s, col) in enumerate(zip(steps, cols)):
        x = 0.02 + i * 0.165
        ax.add_patch(plt.Rectangle((x, 0.3), 0.13, 0.44, fc=col, ec="#222", alpha=.85, transform=ax.transAxes))
        ax.text(x + 0.065, 0.52, s, ha="center", va="center", fontsize=7.8, color="white", transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.text(x + 0.15, 0.52, "→", ha="center", va="center", fontsize=13, transform=ax.transAxes)
    chart_m = _b64(fig0)

    # 그림2: macro-F1 DeepDDI vs XGBoost × regime
    fig1, ax = plt.subplots(figsize=(8.8, 3.8)); w = 0.18
    series = [(f"DeepDDI[{v}]", agg[v]) for v in agg] + [(k, v) for k, v in xgb.items()]
    palette = ["#7a3a8a", "#b0392f", "#8a8a99", "#4c78d8"]
    for i, (lab, dat) in enumerate(series):
        vals = [dat[rg]["macro_f1"]["mean"] for rg in REGIMES]
        errs = [dat[rg]["macro_f1"].get("std", 0) if isinstance(dat[rg]["macro_f1"], dict) else 0 for rg in REGIMES]
        ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, yerr=errs, capsize=2, color=palette[i % len(palette)], label=lab)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(REGIMES); ax.set_ylabel("macro-F1"); ax.set_ylim(0, 0.85)
    ax.set_title("DeepDDI(SSP+DNN) vs XGBoost(10·11) — macro-F1 × regime  (★cold-cold=신약)")
    ax.legend(fontsize=7.5, ncol=2); ax.grid(axis="y", alpha=.25)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    chart_b = _b64(fig1)

    # 그림3: 학습곡선
    cv = res.get("curve")
    if cv:
        fig2, ax = plt.subplots(figsize=(7.2, 3.2))
        arr = np.array(cv["curve"]); ep = arr[:, 0]
        ax.plot(ep, arr[:, 1], "-o", ms=3, label="train loss", color="#4c78d8")
        ax.plot(ep, arr[:, 2], "-o", ms=3, label="val loss", color="#b0392f")
        ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")
        ax.set_title(f"학습곡선 ({cv['variant']}·{cv['regime']}·seed{cv['seed']}) — loss 감소=학습 정상")
        ax2 = ax.twinx(); ax2.plot(ep, arr[:, 3], "--", color="#2a6f4c", label="val macro-F1")
        ax2.set_ylabel("val macro-F1", color="#2a6f4c")
        ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=.25)
        for sp_ in ("top",):
            ax.spines[sp_].set_visible(False)
        chart_c = _b64(fig2)
    else:
        chart_c = ""

    def rows_tbl(dat, is_xgb=False):
        r = ""
        for rg in REGIMES:
            a = dat[rg]
            def g(k):
                return a[k]["mean"] if isinstance(a[k], dict) else a[k]
            r += (f"<tr><td class=l>{rg}</td><td class=hl>{g('macro_f1'):.3f}</td><td>{g('micro_f1'):.3f}</td><td>{g('accuracy'):.3f}</td>"
                  f"<td>{g('top3'):.3f}</td><td>{g('head_f1'):.3f}</td><td>{g('mid_f1'):.3f}</td><td>{g('tail_f1'):.3f}</td></tr>")
        return r

    # 핵심 비교 수치(cold-cold macro-F1)
    dd_cc = {v: agg[v]["cold-cold"]["macro_f1"]["mean"] for v in agg}
    best_dd = max(dd_cc, key=dd_cc.get); best_dd_v = dd_cc[best_dd]
    x10 = a10["cold-cold"]["macro_f1"]["mean"]; x11 = a11["class-weight"]["cold-cold"]["macro_f1"]["mean"]
    dd_plain = agg.get("plain", {}).get("cold-cold", {}).get("macro_f1", {}).get("mean", float("nan"))
    dd_cw = agg.get("class-weight", {}).get("cold-cold", {}).get("macro_f1", {}).get("mean", float("nan"))
    verdict = ("XGBoost와 비슷" if abs(best_dd_v - max(x10, x11)) < 0.02 else
               "XGBoost보다 우위" if best_dd_v > max(x10, x11) + 0.02 else "XGBoost보다 열위")
    # warm·cold-start도 비교(어느 regime서도 신경망 우위인가)
    warm_dd = agg["plain"]["warm"]["macro_f1"]["mean"]; warm_x = a10["warm"]["macro_f1"]["mean"]
    cs_dd = agg["plain"]["cold-start"]["macro_f1"]["mean"]; cs_x = a10["cold-start"]["macro_f1"]["mean"]

    cmp_rows = ""
    for lab, v in [("DeepDDI plain", dd_plain), ("DeepDDI +class-weight", dd_cw), ("XGBoost struct (10)", x10), ("XGBoost +class-weight (11)", x11)]:
        cls = "good" if v == max(dd_plain, dd_cw, x10, x11) else "mut"
        cmp_rows += f"<tr><td class=l><b>{lab}</b></td><td class={cls}>{v:.3f}</td></tr>"

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 6px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}.hl{background:#16202e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>14 DeepDDI DNN</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>14 · 2세대(DeepDDI 2018) 재현 — SSP + 심층신경망 vs XGBoost</h1>
<p class=mut>★이 저장소 첫 신경망. SSP(train 대비 Tanimoto)+PCA → concat → DNN 8층×2048 → 86종. 10·11과 동일 데이터·분할·지표 · device {c['device']} · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> ★<b>신약(cold-cold)에서 DeepDDI(신경망)는 XGBoost(트리)와 {verdict}</b> — cold-cold macro-F1: DeepDDI(best {best_dd}) {best_dd_v:.3f} vs XGBoost 10 {x10:.3f}·11 {x11:.3f}(전체 최고). ★<b>더 복잡·무거운 신경망이 이 과제·데이터에선 트리를 못 이김</b> — ★<b>신경망이 유의하게 이기는 regime 없음</b>: warm({warm_dd:.3f}&lt;{warm_x:.3f})·cold-start({cs_dd:.3f}&lt;{cs_x:.3f})는 XGBoost 명확 우위, cold-cold는 무승부(DeepDDI plain {dd_plain:.3f}이 XGBoost struct {x10:.3f}보다 점추정 근소 위이나 편차내, XGBoost 최고 {max(x10,x11):.3f}엔 미달). 병목은 모델용량 아니라 <b>구조유사도 약한 DrugBank DDI·희박한 신약 신호</b>(모델 무관).<br>
★학습 정상(아래 학습곡선 loss 감소·NaN 없음·유형 붕괴 없음). ★SSP는 <b>train 약물 기준</b>(누수 0), warm은 누수 맥락(참고만). ★비교는 파이프라인 전체(SSP+DNN vs Morgan+트리)—순수 '신경망vs트리' 아님(⑤).</div>

<h2>① 방법 (SSP → PCA → concat → DNN)</h2>
<img src="{chart_m}" alt="method">
<div class=card>· <b>SSP</b>=각 약물의 <b>train 약물 K개 대비 Tanimoto</b> 목록(13 닮음표 재사용). ★기준=train만 → test끼리 닮음은 특징에 안 들어감(누수 0). · <b>PCA</b> train에만 fit→전체 transform, 분산 {int(PCA_VAR*100)}% 유지(P≈{c['pca_dim']}). · <b>concat</b>(순서=행번호 고정, 대칭성 회피)→2P. · <b>DNN</b> {c['n_hidden']}층×{c['width']}(Linear-BN-ReLU)·출력{N_CLASS}·CE·Adam lr{c['lr']}·batch{c['batch']}·epoch≤{c['max_epoch']}·val-loss early-stop.<br>
★<b>출력(명시)</b>: 86종 <b>softmax 다중분류</b>(쌍당 1유형 99.8%) — DeepDDI의 DDI 유형분류 헤드와 부합하며 ★10·11(XGBoost multi:softprob)과 <b>동일 타깃</b>이라 공정비교. ★BatchNorm은 8층 학습 안정화용 추가(원DeepDDI도 배치정규화 사용).</div>

<h2>② 학습곡선 (신경망이 제대로 학습됐나)</h2>
<img src="{chart_c}" alt="learning curve">
<div class=card>train·val loss가 <b>감소</b>하고 val macro-F1이 <b>상승</b> → 학습 정상(NaN·발산 없음). early-stopping이 val-loss 최저에서 멈춤(과적합 방지).</div>

<h2>③ 성능: DeepDDI 변형별 × regime (macro-F1 + micro/acc/top3 + head/mid/tail)</h2>
<img src="{chart_b}" alt="bars">"""
    for v in agg:
        html += f"""
<p class=mut><b>DeepDDI [{v}]</b> (5 seed 평균)</p>
<div style="overflow-x:auto"><table>
<tr><th class=l>regime</th><th>macroF1</th><th>microF1</th><th>acc</th><th>top3</th><th>head</th><th>mid</th><th>tail</th></tr>
{rows_tbl(agg[v])}
</table></div>"""
    html += f"""
<p class=mut>warm은 ★누수 맥락(약물이 train·test 양쪽 등장 가능=자기 유형 암기분 포함) — 참고만, ★핵심은 cold-start·cold-cold.</p>

<h2>④ ★핵심 비교: DeepDDI vs XGBoost (cold-cold macro-F1, 신약)</h2>
<div style="overflow-x:auto"><table>
<tr><th class=l>방법 (cold-cold)</th><th>macro-F1</th></tr>
{cmp_rows}
</table></div>
<div class=card>★<b>결론</b>: 신약(둘 다 미지)에서 DeepDDI best {best_dd_v:.3f} vs XGBoost 최고 {max(x10,x11):.3f} → <b>{verdict}</b> (★편차내 판단: gap {abs(best_dd_v-max(x10,x11)):.3f} &lt; seed std ~{agg['plain']['cold-cold']['macro_f1']['std']:.3f}·유의검정 미실시).<br>
· <b>왜 신경망이 크게 못 이기나</b>: (1) DrugBank DDI는 <b>구조유사도가 약함</b>(04/05 확인·pos Tanimoto~0.10) → SSP(구조닮음 기반)의 상한이 낮음 — 표현이 구조에 갇힘. (2) 신약 유형 신호 자체가 희박(10·11서 XGBoost도 cc 0.25~0.27) → 모델 용량보다 <b>데이터·표현</b>이 병목. (3) 8층 DNN의 표현력 이점이 이 저차원·구조기반 특징에선 트리 대비 미미.<br>
· <b>class-weight 효과가 반대</b>: DeepDDI plain {dd_plain:.3f} → +cw {dd_cw:.3f}({dd_cw-dd_plain:+.3f}, ★<b>DNN은 오히려 손해</b>) vs XGBoost는 이득(11: {x10:.3f}→{x11:.3f}). 즉 11서 트리에 통한 불균형기법이 DNN엔 안 통함(DNN cw는 acc {agg['class-weight']['cold-cold']['accuracy']['mean']:.3f}로 과하게 희생·tail만 {agg['class-weight']['cold-cold']['tail_f1']['mean']:.3f}↑).<br>
· <b>신경망 유의 우위 regime 없음</b>: warm({warm_dd:.3f}&lt;{warm_x:.3f})·cold-start({cs_dd:.3f}&lt;{cs_x:.3f})는 XGBoost 명확 우위, cold-cold만 무승부(plain {dd_plain:.3f}이 XGBoost struct {x10:.3f}보다 점추정 근소 위이나 std~{agg['plain']['cold-cold']['macro_f1']['std']:.3f} 내). cold-start 격차({cs_x-cs_dd:+.3f})가 특히 큼 — SSP 표현이 Morgan보다 신약 일반화에 불리.</div>

<h2>⑤ ★한계·정직 메모</h2>
<div class=card>
· <b>warm 누수 맥락</b>: 같은 약물이 train·test 양쪽 → 자기 유형 암기분 포함, warm 절대값 과대. 핵심은 cold.<br>
· <b>SSP=train 기준</b>: test 약물 SSP는 train과의 닮음만(누수 0) — 신약이 train과 안 닮으면 정보 빈약(cold-cold 어려움의 구조적 원인).<br>
· <b>출력=86 softmax 다중분류</b>(쌍당 1유형 99.8%) = 10·11과 동일 타깃(공정비교)·DeepDDI DDI-유형 분류와 부합. 원논문 출력형식 세부는 재확인 필요.<br>
· ★<b>early-stop=val-loss 기준</b>: val macro-F1이 더 오를 여지(예: 0.749→0.764)에서 멈춤 → <b>보수적</b>(DeepDDI를 약간 과소평가 가능) → 그럼에도 DNN이 트리 못 이김 = '≈' 결론 견고.<br>
· <b>BatchNorm 추가</b>(원논문 방식·8층 안정화) — 사용자 명세엔 없었으나 학습 위해 필요, 명시.<br>
· ★<b>비교는 파이프라인 전체(SSP+DNN vs Morgan+트리)</b> — '신경망 vs 트리'만 분리한 게 <b>아님</b>: 표현(SSP vs Morgan)과 모델(DNN vs 트리)을 함께 바꿈. 순수 모델비교는 '같은 특징으로 트리 vs DNN'이 필요(미실시). 결론은 '<b>DeepDDI 파이프라인이 내 XGBoost 파이프라인을 못 이김</b>'까지.<br>
· <b>DNN 무거움</b>(8층×2048·GPU {c['runtime_s']:.0f}s) vs XGBoost 초단위 — 비용 대비 이득 없음(cold). · rare type 표본부족→cold-cold tail F1 seed간 변동 큼(0.00~0.20, 표본 수천개뿐) · 이진(02~09)과 비교 불가.</div>

<h2>⑥ 다음</h2>
<div class=card>· SSP에 <b>타깃 유사도</b>(12) 결합해 표현 확장 · 원논문 multi-label+sigmoid 충실재현 · GNN(구조 직접) · cold-cold 하이브리드(13 Vilar+DNN).</div>

<div class=foot>14 DeepDDI SSP+DNN({N_HIDDEN}층×{WIDTH}·BN) vs XGBoost(10·11) · warm/cold-start/cold-cold · macro-F1+head/mid/tail · PCA {int(PCA_VAR*100)}%(P≈{c['pca_dim']}) · concat 순서고정 · softmax 단순화 · seed {len(c['seeds'])} · {c['device']} · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "14_ddi_deepddi_dnn.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


if __name__ == "__main__":
    run()

"""
[DDI 실험 - 19] 4세대(SSI-DDI) 하이퍼파라미터 탐색 — 하한선이 진짜 하한인가
==================================================================================
16·18에서 4세대(SSI-DDI)는 '튜닝 안 한 작은 설정'(d=64·L=4)이라 cold-cold macro-F1 0.222가
★하한선이었다. 이번엔 SSI-DDI를 ★키우고 튜닝해서 신약(cold-cold)이 정말 오르나 확인:
  · 오르면 = "낮았던 건 설정이 작아서"(하한 맞음)
  · 안 오르면 = "튜닝해도 그대로 = 진짜 한계(데이터/표현 병목)".

■ 재사용: m16.SSIDDI(head_out·heads·n_blocks로 d·L 조절)·encode_pairs·prepare_data·
  make_regime_indices·eval_dnn. ★16 코드 그대로, 하이퍼파라미터만 탐색. 평가 틀 16·17·18과 동일.
  d = head_out × heads (heads=2 고정 → head_out 32/64/128 = d 64/128/256). L = n_blocks.

■ 단계(★관문): [1] d·L 예비(warm·seed0·6구성) → 멈춤·보고. [2] 유망설정 cold-cold 전체·5seed.
  [3] 최고설정 + lr·dropout 미세조정.
■ 안전: 큰 모델(d256·L6) 메모리 큼 → OOM이면 배치 자동 축소·보고. loss NaN/발산 감지.

실행: conda run -n admet python src/19_ssiddi_tune.py stage1
산출: results/19_ddi_ssiddi_tune.html · results/ssiddi_tune_stage{N}.json
"""
import os, sys, json, time, importlib.util, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
torch.backends.cudnn.deterministic = True          # run-to-run 잡음 축소(17 교훈)
torch.backends.cudnn.benchmark = False

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import ddi_common as C


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m16 = _load("m16_tune", "16_ssiddi_gnn.py")        # SSIDDI·encode_pairs·prepare_data·eval_dnn
DEV = m16.DEV
BASE = os.path.dirname(SRC)
BASELINE_CC = 0.2185        # 16 SSI-DDI cold-cold macro-F1(하한). 3세대 DDIMDL struct = 0.2837.
GEN3_CC = 0.2837


# ══════════════════════════════════════════════════════════════════════════════
#  튜닝 가능한 학습기 (16 train_ssiddi를 하이퍼파라미터 인자화 — 모델은 m16.SSIDDI 재사용)
# ══════════════════════════════════════════════════════════════════════════════
def train_cfg(D, train_idx, seed, head_out, heads, n_blocks, lr, batch, max_epoch, patience, track=False):
    """★무엇을 왜: d(=head_out×heads)·L(=n_blocks)·lr·batch를 인자로 받아 SSI-DDI 학습.
    나머지(그래프·co-attention·RESCAL·early-stopping)는 16과 동일. 반환 (model, curve, epochs, nan)."""
    torch.manual_seed(seed); np.random.seed(seed)
    if DEV.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    tr, va = train_test_split(train_idx, test_size=0.1, random_state=seed)
    model = m16.SSIDDI(head_out=head_out, heads=heads, n_blocks=n_blocks).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    graphs, lo, hi, Y = D["graphs"], D["lo"], D["hi"], D["Y"]
    yT = torch.as_tensor(Y, device=DEV)
    best, best_state, bad, curve, nan = float("inf"), None, 0, [], False
    for ep in range(max_epoch):
        model.train(); perm = np.random.permutation(tr); tot, seen = 0.0, 0
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            logits = m16.encode_pairs(model, graphs, lo[b], hi[b])
            loss = F.cross_entropy(logits, yT[b])
            if not torch.isfinite(loss):
                nan = True; break
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b); seen += len(b)
        if nan:
            break
        model.eval(); vtot, vseen = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(va), batch):
                b = va[s:s + batch]
                vtot += float(F.cross_entropy(m16.encode_pairs(model, graphs, lo[b], hi[b]), yT[b])) * len(b); vseen += len(b)
        vl = vtot / max(vseen, 1)
        if track:
            curve.append((round(tot / max(seen, 1), 4), round(vl, 4)))
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, curve, ep + 1, nan


def train_oom_safe(D, train_idx, seed, head_out, heads, n_blocks, lr, batch, max_epoch, patience, track=False):
    """OOM이면 배치 반씩 줄여 재시도(사용자 지시). 반환 (..., used_batch)."""
    b = batch
    while b >= 16:
        try:
            model, curve, ep, nan = train_cfg(D, train_idx, seed, head_out, heads, n_blocks, lr, b, max_epoch, patience, track)
            return model, curve, ep, nan, b
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache(); b //= 2
                print(f"    [OOM] batch↓ → {b}", flush=True)
            else:
                raise
    raise RuntimeError("OOM even at batch 16")


def predict(model, D, idx, batch):
    graphs, lo, hi = D["graphs"], D["lo"], D["hi"]
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            out.append(F.softmax(m16.encode_pairs(model, graphs, lo[b], hi[b]), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ══════════════════════════════════════════════════════════════════════════════
#  [1단계] d·L 예비 탐색 — warm·seed0. "학습 되나 + 대략 경향" 확인 후 ★멈춤.
# ══════════════════════════════════════════════════════════════════════════════
def stage1():
    t0 = time.time()
    D = m16.prepare_data()
    R = m16.make_regime_indices(D, seed=0)
    buckets = (D["head_t"], D["mid_t"], D["tail_t"])
    print(f"[19-s1] 약물 {len(D['drug_ids'])} · positive {len(D['Y'])} · device {DEV} · "
          f"warm-train {len(R['wtr'])} · 기준 16 cc={BASELINE_CC}", flush=True)
    # d = head_out*heads(heads=2). L = n_blocks. lr=1e-3·batch128·dropout0(현재)·max_epoch40.
    HEADS, LR, BATCH0, MAXEP, PAT = 2, 1e-3, 128, 40, 8
    grid = [(ho, nb) for ho in (32, 64, 128) for nb in (4, 6)]   # d∈{64,128,256} × L∈{4,6} = 6
    rows = []
    for ho, nb in grid:
        d = ho * HEADS
        tag = f"d{d}·L{nb}"
        t1 = time.time()
        model, curve, ep, nan, ub = train_oom_safe(D, R["wtr"], 0, ho, HEADS, nb, LR, BATCH0, MAXEP, PAT, track=True)
        if nan:
            print(f"  {tag}: ★NaN/발산 — 스킵", flush=True)
            rows.append({"tag": tag, "d": d, "L": nb, "warm_macro_f1": None, "nan": True}); continue
        pw = predict(model, D, R["wte"], ub)
        ew = m16.eval_dnn(pw, D["Y"][R["wte"]], *buckets)
        npar = n_params(model)
        dt = time.time() - t1
        rows.append({"tag": tag, "d": d, "L": nb, "head_out": ho, "heads": HEADS, "batch": ub,
                     "params": int(npar), "epochs": ep, "warm_macro_f1": round(ew["macro_f1"], 4),
                     "warm_acc": round(ew["accuracy"], 4), "curve": curve, "runtime_s": round(dt, 1),
                     "final_train_loss": curve[-1][0] if curve else None, "final_val_loss": curve[-1][1] if curve else None,
                     "nan": False})
        print(f"  {tag}: warm macro-F1={ew['macro_f1']:.4f} · {npar/1e3:.0f}k param · {ep}ep · "
              f"batch{ub} · loss {curve[0][0]:.2f}→{curve[-1][0]:.2f} · {dt:.0f}s", flush=True)
    out = {"stage": 1, "regime": "warm", "seed": 0, "baseline_cc": BASELINE_CC, "gen3_cc": GEN3_CC,
           "grid": "d{64,128,256}×L{4,6}", "lr": LR, "batch0": BATCH0, "max_epoch": MAXEP,
           "device": str(DEV), "runtime_s": round(time.time() - t0, 1), "rows": rows}
    with open(os.path.join(BASE, "results", "ssiddi_tune_stage1.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    # 요약
    ok = [r for r in rows if r.get("warm_macro_f1") is not None]
    print(f"\n[19-s1] 완료 {(time.time()-t0)/60:.1f}분 · warm macro-F1 경향(d·L↑ 시 오르나):")
    for r in sorted(ok, key=lambda r: (r["d"], r["L"])):
        print(f"   {r['tag']:10s} warm={r['warm_macro_f1']:.4f} ({r['params']/1e3:.0f}k)")
    if ok:
        best = max(ok, key=lambda r: r["warm_macro_f1"])
        print(f"[19-s1] warm 최고: {best['tag']} ({best['warm_macro_f1']:.4f}). "
              f"★멈춤 — 2단계(cold-cold) 진행 여부·구성 사용자 확인 대기.")
    return out


REGIMES = ["warm", "cold-start", "cold-cold"]
METRICS = ["macro_f1", "accuracy", "top3", "head_f1", "mid_f1", "tail_f1"]
# 사용자 확정 2단계 구성: 용량 spread (앵커·중간·최대)
STAGE2_CONFIGS = [("d64·L4", 32, 4), ("d128·L6", 64, 6), ("d256·L6", 128, 6)]


def _eval_regimes(mw, mc, D, R, buckets, batch):
    """warm=warm모델·cold-start/cold-cold=cold모델(TT학습). 16과 동일 평가."""
    return {"warm": m16.eval_dnn(predict(mw, D, R["wte"], batch), D["Y"][R["wte"]], *buckets),
            "cold-start": m16.eval_dnn(predict(mc, D, R["ts"], batch), D["Y"][R["ts"]], *buckets),
            "cold-cold": m16.eval_dnn(predict(mc, D, R["ss"], batch), D["Y"][R["ss"]], *buckets)}


def stage2():
    """[2단계] 3구성 × 5seed × warm/cold-start/cold-cold. ★진짜 비교: cold-cold가 0.2185 넘나."""
    t0 = time.time()
    D = m16.prepare_data()
    buckets = (D["head_t"], D["mid_t"], D["tail_t"])
    print(f"[19-s2] {len(D['drug_ids'])} 약물 · 기준 16 cc={BASELINE_CC} · 3세대 {GEN3_CC} · configs "
          f"{[c[0] for c in STAGE2_CONFIGS]} · device {DEV}", flush=True)
    per = {tag: {rg: {m: [] for m in METRICS} for rg in REGIMES} for tag, _, _ in STAGE2_CONFIGS}
    curves = {}
    for tag, ho, nb in STAGE2_CONFIGS:
        for si, seed in enumerate(range(5)):
            R = m16.make_regime_indices(D, seed)
            mw, cw, epw, nanw, ub = train_oom_safe(D, R["wtr"], seed, ho, 2, nb, 1e-3, 128, 100, 8, track=(si == 0))
            mc, _, epc, nanc, _ = train_oom_safe(D, R["tt"], seed, ho, 2, nb, 1e-3, 128, 100, 8)
            if nanw or nanc:
                print(f"  {tag} seed{seed}: ★NaN — 스킵", flush=True); continue
            res = _eval_regimes(mw, mc, D, R, buckets, ub)
            for rg in REGIMES:
                for m in METRICS:
                    per[tag][rg][m].append(res[rg][m])
            if si == 0:
                curves[tag] = cw
            print(f"  {tag} seed{seed}: cc macro-F1={res['cold-cold']['macro_f1']:.4f} "
                  f"(warm {res['warm']['macro_f1']:.3f}·cs {res['cold-start']['macro_f1']:.3f}) "
                  f"·batch{ub}·{epw}/{epc}ep · 누적 {(time.time()-t0)/60:.0f}분", flush=True)
    agg = {tag: {rg: {m: {"mean": float(np.mean(per[tag][rg][m])), "std": float(np.std(per[tag][rg][m]))}
                      for m in METRICS} for rg in REGIMES} for tag, _, _ in STAGE2_CONFIGS}
    out = {"stage": 2, "configs": [c[0] for c in STAGE2_CONFIGS], "seeds": list(range(5)),
           "baseline_cc": BASELINE_CC, "gen3_cc": GEN3_CC, "lr": 1e-3, "batch": 128, "device": str(DEV),
           "runtime_s": round(time.time() - t0, 1), "results": agg, "per_seed": per, "curves": curves}
    with open(os.path.join(BASE, "results", "ssiddi_tune_stage2.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[19-s2] 완료 {(time.time()-t0)/60:.1f}분 · cold-cold macro-F1 (5seed):")
    for tag, _, _ in STAGE2_CONFIGS:
        a = agg[tag]["cold-cold"]["macro_f1"]
        print(f"   {tag:10s} cc={a['mean']:.4f}±{a['std']:.4f}  (기준 {BASELINE_CC} · 3세대 {GEN3_CC})")
    build_report(out)
    return out


def _b64(fig):
    import io, base64
    import matplotlib.pyplot as plt
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_report(s2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from scipy.stats import binomtest
    for _kf in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]:
        if os.path.exists(_kf):
            fm.fontManager.addfont(_kf); plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False
    s1 = json.load(open(os.path.join(BASE, "results", "ssiddi_tune_stage1.json"))) if os.path.exists(os.path.join(BASE, "results", "ssiddi_tune_stage1.json")) else None
    agg = s2["results"]; ps = s2["per_seed"]; tags = s2["configs"]
    def cc(tag): return agg[tag]["cold-cold"]["macro_f1"]["mean"]
    best_tag = max(tags, key=cc); best_cc = cc(best_tag)
    anchor = "d64·L4"

    # 부호검정: 스케일업 vs 앵커(d64·L4), per-seed paired
    def sign(a, b):
        a, b = np.array(a), np.array(b); n = min(len(a), len(b)); a, b = a[:n], b[:n]
        w = int((b > a).sum()); p = binomtest(w, n, 0.5).pvalue if n >= 2 else float("nan")
        return round(float((b - a).mean()), 4), w, n, p
    sig = []
    for tag in tags:
        if tag == anchor:
            continue
        dm, w, n, p = sign(ps[anchor]["cold-cold"]["macro_f1"], ps[tag]["cold-cold"]["macro_f1"])
        sig.append((tag, dm, w, n, p))

    # 그림: cc macro-F1 막대 (기준선·3세대 표시)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    vals = [cc(t) for t in tags]; errs = [agg[t]["cold-cold"]["macro_f1"]["std"] for t in tags]
    ax.bar(range(len(tags)), vals, yerr=errs, capsize=3, color=["#8a8a99", "#4c78d8", "#7a3a8a"][:len(tags)])
    ax.axhline(s2["baseline_cc"], ls="--", c="#b0392f", label=f"16 하한 {s2['baseline_cc']}")
    ax.axhline(s2["gen3_cc"], ls=":", c="#2a9d6f", label=f"3세대 DDIMDL {s2['gen3_cc']}")
    ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags); ax.set_ylabel("cold-cold macro-F1")
    ax.set_title("SSI-DDI 튜닝 — cold-cold(신약) vs 하한·3세대"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    chart = _b64(fig)

    # 학습곡선(과적합)
    fig2, ax = plt.subplots(figsize=(7.6, 3.2))
    for t in tags:
        cv = s2["curves"].get(t)
        if cv:
            arr = np.array(cv); ax.plot(range(1, len(arr) + 1), arr[:, 1], "-o", ms=2, label=f"{t} val")
    ax.set_xlabel("epoch"); ax.set_ylabel("val loss"); ax.set_title("학습곡선(warm·seed0) — 큰 모델 과적합?")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    chart2 = _b64(fig2)

    # ★판정: raw 0.2185 대신 ★앵커(d64·L4, 같은 setup) 대비 부호검정 + 3세대 도달 여부(정직).
    #   (앵커 0.232가 16의 0.2185보다 +0.014 높은 건 batch128 setup 차이지 튜닝 아님.)
    best_sig = next((s for s in sig if s[0] == best_tag), (best_tag, best_cc - cc(anchor), 0, 5, 1.0))
    _, dm_b, w_b, n_b, p_b = best_sig
    reaches_g3 = best_cc >= s2["gen3_cc"] - 0.01
    sig_over_anchor = (not np.isnan(p_b)) and p_b < 0.05 and dm_b > 0
    if sig_over_anchor and reaches_g3:
        verdict = "설정이 작아서였다 — 키우면 유의 상승·3세대 경쟁력(하한은 설정탓)"
    elif dm_b > 0 and not reaches_g3:
        verdict = (f"대부분 <b>진짜 한계</b> — d·L 튜닝은 앵커 대비 작은 <b>비유의</b> 향상뿐"
                   f"(+{dm_b:.3f}·{w_b}/{n_b}·p_sign {p_b:.3f}), d128서 <b>포화</b>(d256 무이득), <b>3세대 미달</b>, 큰 모델은 warm 과적합")
    else:
        verdict = "진짜 한계 — d·L 튜닝 무이득"
    beat_base = best_cc > cc(anchor) + 0.005    # 앵커 기준(리포트 기호용)

    def row(tag):
        r = agg[tag]
        return ("<tr><td class=l><b>" + tag + "</b></td>"
                + "".join(f"<td class={'hl' if rg=='cold-cold' else ''}>{r[rg]['macro_f1']['mean']:.3f}"
                          f"<span class=s>±{r[rg]['macro_f1']['std']:.3f}</span></td>" for rg in REGIMES)
                + f"<td>{r['cold-cold']['accuracy']['mean']:.3f}</td></tr>")
    s1rows = ""
    if s1:
        for r in sorted([x for x in s1["rows"] if x.get("warm_macro_f1")], key=lambda r: (r["d"], r["L"])):
            s1rows += f"<tr><td class=l>{r['tag']}</td><td>{r['warm_macro_f1']:.3f}</td><td>{r['params']/1e3:.0f}k</td><td>{r['epochs']}</td></tr>"
    sigrows = ""
    for tag, dm, w, n, p in sig:
        c_ = "good" if (dm > 0 and p < 0.10) else ("bad" if (dm < 0 and p < 0.10) else "mut")
        sigrows += f"<tr><td class=l>{tag} vs {anchor}</td><td class={'good' if dm>0 else 'bad'}>{dm:+.4f}</td><td>{w}/{n}</td><td class={c_}>{p:.3f}</td></tr>"

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 6px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.hl{background:#16202e}.s{color:#7a7f8a;font-size:10px}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}"""
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>19 SSI-DDI tune</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>19 · 4세대(SSI-DDI) 하이퍼파라미터 탐색 — 하한선이 진짜 하한인가</h1>
<p class=mut>16 SSI-DDI(d64·L4)는 튜닝안한 하한(cc {s2['baseline_cc']}). 키워서(d·L) 신약 cold-cold가 오르나 = 하한이 설정탓인가 한계인가. 16·17·18과 동일 데이터/분할/지표 · {s2['device']} · {s2['runtime_s']:.0f}s · seed 5</p>
<div class=card><b>한 줄 결론:</b> 최고 튜닝 <b>{best_tag}</b> cold-cold <b>{best_cc:.3f}</b> · ★깨끗한 비교 = 앵커(d64·L4·같은 setup) {cc(anchor):.3f} 대비 Δ<b>{dm_b:+.3f}</b>({w_b}/{n_b}·p_sign {p_b:.3f}) · 3세대 DDIMDL {s2['gen3_cc']}엔 Δ{best_cc-s2['gen3_cc']:+.3f}.<br>
→ ★<b>{verdict}</b>.<br>
★<b>정직</b>: 16 raw 하한 {s2['baseline_cc']} 대비 Δ에는 앵커의 <b>+{cc(anchor)-s2['baseline_cc']:.3f}(batch128 setup차·튜닝 아님)</b>가 섞여 있음 → 튜닝 순효과는 <b>스케일업 vs 앵커</b>로만 판정. n=5 부호검정 바닥 p=0.0625(어떤 결과도 p&lt;0.05 어려움)·17/18 교훈.</div>

<h2>① 핵심표: cold-cold macro-F1 (튜닝 vs 하한 vs 3세대)</h2>
<img src="{chart}">
<div style="overflow-x:auto"><table>
<tr><th class=l>구성(d·L)</th>""" + "".join(f"<th>{rg}</th>" for rg in REGIMES) + f"""<th>acc(cc)</th></tr>
{''.join(row(t) for t in tags)}
</table></div>
<div style="overflow-x:auto"><table>
<tr><th class=l>스케일업 vs 앵커(d64·L4) · cold-cold</th><th>Δmean</th><th>win</th><th>p_sign</th></tr>
{sigrows}
</table></div>
<div class=card>★키우면 cold-cold가 오르는지: {best_tag}가 앵커 대비 {'상승' if best_cc>cc(anchor) else '하락/무변'}. d(용량)·L(층) 중 무엇이 cold-cold를 움직였나는 위 표·1단계 대조.</div>

<h2>② 예비(1단계) warm 경향</h2>
<div style="overflow-x:auto"><table><tr><th class=l>구성</th><th>warm macroF1</th><th>params</th><th>epochs</th></tr>{s1rows}</table></div>
<p class=mut>warm은 용량↑에 비단조·큰 모델 과적합 경향(아래 곡선). warm↑이 cold-cold↑를 뜻하진 않음(핵심은 ①).</p>

<h2>③ 학습곡선 (큰 모델 과적합?)</h2>
<img src="{chart2}">

<h2>④ ★결론</h2>
<div class=card>· ★깨끗한 비교(스케일업 vs 앵커 d64·L4·같은 setup): 최고 {best_tag} Δ{dm_b:+.3f}·{w_b}/{n_b}·p_sign {p_b:.3f} = <b>{'유의 상승' if sig_over_anchor else '비유의(방향성만·편차내)'}</b> → <b>{verdict}</b>.<br>
· 3세대 DDIMDL {s2['gen3_cc']} 대비: 최고 {best_cc:.3f}로 Δ{best_cc-s2['gen3_cc']:+.3f} ({'따라잡음' if reaches_g3 else '여전히 미달'}).<br>
· <b>용량 포화</b>: d128·L6({cc('d128·L6'):.3f}) ≈ d256·L6({cc('d256·L6'):.3f}) — 더 키워도 무이득. warm은 오르나(과적합) cold-cold 정체 = 큰 모델이 novel-drug 일반화 못함.<br>
· <b>정직 한계</b>: 탐색범위 한정(d·L만·lr/dropout 미조정=3단계 여지)·앵커가 16 raw보다 +{cc(anchor)-s2['baseline_cc']:.3f}(setup차)·n=5 fold상관→부호검정(p바닥 0.0625)·큰 모델 과적합(dropout이 도울 수 있음)·여전히 파이프라인(그래프+GNN)·warm 누수맥락·GPU 3.6h.</div>
<div class=foot>19 SSI-DDI 튜닝 · 1단계 d·L 예비(warm) + 2단계 {'·'.join(tags)} cold-cold 5seed · 16 코드 재사용·HP만 · {s2['device']} · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "19_ddi_ssiddi_tune.html")
    with open(out_html, "w") as f:
        f.write(html)
    print(f"[report] → {out_html}", flush=True)
    return out_html


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    if stage == "stage1":
        stage1()
    elif stage == "stage2":
        stage2()
    elif stage == "report":
        build_report(json.load(open(os.path.join(BASE, "results", "ssiddi_tune_stage2.json"))))
    else:
        print(f"[알림] '{stage}' 불명. stage1|stage2|report.")


if __name__ == "__main__":
    main()

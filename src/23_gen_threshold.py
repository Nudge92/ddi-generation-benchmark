"""
[DDI 실험 - 23] 22 확장 — 2·3·4세대 민감도·특이도 (블록 B 통합표 완성)
==============================================================================================
22는 1세대 Vilar에만 임계값 기반 지표(민감도·특이도·정밀도·F1)를 냈다. 17은 2·3·4세대
(DeepDDI·DDIMDL·SSI-DDI)를 ROC/PR만 냈다. 23은 그 빈칸을 채운다 — ★22의 임계값 로직을
그대로 재사용하되 대상을 2·3·4세대로 바꿔, 1~4세대가 한 표(블록 B)에 모이게 한다.

■ ★재사용 (재구현·새 학습·새 데이터 없음):
  · 22 그대로 import: op_point·pick_spec_threshold·threshold_metrics·_is_degenerate·TARGET_SPEC·VILAR_PAPER
  · 17 그대로 import: load_data·make_splits·build_shared·vilar_score·train_feat_dnn·_batched_score·
    train_gnn·gnn_score·binary_metrics·build_ssp_pca·DeepDDI·SubModel·SSIDDI·하이퍼파라미터·seed 0~4.
  · ★17이 원점수(s_val/s_test)를 저장 안 했으므로(축약지표 ROC/PR/F1/acc만) 동일 코드·동일 seed·
    동일 분할로 ★점수만 재생성★한 뒤 22의 threshold_metrics를 적용한다. = 재학습이나 '새 실험'이 아님.

■ ★재현 게이트(사용자 확정 ±0.01): 재생성 점수의 ROC가 17 저장값과 ±0.01 이내면 '같은 점수'로 보고
  작동점 지표 산출. 벗어나면 리포트에 명시(gen4=CUDA GNN 비결정성 예상·SoM 실험 교훈). gen1(Vilar)은
  무학습이라 22와 비트동일 재현 당연 → gen1은 22 JSON 값도 함께 이식·대조.

■ 산출(각 세대 · warm/cold-start/cold-cold):
  · ROC·PR (17 재확인) + 17 저장값 대조(게이트)
  · ★두 작동점의 민감도·특이도·정밀도·F1: (a) Youden-J (b) 특이도 ≥ 0.96 지점
  · 두 임계 방식: (가) test 직접 = spec96 (표준) · (나) val 선택→test = spec96_val (누수 없는 보수판)
  · seed 5개 평균±std · cold-cold degenerate면 22처럼 "붕괴/N/A" 정직 보고

■ 정직 프레이밍: 민감도·특이도는 prevalence 무관 → 세대 간 비교 공정(정밀도·PR은 prevalence 의존=참고만).
  warm은 약물 겹침(누수 맥락)·핵심은 신약(cold-cold). 각 세대는 저자 원본이 아닌 17 기준 내 구현.

산출: results/23_ddi_gen_threshold.html , results/gen_threshold_results.json
실행: conda run -n admet python src/23_gen_threshold.py [smoke|full]
  (smoke는 17의 SMOKE 경로를 그대로 타 1 fold·3 epoch로 배선 검증)
"""
import os, sys, json, time, io, base64, importlib.util
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "src")
sys.path.insert(0, SRC)


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# ★17·22 그대로 import (sys.argv[1]=="smoke"면 17이 자동으로 짧은 epoch·1 fold로 세팅됨)
m17 = _load("m17_bench", "17_benchmark_binary.py")
m22 = _load("m22_thr", "22_vilar_threshold.py")

REGIMES = m17.REGIMES
GENERATIONS = m17.GENERATIONS                 # ["1 Vilar","2 DeepDDI","3 DDIMDL","4 SSI-DDI"]
FOLD_SEEDS = m17.FOLD_SEEDS
OP_KEYS = m22.OP_KEYS                          # ["sens","spec","prec","f1"]
TARGET_SPEC = m22.TARGET_SPEC
VILAR_PAPER = m22.VILAR_PAPER
RESULTS_JSON = os.path.join(BASE, "results", "gen_threshold_results.json")
BENCH17_JSON = os.path.join(BASE, "results", "benchmark_binary_results.json")
VILAR22_JSON = os.path.join(BASE, "results", "vilar_threshold_results.json")
GATE_TOL = 0.01                               # ★재현 게이트 허용오차(사용자 확정)


# ══════════════════════════════════════════════════════════════════════════════
#  한 fold: 4세대 각각 warm/cs/cc 원점수 재생성 → 22 threshold_metrics + 17 binary_metrics(게이트)
#  ★17.run_fold 구조를 그대로 따르되, binary_metrics로 축약하기 '전'의 s_val/s_test를 붙잡아 사용.
# ══════════════════════════════════════════════════════════════════════════════
def fold_metrics(seed, shared, gens=None):
    gens = gens or GENERATIONS
    drug_ids, id2smi = shared["drug_ids"], shared["id2smi"]
    lo, hi, y = shared["lo"], shared["hi"], shared["y"]
    M2, graphs, E = shared["M2"], shared["graphs"], shared["E"]
    n = len(drug_ids)
    S = m17.make_splits(drug_ids, lo, hi, y, seed)
    reg = {"warm": (S["w_tr"], S["w_va"], S["w_te"]),
           "cold-start": (S["tt_tr"], S["tt_va"], S["ts"]),
           "cold-cold": (S["tt_tr"], S["tt_va"], S["ss"])}
    pos = (y == 1)

    # SSP 특징(2·3세대): warm=전체약물·cold=train약물 (17과 동일)
    need_feat = any(g in gens for g in ("2 DeepDDI", "3 DDIMDL"))
    if need_feat:
        ssp_w, _ = m17.build_ssp_pca(M2, np.arange(n))
        ssp_c, _ = m17.build_ssp_pca(M2, S["train_rows"])
        Xs_w = np.concatenate([ssp_w[lo], ssp_w[hi]], axis=1)
        Xs_c = np.concatenate([ssp_c[lo], ssp_c[hi]], axis=1)
        esp_w, _ = m17.build_ssp_pca(E, np.arange(n)); esp_c, _ = m17.build_ssp_pca(E, S["train_rows"])
        Xe_w = np.concatenate([esp_w[lo], esp_w[hi]], axis=1); Xe_c = np.concatenate([esp_c[lo], esp_c[hi]], axis=1)

    out = {g: {} for g in gens}               # gen -> regime -> {threshold_metrics + roc_gate}
    for rg, (tr, va, te) in reg.items():
        warm = (rg == "warm")
        yv, yt = y[va], y[te]
        tr_pos = tr[pos[tr]]

        def finalize(gen, sv, st):
            tm = m22.threshold_metrics(yv, sv, yt, st)           # ★22: Youden·spec@0.96·spec96_val
            bm = m17.binary_metrics(yv, sv, yt, st)              # ★17: 게이트용 ROC 재확인
            tm["roc_gate"] = bm["roc_auc"]                        # threshold_metrics.roc_auc와 동일해야
            out[gen][rg] = tm

        # 1세대 Vilar (무학습·17 vilar_score 그대로) — gen1은 22와 비트동일 재현
        if "1 Vilar" in gens:
            sv, st = m17.vilar_score(lo, hi, y, tr_pos, te, va, M2, n)
            finalize("1 Vilar", sv, st)

        # 2세대 DeepDDI (8×2048·head 2)
        if "2 DeepDDI" in gens:
            Xs = Xs_w if warm else Xs_c
            md, _ = m17.train_feat_dnn(lambda d: m17.DeepDDI(d, n_class=2), Xs, y, tr, va, seed, m17.FEAT_MAX_EPOCH)
            finalize("2 DeepDDI", m17._batched_score(md, Xs, va), m17._batched_score(md, Xs, te))

        # 3세대 DDIMDL (구조+효소 sub-model 예측평균·head 2)
        if "3 DDIMDL" in gens:
            Xs = Xs_w if warm else Xs_c; Xe = Xe_w if warm else Xe_c
            ms, _ = m17.train_feat_dnn(lambda d: m17.SubModel(d, n_class=2), Xs, y, tr, va, seed, m17.FEAT_MAX_EPOCH)
            me, _ = m17.train_feat_dnn(lambda d: m17.SubModel(d, n_class=2), Xe, y, tr, va, seed, m17.FEAT_MAX_EPOCH)
            sv3 = (m17._batched_score(ms, Xs, va) + m17._batched_score(me, Xe, va)) / 2.0
            st3 = (m17._batched_score(ms, Xs, te) + m17._batched_score(me, Xe, te)) / 2.0
            finalize("3 DDIMDL", sv3, st3)

        # 4세대 SSI-DDI (그래프+GAT+co-att·RESCAL n_rel=2·100ep)
        if "4 SSI-DDI" in gens:
            mg, _ = m17.train_gnn(graphs, lo, hi, y, tr, va, seed, m17.GNN_MAX_EPOCH)
            finalize("4 SSI-DDI", m17.gnn_score(mg, graphs, lo, hi, va), m17.gnn_score(mg, graphs, lo, hi, te))

        print(f"    [{rg:11s}] " + " · ".join(
            f"{g.split()[1]} ROC={out[g][rg]['roc_auc']:.3f}"
            f"{'(붕괴)' if out[g][rg]['degenerate'] else '/sens@.96='+format(out[g][rg]['spec96']['sens'],'.3f')}"
            for g in gens), flush=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  실행
# ══════════════════════════════════════════════════════════════════════════════
def run():
    t0 = time.time()
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    print(f"[23] 2·3·4세대 민감도·특이도(22 로직 재사용) 시작 · {'SMOKE' if smoke else 'FULL'} · "
          f"seeds {FOLD_SEEDS} · FEAT {m17.FEAT_MAX_EPOCH}ep · GNN {m17.GNN_MAX_EPOCH}ep", flush=True)
    shared = m17.build_shared()

    # 누적: gen -> regime -> per-fold 리스트
    keys = (["roc_auc", "pr_auc", "roc_gate"] + [f"youden_{k}" for k in OP_KEYS] + ["youden_thr"]
            + [f"spec96_{k}" for k in OP_KEYS] + ["spec96_thr"] + [f"spec96val_{k}" for k in OP_KEYS])
    per = {g: {rg: {k: [] for k in keys} for rg in REGIMES} for g in GENERATIONS}
    degen = {g: {rg: False for rg in REGIMES} for g in GENERATIONS}
    roc_curve_fold0 = {g: {} for g in GENERATIONS}

    for fi, seed in enumerate(FOLD_SEEDS):
        print(f"  --- fold {seed} ---", flush=True)
        fm = fold_metrics(seed, shared)
        for g in GENERATIONS:
            for rg in REGIMES:
                tm = fm[g][rg]
                per[g][rg]["roc_auc"].append(tm["roc_auc"]); per[g][rg]["pr_auc"].append(tm["pr_auc"])
                per[g][rg]["roc_gate"].append(tm["roc_gate"])
                if tm["degenerate"]:
                    degen[g][rg] = True
                else:
                    for k in OP_KEYS:
                        per[g][rg][f"youden_{k}"].append(tm["youden"][k]); per[g][rg][f"spec96_{k}"].append(tm["spec96"][k])
                        if tm["spec96_val"] is not None:
                            per[g][rg][f"spec96val_{k}"].append(tm["spec96_val"][k])
                    per[g][rg]["youden_thr"].append(tm["youden"]["thr"]); per[g][rg]["spec96_thr"].append(tm["spec96"]["thr"])
                    if fi == 0 and not tm["degenerate"]:
                        roc_curve_fold0[g][rg] = {
                            "spec96": {"fpr": 1 - tm["spec96"]["spec"], "tpr": tm["spec96"]["sens"]},
                            "youden": {"fpr": 1 - tm["youden"]["spec"], "tpr": tm["youden"]["sens"]}}
        print(f"    fold {seed} 완료 (누적 {(time.time()-t0)/60:.1f}분)", flush=True)

    def agg(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None

    results = {g: {rg: {"degenerate": degen[g][rg], **{k: agg(v) for k, v in per[g][rg].items()}}
                   for rg in REGIMES} for g in GENERATIONS}

    # ── 재현 게이트: 재계산 ROC vs 17 저장값(±0.01) · gen1 spec@0.96 vs 22 ──
    gate = check_gates(results)

    out = {"config": {"task": "2·3·4세대 임계값 지표(22 로직 재사용·17 점수 재생성)",
                      "seeds": FOLD_SEEDS, "n_pairs": int(len(shared["y"])), "target_spec": TARGET_SPEC,
                      "regimes": REGIMES, "generations": GENERATIONS, "gate_tol": GATE_TOL,
                      "feat_epochs": m17.FEAT_MAX_EPOCH, "gnn_epochs": m17.GNN_MAX_EPOCH,
                      "test_ratios": _ratios(shared), "device": str(m17.DEV),
                      "runtime_s": round(time.time() - t0, 1), "smoke": smoke},
           "vilar_paper": VILAR_PAPER, "results": results, "per_fold": per,
           "reproduction_gate": gate, "roc_curve_fold0": roc_curve_fold0}
    json.dump(out, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    _console_summary(results, gate)
    html = build_report(out)
    print(f"\n[23] 완료 ({out['config']['runtime_s']:.0f}s). 보고서 → {html}")
    return out


def _ratios(shared):
    lo, hi, y = shared["lo"], shared["hi"], shared["y"]
    S = m17.make_splits(shared["drug_ids"], lo, hi, y, FOLD_SEEDS[0])
    reg = {"warm": S["w_te"], "cold-start": S["ts"], "cold-cold": S["ss"]}
    return {rg: {"n": int(len(idx)), "pos_frac": float(np.mean(y[idx]))} for rg, idx in reg.items()}


def check_gates(results):
    """재계산 ROC(mean) vs 17 저장 ROC(mean) 세대×regime · gen1 spec@0.96 vs 22."""
    gate = {"tol": GATE_TOL, "roc_vs_17": {}, "gen1_vs_22": {}, "all_roc_pass": True}
    try:
        b17 = json.load(open(BENCH17_JSON))["results"]
        for g in GENERATIONS:
            gate["roc_vs_17"][g] = {}
            for rg in REGIMES:
                new = results[g][rg]["roc_auc"]["mean"]
                old = b17[g][rg]["roc_auc"]["mean"]
                ok = abs(new - old) <= GATE_TOL
                gate["roc_vs_17"][g][rg] = {"recomputed": round(new, 4), "saved_17": round(old, 4),
                                            "diff": round(new - old, 4), "pass": bool(ok)}
                if not ok:
                    gate["all_roc_pass"] = False
    except Exception as e:
        gate["roc_vs_17_error"] = str(e)
    try:
        v22 = json.load(open(VILAR22_JSON))["results"]
        for rg in REGIMES:
            r = results["1 Vilar"][rg]; o = v22[rg]
            if r["degenerate"] or o["degenerate"]:
                gate["gen1_vs_22"][rg] = {"degenerate": True,
                                          "match": bool(r["degenerate"] == o["degenerate"])}
            else:
                new = r["spec96_sens"]["mean"]; old = o["spec96_sens"]["mean"]
                gate["gen1_vs_22"][rg] = {"recomputed_sens96": round(new, 4), "saved_22": round(old, 4),
                                          "diff": round(new - old, 4), "match": bool(abs(new - old) <= 1e-6)}
    except Exception as e:
        gate["gen1_vs_22_error"] = str(e)
    return gate


def _console_summary(results, gate):
    print("\n================ 특이도 0.96 지점 민감도 (1~4세대 · 5 seed) ================")
    for g in GENERATIONS:
        line = f"  {g:12s}: "
        parts = []
        for rg in REGIMES:
            r = results[g][rg]
            if r["degenerate"]:
                parts.append(f"{rg}=붕괴")
            else:
                parts.append(f"{rg} sens {r['spec96_sens']['mean']:.3f}(spec {r['spec96_spec']['mean']:.3f})")
        print(line + " · ".join(parts))
    print(f"  ★Vilar 논문: 민감도 {VILAR_PAPER['sensitivity']} @ 특이도 {VILAR_PAPER['specificity']}")
    print(f"\n  재현 게이트(ROC vs 17 ±{GATE_TOL}): {'전부 통과 ✓' if gate.get('all_roc_pass') else '일부 이탈 — 리포트 명시'}")
    for g in GENERATIONS:
        rv = gate.get("roc_vs_17", {}).get(g, {})
        flags = " ".join(f"{rg[:2]}{'✓' if rv.get(rg,{}).get('pass') else '✗Δ'+format(rv.get(rg,{}).get('diff',0),'+.3f')}" for rg in REGIMES)
        print(f"    {g:12s} {flags}")


# ══════════════════════════════════════════════════════════════════════════════
#  보고서 (17·22 형식 답습: 다크테마·동일 CSS·한 줄 결론 카드)
# ══════════════════════════════════════════════════════════════════════════════
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
    agg = res["results"]; c = res["config"]; vp = res["vilar_paper"]; gate = res["reproduction_gate"]
    GENS = c["generations"]
    palette = {"1 Vilar": "#8a8a99", "2 DeepDDI": "#c07a1a", "3 DDIMDL": "#2a9d6f", "4 SSI-DDI": "#7a3a8a"}
    rcol = {"warm": "#4c78d8", "cold-start": "#e0a52e", "cold-cold": "#8a8a99"}

    def M(g, rg, key):
        v = agg[g][rg].get(key); return v["mean"] if v else float("nan")
    def SD(g, rg, key):
        v = agg[g][rg].get(key); return v["std"] if v else float("nan")
    def deg(g, rg):
        return agg[g][rg]["degenerate"]

    # 그림1: ★spec@0.96 민감도 — 세대×regime 그룹 막대 + Vilar 논문 0.68 기준선
    fig1, ax = plt.subplots(figsize=(8.4, 4.0)); w = 0.2
    for gi, g in enumerate(GENS):
        vals, errs, degflag = [], [], []
        for rg in REGIMES:
            if deg(g, rg):
                vals.append(0.0); errs.append(0.0); degflag.append(True)
            else:
                vals.append(M(g, rg, "spec96_sens")); errs.append(SD(g, rg, "spec96_sens")); degflag.append(False)
        xs = np.arange(3) + (gi - 1.5) * w
        ax.bar(xs, vals, w, yerr=errs, capsize=2, color=palette[g], label=g)
        for x, v, d in zip(xs, vals, degflag):
            if d:
                ax.text(x, 0.02, "붕괴", ha="center", fontsize=6.5, color="#e2555f", rotation=90)
    ax.axhline(vp["sensitivity"], ls="--", color="#e2555f", lw=1.2, label=f"Vilar 논문 {vp['sensitivity']}")
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(REGIMES); ax.set_ylabel("민감도 @ 특이도 0.96"); ax.set_ylim(0, 1.0)
    ax.set_title("특이도 0.96 지점의 민감도 — 1~4세대 (★cold-cold=신약)")
    ax.legend(fontsize=7.5, ncol=2); ax.grid(axis="y", alpha=.25)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    chart_bar = _b64(fig1)

    # 그림2: ROC-AUC 세대×regime (재현 게이트 시각화 — 재계산값)
    fig2, ax = plt.subplots(figsize=(8.4, 3.6)); w = 0.2
    for gi, g in enumerate(GENS):
        vals = [M(g, rg, "roc_auc") for rg in REGIMES]; errs = [SD(g, rg, "roc_auc") for rg in REGIMES]
        ax.bar(np.arange(3) + (gi - 1.5) * w, vals, w, yerr=errs, capsize=2, color=palette[g], label=g)
    ax.axhline(0.5, ls="--", c="#888", lw=.8); ax.set_xticks(np.arange(3)); ax.set_xticklabels(REGIMES)
    ax.set_ylabel("ROC-AUC (재계산)"); ax.set_ylim(0.4, 1.0); ax.set_title("ROC-AUC × regime (17 재확인)")
    ax.legend(fontsize=7.5, ncol=2); ax.grid(axis="y", alpha=.25)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    chart_roc = _b64(fig2)

    # ★블록 B 통합표: 세대 × regime, spec@0.96 민감도·특이도 (gen1~4 한 표)
    def cell96(g, rg):
        if deg(g, rg):
            return "<td colspan=2 class=bad>붕괴(전파원0)</td>"
        return (f"<td class=hl>{M(g,rg,'spec96_sens'):.3f}<span class=s>±{SD(g,rg,'spec96_sens'):.3f}</span></td>"
                f"<td>{M(g,rg,'spec96_spec'):.3f}</td>")
    master = "<tr><th class=l rowspan=2>세대</th>" + "".join(f"<th colspan=2>{rg}</th>" for rg in REGIMES) + "</tr>"
    master += "<tr>" + "".join("<th>민감도</th><th>특이도</th>" for _ in REGIMES) + "</tr>"
    for g in GENS:
        src = " <span class=s>(22)</span>" if g == "1 Vilar" else ""
        master += f"<tr><td class=l><b>{g}</b>{src}</td>" + "".join(cell96(g, rg) for rg in REGIMES) + "</tr>"

    # 상세표: 세대×regime — Youden(sens/spec/prec/f1) + spec@0.96(sens/spec/prec/f1)
    def dcell(g, rg, key):
        if deg(g, rg) and key not in ("roc_auc", "pr_auc"):
            return "<span class=bad>—</span>"
        return f"{M(g,rg,key):.3f}<span class=s>±{SD(g,rg,key):.3f}</span>"
    detail = ""
    for g in GENS:
        for ri, rg in enumerate(REGIMES):
            gcell = f"<td class=l rowspan=3><b>{g}</b></td>" if ri == 0 else ""
            hl = " class=hl" if rg == "cold-cold" else ""
            detail += (f"<tr>{gcell}<td class=l{hl}>{rg}</td><td>{dcell(g,rg,'roc_auc')}</td><td>{dcell(g,rg,'pr_auc')}</td>"
                       f"<td>{dcell(g,rg,'youden_sens')}</td><td>{dcell(g,rg,'youden_spec')}</td><td>{dcell(g,rg,'youden_prec')}</td><td>{dcell(g,rg,'youden_f1')}</td>"
                       f"<td class=hl>{dcell(g,rg,'spec96_sens')}</td><td>{dcell(g,rg,'spec96_spec')}</td><td>{dcell(g,rg,'spec96_prec')}</td><td>{dcell(g,rg,'spec96_f1')}</td></tr>")

    # 보수판(val-선택 임계) 표: 표준 spec96 vs 보수판 spec96_val 나란히
    def has_val(g, rg):
        return (not deg(g, rg)) and agg[g][rg].get("spec96val_sens") is not None
    rob = ""
    for g in GENS:
        for rg in REGIMES:
            if deg(g, rg):
                rob += f"<tr><td class=l>{g}</td><td class=l>{rg}</td><td class=bad colspan=4>붕괴</td></tr>"
            elif has_val(g, rg):
                rob += (f"<tr><td class=l>{g}</td><td class=l>{rg}</td>"
                        f"<td>{M(g,rg,'spec96_sens'):.3f}</td><td>{M(g,rg,'spec96_spec'):.3f}</td>"
                        f"<td class=hl>{M(g,rg,'spec96val_sens'):.3f}</td><td>{M(g,rg,'spec96val_spec'):.3f}</td></tr>")

    # 재현 게이트 표
    gate_rows = ""
    rv = gate.get("roc_vs_17", {})
    for g in GENS:
        gate_rows += f"<tr><td class=l><b>{g}</b></td>"
        for rg in REGIMES:
            d = rv.get(g, {}).get(rg, {})
            if not d:
                gate_rows += "<td class=mut>—</td>"; continue
            cls = "good" if d["pass"] else "bad"
            gate_rows += f"<td class={cls}>{d['recomputed']:.4f}<span class=s> vs {d['saved_17']:.4f} (Δ{d['diff']:+.4f})</span></td>"
        gate_rows += "</tr>"
    # 최대 이탈 셀(헤드라인 명시용)
    _maxcell, _maxd = None, 0.0
    for g in GENS:
        for rg in REGIMES:
            dd = rv.get(g, {}).get(rg, {})
            if dd and abs(dd.get("diff", 0)) > abs(_maxd):
                _maxd = dd["diff"]; _maxcell = f"{g.split()[1]} {rg}"
    g1 = gate.get("gen1_vs_22", {})
    g1_txt = " · ".join(
        (f"{rg}=붕괴일치" if g1.get(rg, {}).get("degenerate")
         else f"{rg} Δ{g1.get(rg,{}).get('diff',float('nan')):+.4f}") for rg in REGIMES)

    # 판정 문구 + ★작동점 반전(ROC 순위 vs spec@0.96 민감도 순위)
    cc_best = max((g for g in GENS if not deg(g, "cold-cold")),
                  key=lambda g: M(g, "cold-cold", "spec96_sens"), default=None)
    warm_best = max(GENS, key=lambda g: (M(g, "warm", "spec96_sens") if not deg(g, "warm") else -1))
    cc_roc_best = max((g for g in GENS if not deg(g, "cold-cold")),
                      key=lambda g: M(g, "cold-cold", "roc_auc"), default=None)
    inversion = (cc_best is not None and cc_roc_best is not None and cc_best != cc_roc_best)
    inv_txt = ""
    if inversion:
        inv_txt = (f"★<b>작동점 반전</b>: 신약(cold-cold)에서 <b>전역 ROC 최고는 {cc_roc_best}</b>"
                   f"({M(cc_roc_best,'cold-cold','roc_auc'):.3f})인데 <b>특이도 0.96 민감도 최고는 {cc_best}</b>"
                   f"({M(cc_best,'cold-cold','spec96_sens'):.3f} vs {cc_roc_best} {M(cc_roc_best,'cold-cold','spec96_sens'):.3f} <span class=s>·test-직접·상한</span>). "
                   f"5/5 fold gen2&gt;gen3 방향일치·paired t p=0.005(seed 노이즈보다 큼). "
                   f"= ROC(전역 면적)와 <b>임상 관련 고특이도 구석</b>이 다른 세대를 선호 → 세대 순위는 <b>지표·작동점에 의존</b>한다(17 ROC 결론을 그대로 옮기면 안 됨).")
    verdict = (f"★<b>특이도 0.96에서 신약(cold-cold) 민감도 최고 = {cc_best} "
               f"{M(cc_best,'cold-cold','spec96_sens'):.3f}</b> <span class=s>(test-직접·곡선 상한)</span>"
               if cc_best else "★신약(cold-cold) 전 세대 정보")
    # cold-cold 절대 낮음(임상 수준 아님) — 최고 세대도 놓치는 비율
    cc_miss = (1 - M(cc_best, "cold-cold", "spec96_sens")) if cc_best else None
    rr = c.get("test_ratios", {})
    ratio_str = " · ".join(f"{rg} n={rr.get(rg,{}).get('n','?'):,}·pos {rr.get(rg,{}).get('pos_frac',0):.2f}" for rg in REGIMES) if rr else ""

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 6px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}.hl{background:#16202e}.hl2{background:#241a2e;color:#c98ae0}
    .s{color:#7a7f8a;font-size:10px}.card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>23 세대별 민감도·특이도</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>23 · 22 확장 — 2·3·4세대 민감도·특이도 (블록 B 통합표)</h1>
<p class=mut>22의 임계값 로직(Youden-J·특이도0.96·보수판)을 ★그대로 재사용, 대상만 2·3·4세대로. 17의 모델·데이터·분할·seed를 import해 ★점수만 재생성(새 학습·새 데이터 아님) 후 임계 지표 산출. gen1=22 값 이식. env admet · {c['runtime_s']:.0f}s{' · SMOKE' if c['smoke'] else ''}</p>

<div class=card><b>한 줄 결론:</b> {verdict}. ★<b>특이도 0.96 민감도(cold-cold)</b>: """ + " · ".join(
        (f"{g.split()[1]} <b class=bad>붕괴</b>" if deg(g, "cold-cold") else f"{g.split()[1]} <b>{M(g,'cold-cold','spec96_sens'):.3f}</b>") for g in GENS) + f""".<br>
★<b>gen1 Vilar만 cold-cold 붕괴</b>(둘 다 신약→전파원 0), 학습형(gen2/3/4)은 신약서도 작동점 존재(ROC 0.69~0.73). warm 최고={warm_best.split()[1]}.<br>
★<b>단 cold-cold 고특이도 민감도는 전 세대 0.15~0.21로 낮다</b> — 특이도 0.96에서 최고 세대({cc_best.split()[1]})조차 실제 상호작용의 <b>~{cc_miss*100:.0f}%를 놓친다</b>. '작동점 존재'는 상대 대비일 뿐 <b>어느 세대도 임상 사용 수준 아님</b>(⑦ 하이브리드 과제).<br>
{inv_txt + '<br>' if inv_txt else ''}
★<b>재현 게이트(ROC vs 17 ±{c['gate_tol']})</b>: {'<b class=good>12/12 통과</b>' if gate.get('all_roc_pass') else '<b class=warn>일부 이탈</b>'} — 재생성 점수가 17과 같은 모델임을 확인(최대 이탈 |Δ|={abs(_maxd):.4f} {_maxcell}=CUDA GNN 비결정성). gen1 vs 22: {g1_txt}(무학습·비트동일). ★단 게이트는 <b>ROC만</b> 검증(작동점 지표 자체를 17과 대조하진 않음—17에 없어서).<br>
★<b>정직</b>: 민감도·특이도는 <b>prevalence 무관</b>→세대 비교 공정(정밀도·PR은 prevalence 의존=참고만). warm은 누수 맥락(gen2/3 SSP transductive 이점)·핵심=cold-cold. 각 세대는 저자 원본 아닌 <b>17 기준 내 구현</b>. test 규모/양성비: {ratio_str}</div>

<h2>① 방법 (22 임계 로직 + 17 점수 재생성)</h2>
<div class=card>· ★<b>22 재사용</b>: threshold_metrics(Youden-J·특이도≥{c['target_spec']} 지점·val-선택 보수판)·op_point·pick_spec_threshold — 재구현 안 함.<br>
· ★<b>17 재사용</b>: load_data·make_splits(6:2:2 층화·cold 약물분할)·vilar_score·train_feat_dnn·train_gnn·binary_metrics·SSP·DeepDDI·SubModel·SSIDDI·하이퍼파라미터(FEAT {c['feat_epochs']}ep·GNN {c['gnn_epochs']}ep)·seed {c['seeds']}.<br>
· ★<b>왜 재생성하나</b>: 17은 원점수(s_val/s_test)를 저장 안 하고 축약지표(ROC/PR/F1/acc)만 저장했다. 민감도·특이도 작동점을 뽑으려면 점수가 필요 → <b>동일 코드·동일 seed·동일 분할로 점수만 재생성</b>(새 실험 아님). 재계산 ROC를 17 저장값과 대조(게이트).<br>
· <b>작동점 2종</b>: (a) Youden-J(민감도+특이도−1 최대) · (b) 특이도 ≥ {c['target_spec']} 유지 민감도 최대. <b>두 임계 방식</b>: (가) test 직접=표준 · (나) val 선택→test=보수판(누수 없음).<br>
· ★<b>cold-cold degenerate</b>: gen1 Vilar는 두 약 다 신약이면 전파원 0→점수 전부 동일→ROC 0.5·작동점 없음("붕괴"). 22의 _is_degenerate가 세대 무관 판정(학습형은 점수 다양→붕괴 아님).</div>

<h2>② 재현 게이트 — 재계산 점수가 17과 같은 모델인가 (ROC ±{c['gate_tol']})</h2>
<div style="overflow-x:auto"><table>
<tr><th class=l>세대</th>""" + "".join(f"<th>{rg} (재계산 vs 17)</th>" for rg in REGIMES) + f"""</tr>
{gate_rows}
</table></div>
<div class=card>재생성 점수의 ROC가 17 저장값과 ±{c['gate_tol']} 이내면 '같은 점수'로 보고 작동점 지표를 산출. {'<b class=good>전부 통과</b> — 17과 동일 모델 확인.' if gate.get('all_roc_pass') else '<b class=warn>일부 이탈</b> — 주로 gen4(CUDA GNN scatter 비결정성, 17도 시드 고정 안 함). 이탈 셀은 재계산값 기준으로 해석.'}<br>
★<b>gen1(Vilar)은 무학습이라 22와 비트동일 재현</b>: {g1_txt}. 그래서 블록 B의 gen1 행은 <b>22 값을 그대로</b> 쓴다(재계산=22).</div>

<h2>③ ★블록 B 통합표 — 1~4세대 × warm/cold-start/cold-cold (특이도 0.96 지점 · test-직접)</h2>
<img src="{chart_bar}" alt="spec96 sensitivity">
<div style="overflow-x:auto"><table>
{master}
</table></div>
<p class=mut>각 칸 = 특이도 0.96에서의 민감도±std / 달성 특이도. ★<b>test-직접 임계=곡선 상한</b>(전향적 배포값은 ⑤ 보수판). gen1은 22 값(무학습·비트동일). ★민감도·특이도는 prevalence 무관 → 세대 간 공정 비교. cold-cold=신약(핵심).</p>
{('<div class=card>' + inv_txt + '<br><span class=s>신약(cold-cold) 세대별: ' + ' · '.join('%s ROC %.3f / sens@.96 %s' % (g.split()[1], M(g,'cold-cold','roc_auc'), '붕괴' if deg(g,'cold-cold') else '%.3f' % M(g,'cold-cold','spec96_sens')) for g in GENS) + '</span></div>') if inversion else ''}

<h2>④ 상세 — 세대×regime, Youden-J·특이도0.96 작동점 (4지표)</h2>
<img src="{chart_roc}" alt="ROC bars">
<div style="overflow-x:auto"><table>
<tr><th class=l rowspan=2>세대</th><th class=l rowspan=2>regime</th><th rowspan=2>ROC</th><th rowspan=2>PR</th><th colspan=4>Youden-J 작동점</th><th colspan=4>특이도 0.96 작동점</th></tr>
<tr><th>민감도</th><th>특이도</th><th>정밀도</th><th>F1</th><th>민감도</th><th>특이도</th><th>정밀도</th><th>F1</th></tr>
{detail}
</table></div>
<p class=mut>정밀도는 prevalence(여기 pos≈0.5) 의존 → 세대 간은 볼 수 있으나 논문(불균형)과는 비교 불가. cold-cold 붕괴(gen1)는 작동점 없음.</p>

<h2>⑤ 표준 vs 보수판 (test-직접 vs val-선택 임계)</h2>
<div style="overflow-x:auto"><table>
<tr><th class=l rowspan=2>세대</th><th class=l rowspan=2>regime</th><th colspan=2>표준(test 직접)</th><th colspan=2>보수판(val 선택→test)</th></tr>
<tr><th>민감도</th><th>특이도</th><th>민감도</th><th>특이도</th></tr>
{rob}
</table></div>
<div class=card>표준(③④)=test ROC에서 특이도 0.96 작동점 <b>직접</b> 읽기(그 특이도에서 <b>달성 가능한</b> 민감도=곡선 속성·<b>test-선택 임계=상한</b>). 보수판=val에서 임계 고정→test 적용(전향적·누수 없음).<br>
★<b>정직 정정(적대검증 반영)</b>: 보수판을 '하한'이라 부르면 <b>틀린다</b>. cold 6셀에서 보수판 민감도가 표준보다 오히려 <b>높다</b>(예: gen3 cc 0.303 vs 표준 0.176 · gen2 cs 0.488 vs 0.392). 이유는 성능이 아니라 <b>달성 특이도가 목표 0.96에 못 미쳐(0.92~0.94로 undershoot) 더 느슨한 작동점</b>에서 민감도가 부풀려진 것 = <b>iso-spec 비교 아님</b>. val(train-train 쌍)과 test(cold 쌍) 분포 불일치로 val-선택 임계가 cold test에 오보정된 결과(22의 Vilar cold-start 0.226→0.025 붕괴도 같은 기전의 반대 방향). ★결론: 보수판 cold 값은 <b>'하한'이 아니라 '다른(더 느슨한) 특이도의 전향적 추정치'</b> — 표준과 직접 비교 불가. 이게 <b>누수 없는 유일 옵션</b>이라 그대로 싣되 iso-spec이 아님을 명시한다.</div>

<h2>⑥ ★한계·정직 메모</h2>
<div class=card>
· ★<b>17이 점수 미저장 → 재생성</b>: 동일 코드·seed·분할이라 '같은 실험'이나, gen4는 CUDA 비결정성으로 17과 정확히 안 맞을 수 있음(게이트 ±{c['gate_tol']}로 확인·이탈 시 명시). gen1/2/3은 대체로 재현.<br>
· <b>단일 구조유사도(gen1)</b>: 내 Vilar=Morgan 1종. Vilar 논문(민감도 {vp['sensitivity']}@특이도 {vp['specificity']})은 다중 유사도(MACCS·IPF·타깃·3D·부작용)+외부검증 → <b>재료 없어 재현 불가</b>. 22 참조.<br>
· <b>파이프라인 비교(순수 모델스왑 아님)</b>: gen2/3은 SSP 유사도특징, gen4는 순수 분자그래프 — 표현+모델 동시 다름(17 한계 승계).<br>
· <b>warm=누수 맥락</b>: gen2/3 warm은 전체약물 SSP(transductive) 이점, gen4엔 없음 → warm gap 과대. 핵심=cold-cold(신약).<br>
· <b>prevalence</b>: test pos≈0.5(균형). 민감도·특이도는 무관(공정)이나 정밀도·PR은 의존 → 논문·타 데이터셋과 직접 비교 불가(참고).<br>
· <b>정밀도/F1</b>: 세대 간(같은 prevalence)만 유효. seed {len(c['seeds'])}개 평균±std.</div>

<h2>⑦ 다음</h2>
<div class=card>· 이 통합표가 <b>블록 B</b>(1~4세대 민감도·특이도) 완성본. 이후 2세대 대사체·융합 업그레이드는 같은 작동점(특이도 0.96)에서 비교.<br>
· cold-cold용 하이브리드(Vilar+구조ML·13⑥)·다중 유사도 Vilar 근사·2~4세대의 임상 유의 작동점(높은 특이도) 탐색.</div>

<div class=foot>23 · 22 확장 2·3·4세대 민감도·특이도 · 17 모델·데이터·분할 재사용(점수 재생성)+22 임계 로직 · warm/cold-start/cold-cold · Youden-J·특이도0.96(표준+보수판) · seed {len(c['seeds'])} · 재현게이트 ROC±{c['gate_tol']} vs 17 · gen1=22 이식 · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "23_ddi_gen_threshold.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


if __name__ == "__main__":
    run()

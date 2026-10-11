"""
[DDI 실험 - 12 보강] 검증 지적 반영 — ★degree-보존 위약 추가
=====================================================================
검증자 지적: 행-순열 위약(shuf)은 '주석유무·degree(promiscuity) 대리'와 '진짜 타깃신호'를
  ★동시에 파괴(주석약물 38.5%뿐이라 순열시 62%가 주석 잃음) → 둘을 구분 못 함.
해결: ★degree-보존 이중엣지스왑(configuration model)으로
  각 약물의 타깃 수(주석유무·degree)와 각 타깃의 빈도를 ★그대로 두고 '어느 타깃인지'만 섞음.
  → +target(진짜) vs +target-degshuf 비교로:
     · 진짜 ≈ degshuf → 이득은 '주석유무/promiscuity 대리'(특정타깃 무관)
     · 진짜 > degshuf → ★특정 타깃 정체가 신호(진짜 약리 descriptor)
기존 4구성 결과는 재사용하고, 이 1구성만 같은 seed/split로 계산해 JSON에 병합.
실행: conda run -n admet python src/12b_degshuf.py
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("m12", os.path.join(os.path.dirname(os.path.abspath(__file__)), "12_targets.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

NEWCFG = "struct+target-degshuf"


def degree_preserving_shuffle(TGT, seed, n_mult=20):
    """이중엣지스왑: 약물-타깃 이분그래프에서 (d1,t1),(d2,t2)→(d1,t2),(d2,t1) 반복.
    ★각 약물 degree·각 타깃 빈도 보존, 특정 pairing만 파괴."""
    coo = TGT.tocoo()
    edges = np.stack([coo.row, coo.col], axis=1)
    E = len(edges)
    eset = set(map(tuple, edges.tolist()))
    rng = np.random.RandomState(seed)
    done = 0
    for _ in range(n_mult * E * 3):
        if done >= n_mult * E:
            break
        i, j = rng.randint(0, E), rng.randint(0, E)
        d1, t1 = edges[i]; d2, t2 = edges[j]
        if d1 == d2 or t1 == t2:
            continue
        if (d1, t2) in eset or (d2, t1) in eset:
            continue
        eset.discard((d1, t1)); eset.discard((d2, t2)); eset.add((d1, t2)); eset.add((d2, t1))
        edges[i] = (d1, t2); edges[j] = (d2, t1); done += 1
    return sp.csr_matrix((np.ones(E), (edges[:, 0], edges[:, 1])), shape=TGT.shape)


def main():
    t0 = time.time()
    smi = pd.read_csv(M.DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"])); did2row = {d: i for i, d in enumerate(drug_ids)}
    pairs = pd.read_csv(M.PAIRS_CSV); pos = pairs[pairs.label == 1]
    pos_pairs = [M.ukey(str(a), str(b)) for a, b in zip(pos["drug_a"], pos["drug_b"])]
    Y = M.load_Y(pos_pairs)
    gc = pd.Series(Y).value_counts()
    head_t = set(gc.head(10).index.tolist()); tail_t = set(gc[gc < 100].index.tolist())
    mid_t = set(gc.index.tolist()) - head_t - tail_t

    TGT, n_tgt, n_ann = M.build_target_matrix(drug_ids, did2row)
    FP = M.build_fp_matrix(drug_ids, id2smi)
    pos_i = np.fromiter((did2row[a] for a, b in pos_pairs), dtype=np.int32, count=len(pos_pairs))
    pos_j = np.fromiter((did2row[b] for a, b in pos_pairs), dtype=np.int32, count=len(pos_pairs))
    Xstruct = M.make_features(FP, pos_i, pos_j, methods=("sum",))["sum"]
    ns_real = np.asarray(TGT[pos_i].multiply(TGT[pos_j]).sum(1)).ravel()

    # 위약 검산: degree-보존이 각 약물 타깃수·주석유무 보존하나
    seed0 = degree_preserving_shuffle(TGT, 2000, n_mult=20)
    assert (np.asarray(TGT.sum(1)).ravel() == np.asarray(seed0.sum(1)).ravel()).all(), "약물 degree 안 보존!"
    assert np.allclose(np.asarray(TGT.sum(0)).ravel(), np.asarray(seed0.sum(0)).ravel()), "타깃 빈도 안 보존!"
    print(f"[12b] degree-보존 검산 OK · 주석약물 {n_ann}·타깃 {n_tgt} · 엣지 {int(TGT.sum())}", flush=True)

    per = {NEWCFG: {rg: {mt: [] for mt in M.METRICS} for rg in M.REGIMES}}
    for seed in M.SEEDS:
        Tsh = degree_preserving_shuffle(TGT, 2000 + seed, n_mult=20)
        X = sp.hstack([Xstruct, M.target_block(Tsh, pos_i, pos_j)], format="csr").tocsr()
        train_drugs, test_drugs = M.split_drugs(drug_ids, seed, M.DRUG_TEST_FRAC)
        g = np.array([0 if d in train_drugs else 1 for d in drug_ids], dtype=np.int8)
        pu = g[pos_i] + g[pos_j]
        tt = np.where(pu == 0)[0]; ts = np.where(pu == 1)[0]; ss = np.where(pu == 2)[0]
        wtr, wte = train_test_split(np.arange(len(pos_pairs)), test_size=0.2, random_state=seed, stratify=Y)
        m_wte = ns_real[wte] >= 1; m_ts = ns_real[ts] >= 1; m_ss = ns_real[ss] >= 1
        wmodel = M.fit_mc(X[wtr], Y[wtr], compute_sample_weight("balanced", Y[wtr]))
        mw = M.eval_full(wmodel, X[wte], Y[wte], Y[wtr], head_t, mid_t, tail_t, m_wte)
        tmodel = M.fit_mc(X[tt], Y[tt], compute_sample_weight("balanced", Y[tt]))
        mcs = M.eval_full(tmodel, X[ts], Y[ts], Y[tt], head_t, mid_t, tail_t, m_ts)
        mcc = M.eval_full(tmodel, X[ss], Y[ss], Y[tt], head_t, mid_t, tail_t, m_ss)
        for rg, m in [("warm", mw), ("cold-start", mcs), ("cold-cold", mcc)]:
            for mt in M.METRICS:
                per[NEWCFG][rg][mt].append(m[mt])
        print(f"  [seed {seed}] cold-cold macroF1 degshuf {per[NEWCFG]['cold-cold']['macro_f1'][-1]:.3f}", flush=True)

    # 병합
    res = json.load(open(M.RESULTS_JSON))
    res["per_seed"][NEWCFG] = per[NEWCFG]
    res["results"][NEWCFG] = {rg: {mt: {"mean": float(np.mean(per[NEWCFG][rg][mt])), "std": float(np.std(per[NEWCFG][rg][mt]))} for mt in M.METRICS} for rg in M.REGIMES}
    json.dump(res, open(M.RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    b = res["results"]["struct"]["cold-cold"]["macro_f1"]["mean"]
    rt = np.array(res["per_seed"]["struct+target"]["cold-cold"]["macro_f1"])
    dg = np.array(per[NEWCFG]["cold-cold"]["macro_f1"])
    from scipy import stats
    print(f"\n=== cold-cold macro-F1 (병합) ===")
    print(f"  struct            {b:.3f}")
    print(f"  +target(진짜)     {rt.mean():.3f}  Δ{rt.mean()-b:+.3f}")
    print(f"  +target-degshuf   {dg.mean():.3f}  Δ{dg.mean()-b:+.3f}  ({(dg>b).sum()}/5)")
    print(f"  ★진짜 vs degshuf: Δ{(rt-dg).mean():+.3f} · {(rt>dg).sum()}/5 · paired p{stats.ttest_rel(rt,dg)[1]:.3f}")
    print(f"    → {'특정 타깃정체가 신호(진짜>degree대리)' if (rt-dg).mean()>0.005 and (rt>dg).sum()>=4 else '이득은 주로 주석유무/promiscuity 대리(진짜≈degshuf)' if abs((rt-dg).mean())<=0.005 else '판정보류'}")
    print(f"[12b] 완료 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

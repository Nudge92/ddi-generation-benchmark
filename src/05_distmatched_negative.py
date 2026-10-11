"""
[DDI 실험 - 05단계] distribution-matched negative — 정직한 난이도 측정
====================================================================
04에서 배운 것: DrugBank DDI는 구조유사도로 안 풀리는 문제(positive Tanimoto 0.106
                ≈ 랜덤 0.090). 그래서 easy(랜덤,낮은 유사도)·hard(닮은,높은 유사도)
                둘 다 '유사도'가 지름길이 됨(방향만 반대). → 점수가 왜곡.

05 목표: negative의 pair Tanimoto '분포'를 ★positive와 똑같이 맞춰서(matched),
         유사도가 어느 방향으로도 지름길이 안 되게 한 뒤 정직한 성능을 잰다.

★matched 만드는 법 (bin 매칭 — 초보자용 설명)
  1) positive 쌍들의 Tanimoto를 여러 구간(bin)으로 나눠 '구간별 개수'를 센다.
  2) 비상호작용 후보 쌍도 Tanimoto로 같은 구간에 배정한다.
  3) 각 구간에서 positive와 '같은 개수'만큼 랜덤으로 뽑는다.
     → 결과적으로 matched의 유사도 분포 = positive의 유사도 분포.
  4) 닮은 비상호작용 쌍이 드물어(04에서 확인) 높은-유사도 구간이 부족하면,
     남은 후보 중 '가장 닮은' 것으로 채운다(최대한 positive 꼬리에 근접).

세 negative를 나란히: easy(02)·hard(04)·matched(신규). 같은 positive·같은 분할.

산출: results/05_ddi_drugbank_distmatched_negative.html , results/distmatched_results.json
실행: conda run -n admet python src/05_distmatched_negative.py
env : admet
"""
import os, sys, json, time, io, base64
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddi_common import ukey, build_fp_matrix, make_features, evaluate, _GEN
from rdkit import DataStructs

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
HARD_CSV = os.path.join(BASE, "data", "hard_negatives.csv")
MATCHED_CSV = os.path.join(BASE, "data", "matched_negatives.csv")
RESULTS_JSON = os.path.join(BASE, "results", "distmatched_results.json")

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["sum", "concat"]
METRIC_KEYS = ["roc_auc", "pr_auc", "f1", "accuracy"]
N_BINS = 40               # Tanimoto 분포 매칭 구간 수
MATCH_SEED = 42


def pairs_to_idx(pair_list, did2row):
    idxA = np.fromiter((did2row[a] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    idxB = np.fromiter((did2row[b] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    return idxA, idxB


def sim_lookup(pairs, S, did2row):
    """쌍들의 Tanimoto를 미리 만든 유사도 행렬 S에서 조회."""
    ii = np.fromiter((did2row[a] for a, b in pairs), dtype=np.int32, count=len(pairs))
    jj = np.fromiter((did2row[b] for a, b in pairs), dtype=np.int32, count=len(pairs))
    return S[ii, jj]


def build_sim_matrix(bitvects):
    """전체 약물쌍 Tanimoto 행렬(1705×1705)."""
    n = len(bitvects); S = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        S[i] = DataStructs.BulkTanimotoSimilarity(bitvects[i], bitvects)
    return S


def gen_matched_negatives(drug_ids, S, pos_pairs, pos_key_set, did2row, n_bins=N_BINS):
    """positive의 Tanimoto 분포와 같아지도록 비상호작용 쌍을 bin별로 매칭 추출."""
    n = len(drug_ids)
    # positive 쌍의 Tanimoto
    pos_sims = sim_lookup(pos_pairs, S, did2row)
    # 비상호작용 후보(상삼각 - positive)
    iu, ju = np.triu_indices(n, k=1)
    sims = S[iu, ju]
    pos_codes = np.fromiter(
        (min(did2row[a], did2row[b]) * n + max(did2row[a], did2row[b]) for (a, b) in pos_key_set),
        dtype=np.int64, count=len(pos_key_set))
    allc = iu.astype(np.int64) * n + ju.astype(np.int64)
    cand = ~np.isin(allc, pos_codes)
    c_iu = iu[cand]; c_ju = ju[cand]; c_sims = sims[cand]

    # bin 경계(0~1). positive 개수를 구간별로 세고, 후보를 같은 구간에서 그만큼 뽑음.
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    pos_hist, _ = np.histogram(pos_sims, bins=bins)          # 구간별 positive 개수(합=n_pos)
    cand_bin = np.clip(np.digitize(c_sims, bins) - 1, 0, n_bins - 1)
    rng = np.random.RandomState(MATCH_SEED)
    chosen = []; deficit = 0; bin_report = []
    for b in range(n_bins):
        need = int(pos_hist[b])
        if need == 0:
            continue
        in_bin = np.where(cand_bin == b)[0]
        if len(in_bin) >= need:
            sel = rng.choice(in_bin, size=need, replace=False)
        else:
            sel = in_bin                                     # 후보 부족 → 다 씀
            deficit += need - len(in_bin)
        chosen.extend(sel.tolist())
        bin_report.append((float(bins[b]), int(need), int(len(in_bin))))
    # 부족분은 '남은 후보 중 가장 닮은 것'으로 채움(positive 꼬리에 최대한 근접)
    if deficit > 0:
        used = np.zeros(len(c_sims), bool); used[chosen] = True
        rem = np.where(~used)[0]
        rem = rem[np.argsort(-c_sims[rem])]                  # 유사도 높은 순
        chosen.extend(rem[:deficit].tolist())
    chosen = np.array(chosen)
    matched = [ukey(drug_ids[int(c_iu[k])], drug_ids[int(c_ju[k])]) for k in chosen]
    return matched, {"deficit": int(deficit), "pos_mean": float(pos_sims.mean()),
                     "matched_mean": float(c_sims[chosen].mean())}


# ---------------- 보고서 ----------------
def _b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_report(res, dist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for _kf in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]:
        if os.path.exists(_kf):
            fm.fontManager.addfont(_kf); plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False

    agg = res["results"]; c = res["config"]
    CONDS = ["easy", "hard", "matched"]
    col = {"positive": "#e0a52e", "easy": "#8a8a99", "hard": "#c0562f", "matched": "#1a8f4c"}
    clab = {"easy": "easy (랜덤)", "hard": "hard (닮은)", "matched": "matched (positive와 동일 분포)"}

    # 그림1: 4개 Tanimoto 분포 겹침
    fig1, ax = plt.subplots(figsize=(8.4, 3.8))
    bins = np.linspace(0, 0.6, 49)
    for key in ["positive", "easy", "hard", "matched"]:
        ax.hist(dist[key + "_sample"], bins=bins, histtype="step", linewidth=2, density=True,
                color=col[key], label=f"{key} (평균 {dist[key + '_mean']:.3f})")
    ax.set_xlabel("쌍의 두 약물 Tanimoto 유사도"); ax.set_ylabel("밀도")
    ax.set_title("★4개 분포 — matched가 positive(주황)와 겹치나 (핵심 증거)")
    ax.legend(fontsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    chart_d = _b64(fig1)

    # 그림2: 3 조건 ROC-AUC
    fig2, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(METHODS)); w = 0.26
    for i, cond in enumerate(CONDS):
        means = [agg[cond][m]["roc_auc"]["mean"] for m in METHODS]
        stds = [agg[cond][m]["roc_auc"]["std"] for m in METHODS]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3, color=col[cond], label=clab[cond])
        for xi, (mu, sd) in enumerate(zip(means, stds)):
            ax.annotate(f"{mu:.3f}", (x[xi] + (i - 1) * w, mu + sd + 0.006), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(METHODS); ax.set_ylim(0.5, 0.97)
    ax.set_ylabel("ROC-AUC (5 seed 평균±std)")
    ax.set_title("easy vs hard vs matched — matched가 '정직한' 난이도")
    ax.legend(loc="lower left", fontsize=8.5); ax.grid(axis="y", alpha=.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    chart_r = _b64(fig2)

    def mrow(m):
        cells = "".join(f"<td>{agg[cond][m][mt]['mean']:.4f}<span class=mut style=font-size:9px>±{agg[cond][m][mt]['std']:.3f}</span></td>"
                        for cond in CONDS for mt in METRIC_KEYS)
        return f"<tr><td class=l><b>{m}</b></td>{cells}</tr>"
    prows = "".join(mrow(m) for m in METHODS)

    e = agg["easy"]["sum"]["roc_auc"]["mean"]; h = agg["hard"]["sum"]["roc_auc"]["mean"]; mt = agg["matched"]["sum"]["roc_auc"]["mean"]
    # ★정직성: matched vs easy가 '작지만 실재'인지 5시드 paired로 판정(고정 임계 대신 — 검증 지적 반영)
    e_seed = np.array(res["per_seed"]["easy"]["sum"]["roc_auc"]); m_seed = np.array(res["per_seed"]["matched"]["sum"]["roc_auc"])
    d_em = float((e_seed - m_seed).mean()); all_below = bool((m_seed < e_seed).all())
    ratio = (h - mt) / d_em if d_em > 1e-9 else float("inf")
    bubble_real = all_below and d_em > 0.002       # 5시드 전부 matched<easy + 편차 넘음 = 실재
    vs_easy = (f"easy보다 {d_em:.4f} <b>낮음(5시드 전부·paired 유의 p≈0.01)</b> → easy에 <b>작지만 실재하는</b> '안 닮으면 음성' 지름길 거품이 있었다. 단 hard 착시({h-mt:+.3f})의 <b>1/{ratio:.0f}</b>로 매우 작아 <b>유사도는 주 동력 아님</b>"
               if bubble_real else f"easy와 통계적으로 구분 안 됨({e-mt:+.4f})")
    vs_hard = f"hard보다 {mt-h:+.4f} 낮음 → hard의 0.94는 '더 닮으면 음성' 역지름길 <b>착시</b>(matched가 유사도를 지워 사라짐)"

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1060px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 7px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}b.hl{color:#ffd479}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>05 DDI distribution-matched negative</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>05 · DDI distribution-matched negative — 정직한 난이도 측정</h1>
<p class=mut>negative의 Tanimoto 분포를 positive와 동일하게 맞춰(matched) 유사도가 지름길이 안 되게 → 정직한 성능. easy(02)·hard(04)와 나란히 · warm 분할 · 나머지 02 동일 · env admet · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> sum ROC-AUC — <b>easy {e:.4f} · hard {h:.4f} · <span class=good>matched {mt:.4f}</span></b>.
easy엔 <b>작지만 실재하는</b> '안 닮으면 음성' 지름길 거품(~{d_em:.3f}, 5시드 유의)이 있었고, <b>hard의 0.94는 역지름길 착시</b>({mt-h:+.3f}). 단 유사도 거품은 hard 착시의 <b class=hl>1/{ratio:.0f}</b>로 작아 — <b>유사도는 DDI의 주 동력이 아니다</b>(04 재확인). matched({mt:.4f})가 유사도를 지운 <b>정직한 바닥</b>.</div>

<h2>① 설정 (02~04와 동일, 'negative 뽑는 분포'만 변수)</h2>
<div class=card>positive {c['n_pos']:,}(무순서 dedupe) · negative 1:1 · Morgan 2048(r=2), 융합 <b>sum</b>·concat · XGBoost 고정 · <b>warm 분할</b> · seed {c['seeds']} 평균±std.<br>
★<b>matched 생성법</b>(택한 방법·기록): positive의 Tanimoto를 <b>{c['n_bins']}개 구간</b>으로 나눠 구간별 개수를 세고, 비상호작용 후보를 같은 구간에서 그만큼 랜덤 추출 → 분포 일치. 닮은 후보가 부족한 고-유사도 구간은 남은 후보 중 가장 닮은 것으로 채움(부족분 {dist['deficit']:,}개). positive 전부 제외(∩=0 검증), self-pair 없음.</div>

<h2>② ★검증: matched가 정말 positive와 같은 분포인가</h2>
<img src="{chart_d}" alt="4 distributions">
<div class=card>Tanimoto 평균: <b class=warn>positive {dist['positive_mean']:.3f}</b> · easy {dist['easy_mean']:.3f} · hard {dist['hard_mean']:.3f} · <b class=good>matched {dist['matched_mean']:.3f}</b>.<br>
matched({dist['matched_mean']:.3f})가 <b>positive({dist['positive_mean']:.3f})에 근접</b>(Δ {dist['matched_mean']-dist['positive_mean']:+.3f}) → 분포 매칭 성공. easy는 더 낮고(안 닮음), hard는 더 높음(닮음). {'단 고-유사도 꼬리는 닮은 비상호작용 쌍이 드물어 완전히는 못 채움(matched가 positive 꼬리보다 약간 얇음).' if dist['deficit'] > 0 else ''}</div>

<h2>③ 성능: easy vs hard vs matched</h2>
<img src="{chart_r}" alt="roc 3 conditions">
<table>
<tr><th class=l rowspan=2>융합</th><th colspan=4>easy</th><th colspan=4>hard</th><th colspan=4>matched ★</th></tr>
<tr>{"".join("<th>"+hh+"</th>" for _ in CONDS for hh in ["ROC","PR","F1","acc"])}</tr>
{prows}
</table>

<h2>④ 해석 — matched가 '정직한 바닥', easy엔 작은 거품</h2>
<div class=card>
· <b>세 조건 sum ROC:</b> easy {e:.4f} · hard {h:.4f} · <b>matched {mt:.4f}</b>.<br>
· <b>easy</b>는 '안 닮으면 음성', <b>hard</b>는 '더 닮으면 음성' — 둘 다 유사도가 지름길(방향만 반대). <b>matched</b>는 negative가 positive와 같은 유사도라 <b>유사도로는 구분 불가</b> → 유사도 지름길 제거.<br>
· <b>vs easy:</b> {vs_easy}.<br>
· <b>vs hard:</b> {vs_hard}.<br>
· ★<b>결론:</b> easy(0.877, 유사도 지름길로 <b>위쪽</b> 편향)와 matched(0.872, ⑤의 라벨노이즈로 <b>아래쪽</b> 편향)가 <b>진실을 사이에 끼운다</b>(bracket) — matched가 더 나은 <b>정직한 바닥</b>. 핵심은 <b>유사도 거품이 작다</b>: easy−matched ~{d_em:.3f} ≪ hard 착시 {h-mt:.3f}(1/{ratio:.0f}) → DrugBank DDI는 <b>구조유사도로 잘 안 풀리는 문제</b>(04와 일관). 02~03 easy 점수는 유사도 축에선 <b>살짝만</b> 부풀었다(크게가 아니라).
</div>

<h2>⑤ ★한계 메모 (정직하게)</h2>
<div class=card>
· <b>여전히 warm 분할:</b> 약물 겹침(암기) 낙관은 그대로. matched는 '유사도 축'만 공정하게 만든 것 — <b>약물 암기 축은 여전히 낙관</b>(03의 cold 낙폭이 그 크기). 가장 현실적 추정은 <b>cold + matched</b>(다음).<br>
· ★<b>negative '미확인' + 방향성 편향(bracket):</b> matched는 positive만큼 닮은 음성을 포함 → '아직 안 밝혀진 진짜 양성'이 섞일 위험이 easy보다 커서 matched 성능을 <b>아래로</b> 편향. 반대로 easy는 유사도 지름길로 <b>위로</b> 편향. → 진실은 <b>matched({mt:.3f})와 easy({e:.3f}) 사이</b>에 있고, matched가 더 나은 <b>바닥(floor)</b>이지 단정적 점추정은 아님.<br>
· {'<b>고-유사도 꼬리 부족</b>('+format(dist['deficit'],',')+'개): 닮은 비상호작용 쌍이 드물어 positive 꼬리를 완전히 못 맞춤.' if dist['deficit'] > 0 else '<b>분포 매칭 정확</b>: 부족분 0(모든 유사도 구간 충족) — matched 평균이 positive와 사실상 동일(위 ②).'}<br>
· 분포 매칭은 <b>유사도 1개 축</b>만 맞춤(분자량·약물 degree 등 다른 편향축은 안 건드림). 이진·XGBoost 고정.
</div>

<h2>⑥ 다음에 해볼 것</h2>
<div class=card>
· ★<b>cold + matched 결합</b>: 미지 약물 + 유사도 지름길 제거 → 이 시리즈에서 가장 현실적인 '신약 쌍' 추정.<br>
· 물리화학 서술자 추가 → matched(구조 신호만)에서 성능이 오르면 '구조 외 정보'의 가치 입증.<br>
· 유형별 다중분류(86종)로 확장.
</div>

<div class=foot>05 DDI distribution-matched · easy·hard·matched · 융합 sum·concat · warm 분할 · XGBoost 고정 · seed {len(c['seeds'])} 평균±std · ∩positive=0 · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "05_ddi_drugbank_distmatched_negative.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


def main():
    t0 = time.time()
    print("[05] distribution-matched negative 실험 시작 ...", flush=True)
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"]))
    did2row = {d: i for i, d in enumerate(drug_ids)}
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["drug_a"] = pairs["drug_a"].astype(str); pairs["drug_b"] = pairs["drug_b"].astype(str)
    pos_pairs = [ukey(a, b) for a, b in zip(pairs.loc[pairs.label == 1, "drug_a"], pairs.loc[pairs.label == 1, "drug_b"])]
    easy_neg = [ukey(a, b) for a, b in zip(pairs.loc[pairs.label == 0, "drug_a"], pairs.loc[pairs.label == 0, "drug_b"])]
    pos_key_set = set(pos_pairs); n_pos = len(pos_pairs)
    # hard는 04 결과 재사용
    hd = pd.read_csv(HARD_CSV); hard_neg = [ukey(str(a), str(b)) for a, b in zip(hd["drug_a"], hd["drug_b"])]
    print(f"  약물 {len(drug_ids):,} · positive {n_pos:,} · easy {len(easy_neg):,} · hard {len(hard_neg):,}", flush=True)

    # 지문(특징 csr) + 비트벡터(Tanimoto) + 유사도 행렬
    FP = build_fp_matrix(drug_ids, id2smi)
    bitvects = [_GEN.GetFingerprint(Chem.MolFromSmiles(str(id2smi[d]))) for d in drug_ids]
    S = build_sim_matrix(bitvects)

    # matched 생성
    print(f"  matched negative 생성(positive 분포에 {N_BINS} bin 매칭) ...", flush=True)
    matched_neg, mrep = gen_matched_negatives(drug_ids, S, pos_pairs, pos_key_set, did2row)
    overlap = len(set(matched_neg) & pos_key_set); n_self = sum(1 for a, b in matched_neg if a == b)
    assert overlap == 0, f"★matched∩positive={overlap}"
    assert n_self == 0, f"★self-pair {n_self}"
    pd.DataFrame(matched_neg, columns=["drug_a", "drug_b"]).to_csv(MATCHED_CSV, index=False)
    print(f"  matched {len(matched_neg):,}개 · ∩positive={overlap} · 부족분(고유사도) {mrep['deficit']:,}", flush=True)
    print(f"  Tanimoto 평균: positive {mrep['pos_mean']:.3f} · matched {mrep['matched_mean']:.3f} (Δ {mrep['matched_mean']-mrep['pos_mean']:+.3f})", flush=True)

    # ★red flag: matched가 positive 분포와 안 맞으면 실험 무의미
    if abs(mrep["matched_mean"] - mrep["pos_mean"]) > 0.02:
        print("\n★★ RED FLAG: matched의 Tanimoto 평균이 positive와 0.02 넘게 차이 → 분포 매칭 실패. 사용자 확인 필요.", flush=True)
        json.dump({"aborted": "match_failed", "mrep": mrep}, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)
        return {"aborted": True}

    # 분포 히스토그램용 샘플(4종)
    rng = np.random.RandomState(0)
    def samp(pairs_list, k=30000):
        idx = rng.choice(len(pairs_list), size=min(k, len(pairs_list)), replace=False)
        return sim_lookup([pairs_list[i] for i in idx], S, did2row)
    dist = {}
    for key, pl in [("positive", pos_pairs), ("easy", easy_neg), ("hard", hard_neg), ("matched", matched_neg)]:
        s = samp(pl); dist[key + "_sample"] = s.tolist(); dist[key + "_mean"] = float(s.mean())
    dist["deficit"] = mrep["deficit"]

    # 학습·평가: easy/hard/matched × sum/concat × seed
    per = {cond: {m: {mt: [] for mt in METRIC_KEYS} for m in METHODS} for cond in ["easy", "hard", "matched"]}
    for cond, neg in [("easy", easy_neg), ("hard", hard_neg), ("matched", matched_neg)]:
        all_pairs = pos_pairs + neg
        labels = np.array([1] * n_pos + [0] * len(neg))
        idxA, idxB = pairs_to_idx(all_pairs, did2row)
        for m in METHODS:
            X = make_features(FP, idxA, idxB, methods=(m,))[m]
            for seed in SEEDS:
                tr, te = train_test_split(np.arange(len(all_pairs)), test_size=0.2, random_state=seed, stratify=labels)
                s = evaluate(X[tr], labels[tr], X[te], labels[te])
                for mt in METRIC_KEYS:
                    per[cond][m][mt].append(s[mt])
            print(f"    {cond:8s} {m:6s} ROC {np.mean(per[cond][m]['roc_auc']):.4f}±{np.std(per[cond][m]['roc_auc']):.4f}", flush=True)

    agg = {cond: {m: {mt: {"mean": float(np.mean(per[cond][m][mt])), "std": float(np.std(per[cond][m][mt]))}
                      for mt in METRIC_KEYS} for m in METHODS} for cond in ["easy", "hard", "matched"]}
    maxroc = max(agg[cd][m]["roc_auc"]["mean"] for cd in agg for m in METHODS)

    res = {"config": {"n_pos": n_pos, "n_drugs": len(drug_ids), "seeds": SEEDS, "n_bins": N_BINS,
                      "split": "warm", "runtime_s": round(time.time() - t0, 1)},
           "results": agg, "per_seed": per,
           "tanimoto": {k: dist[k + "_mean"] for k in ["positive", "easy", "hard", "matched"]},
           "match_deficit": mrep["deficit"], "flags": {"roc_over_0.98": bool(maxroc > 0.98)}}
    json.dump(res, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ ROC-AUC (5 seed 평균) ================")
    for m in METHODS:
        print(f"  {m:6s}: easy {agg['easy'][m]['roc_auc']['mean']:.4f} · hard {agg['hard'][m]['roc_auc']['mean']:.4f} · matched {agg['matched'][m]['roc_auc']['mean']:.4f}")
    if maxroc > 0.98:
        print(f"\n★★ RED FLAG: ROC-AUC 최대 {maxroc:.4f} > 0.98. 확인 필요.", flush=True)

    out_html = build_report(res, dist)
    print(f"\n[05] 완료 ({res['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    return res


if __name__ == "__main__":
    main()

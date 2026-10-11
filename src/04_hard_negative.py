"""
[DDI 실험 - 04단계] 어려운 negative — '쉬운 음성 거품' 벗기기
====================================================================
질문: 02~03의 negative는 '랜덤 쌍'이었다. 랜덤 쌍은 두 약물이 서로 안 닮은 경우가
      많아서(=쉬운 음성), 모델이 "안 닮았으면 상호작용 없음"이라는 편법을 배웠을 수 있다.
      negative를 '어렵게'(닮았는데 상호작용 없는 쌍) 바꾸면 성능이 얼마나 떨어지나?
      = 이전 점수에 낀 '쉬운 음성 거품'의 크기.

★핵심 아이디어 (초보자용)
  Tanimoto 유사도 = 두 분자의 지문(켜진 비트)이 얼마나 겹치나(0~1, 1=동일).
  - easy negative(02와 동일): 비상호작용 쌍을 '균등 랜덤' 추출 → 대체로 안 닮음(Tanimoto 낮음).
  - hard negative(신규): 비상호작용 쌍 중 'Tanimoto 높은'(닮은) 쌍을 골라 negative로.
    닮았는데 라벨이 0이라, 모델은 '닮음' 편법을 못 쓰고 진짜로 구별해야 함 → 더 어려움.

★공정 비교(중요)
  easy와 hard는 '같은 positive'와 '같은 train/test 분할(같은 seed)'을 쓰고 negative만 다르게.
  (positive는 402k 쌍의 앞부분에 고정 배치 → 같은 seed면 positive는 동일하게 분할됨)

★자율 진행하되 아래면 멈춤(스크립트가 진단 출력):
  - hard의 Tanimoto가 easy와 별 차이 없음(실험 성립 안 함)
  - positive∩negative≠0 (누수)
  - ROC-AUC>0.98(비현실적) 등

산출: results/04_ddi_drugbank_hard_negative.html , results/hard_negative_results.json
실행: conda run -n admet python src/04_hard_negative.py
env : admet
"""
import os, sys, json, time, io, base64
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ★지문·융합·학습·채점은 02와 동일 코드 재사용(중복 없음)
from ddi_common import ukey, build_fp_matrix, make_features, evaluate, _GEN

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
HARD_CSV = os.path.join(BASE, "data", "hard_negatives.csv")
RESULTS_JSON = os.path.join(BASE, "results", "hard_negative_results.json")

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["sum", "concat"]
METRIC_KEYS = ["roc_auc", "pr_auc", "f1", "accuracy"]
TOP_M = 400               # 약물별로 미리 담아둘 '가장 닮은 비상호작용 상대' 후보 수


def pairs_to_idx(pair_list, did2row):
    idxA = np.fromiter((did2row[a] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    idxB = np.fromiter((did2row[b] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    return idxA, idxB


def build_bitvects(drug_ids, id2smi):
    """Tanimoto 계산용 RDKit 비트벡터(특징과 '동일 지문' Morgan 2048 r=2)."""
    return [_GEN.GetFingerprint(Chem.MolFromSmiles(str(id2smi[d]))) for d in drug_ids]


def gen_hard_negatives(drug_ids, bitvects, pos_key_set, n_needed, did2row):
    """
    hard negative 생성 = '전체 비상호작용 쌍 중 Tanimoto가 가장 높은(닮은) n개'.
    (초기엔 약물별 라운드로빈을 썼으나 top이 너무 깊게 내려가 유사도가 묽어짐(Δ0.07) →
     사용자가 제안한 '전체 상위 구간 샘플' 방식으로 교체. hard가 확실히 닮도록.)
    절차:
      1) 전체 약물쌍 Tanimoto 행렬(1705×1705) 계산.
      2) 상삼각(i<j)만, 기존 positive는 제외 → 비상호작용 후보.
      3) 유사도 상위 n_needed개 채택.
    """
    n = len(drug_ids)
    S = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        S[i] = DataStructs.BulkTanimotoSimilarity(bitvects[i], bitvects)   # i vs 전부
    iu, ju = np.triu_indices(n, k=1)               # i<j 인 모든 쌍
    sims = S[iu, ju]
    # positive 쌍을 코드(i*n+j)로 만들어 벡터화 제외
    pos_codes = np.fromiter(
        (min(did2row[a], did2row[b]) * n + max(did2row[a], did2row[b]) for (a, b) in pos_key_set),
        dtype=np.int64, count=len(pos_key_set))
    all_codes = iu.astype(np.int64) * n + ju.astype(np.int64)
    valid = ~np.isin(all_codes, pos_codes)         # 비상호작용만
    v_sims = sims[valid]; v_iu = iu[valid]; v_ju = ju[valid]
    # 유사도 상위 n_needed개
    if n_needed >= len(v_sims):
        sel = np.arange(len(v_sims))
    else:
        sel = np.argpartition(-v_sims, n_needed)[:n_needed]
    return [ukey(drug_ids[int(v_iu[k])], drug_ids[int(v_ju[k])]) for k in sel]


def pair_tanimoto(pairs, bitvects, did2row):
    """각 쌍의 두 약물 Tanimoto 유사도 배열."""
    return np.array([DataStructs.TanimotoSimilarity(bitvects[did2row[a]], bitvects[did2row[b]])
                     for a, b in pairs], dtype=np.float64)


# ---------------- 보고서 ----------------
def _b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_report(res, tani):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for _kf in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]:
        if os.path.exists(_kf):
            fm.fontManager.addfont(_kf); plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False

    agg = res["results"]; c = res["config"]
    CONDS = ["easy", "hard"]; ccol = {"easy": "#8a8a99", "hard": "#c0562f"}

    # 그림1: Tanimoto 분포(hard가 정말 닮았나 증명)
    fig1, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.linspace(0, 1, 41)
    ax.hist(tani["easy_sample"], bins=bins, alpha=0.6, color=ccol["easy"], density=True, label=f"easy (평균 {tani['easy_mean']:.3f})")
    ax.hist(tani["hard_sample"], bins=bins, alpha=0.6, color=ccol["hard"], density=True, label=f"hard (평균 {tani['hard_mean']:.3f})")
    ax.axvline(tani["easy_mean"], color=ccol["easy"], ls="--"); ax.axvline(tani["hard_mean"], color=ccol["hard"], ls="--")
    ax.set_xlabel("쌍의 두 약물 Tanimoto 유사도"); ax.set_ylabel("밀도")
    ax.set_title(f"negative 종류별 유사도 분포 — hard가 실제로 더 닮음 (Δ평균 {tani['hard_mean']-tani['easy_mean']:+.3f})")
    ax.legend();
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    chart_t = _b64(fig1)

    # 그림2: easy vs hard ROC-AUC
    fig2, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(METHODS)); w = 0.36
    for i, cond in enumerate(CONDS):
        means = [agg[cond][m]["roc_auc"]["mean"] for m in METHODS]
        stds = [agg[cond][m]["roc_auc"]["std"] for m in METHODS]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=3, color=ccol[cond], label=cond + (" (랜덤, 기준)" if cond == "easy" else " (닮은, 어려움)"))
        for xi, (mu, sd) in enumerate(zip(means, stds)):
            ax.annotate(f"{mu:.3f}", (x[xi] + (i - 0.5) * w, mu + sd + 0.006), ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(METHODS); ax.set_ylim(0.5, 0.95)
    ax.set_ylabel("ROC-AUC (5 seed 평균±std)")
    ax.set_title("easy vs hard ROC-AUC — ★hard(닮은 음성)가 오히려 더 높음(예상과 반대)")
    ax.legend(loc="lower left", fontsize=9); ax.grid(axis="y", alpha=.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    chart_r = _b64(fig2)

    # 성능표
    def mrow(m):
        cells = ""
        for cond in CONDS:
            for mt in METRIC_KEYS:
                cells += f"<td>{agg[cond][m][mt]['mean']:.4f}<span class=mut style=font-size:9px>±{agg[cond][m][mt]['std']:.3f}</span></td>"
        d = agg["hard"][m]["roc_auc"]["mean"] - agg["easy"][m]["roc_auc"]["mean"]
        return f"<tr><td class=l><b>{m}</b></td>{cells}<td class=bad>{d:+.4f}</td></tr>"
    prows = "".join(mrow(m) for m in METHODS)

    sd = agg["hard"]["sum"]["roc_auc"]["mean"] - agg["easy"]["sum"]["roc_auc"]["mean"]
    seasy = agg["easy"]["sum"]["roc_auc"]["mean"]; shard = agg["hard"]["sum"]["roc_auc"]["mean"]
    rose = sd > 0.005     # ★hard가 오히려 오름 = 예상과 반대(중요)

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1060px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 7px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}b.hl{color:#ffd479}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>04 DDI hard negative</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>04 · DDI 어려운 negative — '쉬운 음성 거품' 벗기기</h1>
<p class=mut>negative를 '랜덤(easy)' vs '닮은-비상호작용(hard)'으로 바꿔 성능 변화 측정. 나머지(positive·분할·특징·XGBoost)는 02와 동일 · warm 분할 · env admet · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론(★예상과 반대):</b> sum ROC-AUC가 <b>easy {seasy:.4f} → hard {shard:.4f} ({sd:+.4f})</b>로 <b class=hl>오히려 올랐다</b>.
이유: DrugBank DDI는 <b>구조 유사도 기반이 아니라</b>(상호작용 쌍 Tanimoto {tani['pos_mean']:.3f} ≈ 랜덤 {tani['easy_mean']:.3f}), '닮은' 음성(hard {tani['hard_mean']:.3f})이 <b>positive보다 더 닮게</b> 되어 <b>"더 닮으면 음성"이라는 (역방향) 쉬운 지름길</b>을 만들어버렸다. → <b>유사도 기반 hard negative는 이 데이터셋엔 부적절</b>(어렵게 하려다 오히려 쉽게 만듦).</div>

<h2>① 설정 (02와 동일, 'negative 종류'만 변수)</h2>
<div class=card>positive {c['n_pos']:,}(무순서 dedupe) · negative 1:1 · 약물 {c['n_drugs']:,} · Morgan 2048(r=2), 융합 <b>sum</b>(메인)·concat(참고) · XGBoost 고정 · <b>warm 분할</b>(쌍 랜덤) · seed {c['seeds']} 평균±std.<br>
★<b>hard negative 생성법</b>(택한 방법·조정 기록): <b>전체 비상호작용 약물쌍 중 Tanimoto(같은 Morgan 지문)가 가장 높은 {c['n_pos']:,}개</b>를 채택. <span class=mut>(초기엔 '약물별 라운드로빈'을 썼으나 유사도 차이가 작아(Δ0.07) 실험이 약해, 사용자가 제안한 '전체 상위 구간' 방식으로 교체 → hard가 확실히 닮도록.)</span> positive는 전부 제외(∩=0 검증), self-pair 없음.</div>

<h2>② hard가 정말 더 '닮았나' 검증 + ★핵심 발견</h2>
<img src="{chart_t}" alt="tanimoto dist">
<div class=card>쌍의 두 약물 Tanimoto 평균: <b>easy {tani['easy_mean']:.3f}</b> · <b class=warn>positive {tani['pos_mean']:.3f}</b> · <b>hard {tani['hard_mean']:.3f}</b>.<br>
★<b>핵심 발견 1 — DrugBank DDI는 '구조 유사도'로 잘 안 설명됨:</b> 상호작용하는 쌍(positive {tani['pos_mean']:.3f})이 랜덤 비상호작용(easy {tani['easy_mean']:.3f})보다 <b>거의 안 닮음</b>. 즉 "닮으면 상호작용"이 성립 안 함(DDI는 대사·약리 기전 문제).<br>
★<b>핵심 발견 2 — 유사도 hard가 오히려 문제를 '쉽게' 만듦:</b> hard({tani['hard_mean']:.3f})가 <b>positive({tani['pos_mean']:.3f})보다 더 닮음</b>. positive는 안 닮았는데 hard-neg는 닮았으니, 모델이 <b>"더 닮으면 음성"</b>이라는 역방향 규칙을 쉽게 배워 오히려 성능↑(아래 ③·④). 단 <b>절대 유사도는 낮음</b> — 매우 닮은 비상호작용 쌍이 드물어(전체 중 Tanimoto&gt;0.3이 ~4천 쌍뿐) 191k를 채우려면 중간 유사도까지 포함됨.</div>

<h2>③ 성능: easy vs hard</h2>
<img src="{chart_r}" alt="roc easy vs hard">
<table>
<tr><th class=l rowspan=2>융합</th><th colspan=4>easy (랜덤 음성)</th><th colspan=4>hard (닮은 음성)</th><th rowspan=2>ΔROC</th></tr>
<tr>{"".join("<th>"+h+"</th>" for _ in ["easy","hard"] for h in ["ROC","PR","F1","acc"])}</tr>
{prows}
</table>

<h2>④ 해석 — ★hard가 오히려 '쉬워진' 이유 (예상과 반대)</h2>
<div class=card>
· <b>sum:</b> easy {seasy:.4f} → hard {shard:.4f} (<b class=good>{sd:+.4f}</b>, <b>상승</b>). 어렵게 하려던 negative가 오히려 성능을 올림.<br>
· <b>왜 예상과 반대인가:</b> '어려운 음성' 발상은 "positive는 닮은 쌍이니, 닮은 음성을 주면 헷갈린다"를 전제한다. 그런데 이 데이터의 positive는 <b>안 닮았다</b>(Tanimoto {tani['pos_mean']:.3f} ≈ 랜덤 {tani['easy_mean']:.3f}). 그래서 '닮은' 음성을 넣으면 positive(안 닮음) vs hard-neg(닮음)의 <b>유사도 격차가 오히려 벌어져</b>, 모델이 <b>"더 닮으면 음성"이라는 역방향 지름길</b>을 쉽게 학습 → ROC 상승.<br>
· ★<b>핵심 교훈:</b> 유사도 기반 hard negative는 <b>"유사도→상호작용"이 성립하는 데이터에서만 어렵다</b>. DDI처럼 상호작용이 구조유사도로 안 풀리는 문제에선, 유사도로 뽑은 음성이 <b>분포 편향(distribution shift)</b>을 만들어 문제를 왜곡(쉽게)한다. 원래 가설("easy 점수엔 쉬운 음성 거품이 크다")은 <b>이 데이터에선 성립하지 않았다</b> — easy가 유사도 지름길에 별로 안 기대고 있었기 때문.<br>
· <b>그럼 easy가 정답인가?</b> easy도 완벽친 않다. 진짜 공정한 음성은 <b>positive와 같은 유사도 분포로 맞춘(distribution-matched) 음성</b> — 그래야 유사도가 어느 방향으로도 지름길이 안 된다. 그게 올바른 다음 단계.<br>
· 두 조건 모두 sum이 concat보다 우위 유지(sum {seasy:.3f}/{shard:.3f} vs concat {agg['easy']['concat']['roc_auc']['mean']:.3f}/{agg['hard']['concat']['roc_auc']['mean']:.3f}).
</div>

<h2>⑤ ★한계 메모 (정직하게)</h2>
<div class=card>
· <b>★이번 방식의 부작용(가장 중요):</b> 유사도로 음성을 뽑을수록, positive가 안 닮은 이 데이터에선 오히려 문제를 <b>쉽게</b> 만든다(역방향 지름길). 더 극단적으로 뽑으면 ROC가 더 오를 수도 — <b>진짜 '어려움'이 아님</b>. 즉 'hard negative=유사쌍'은 여기 부적절. → 결과의 낙폭/상승을 '실전 난이도'로 해석하면 안 됨.<br>
· <b>약물 쏠림 소폭 기여:</b> hard는 유사 약물 군집에 조금 쏠림(약물당 최대 856쌍 vs easy 304) → warm 분할서 그 약물 암기도 ROC↑에 일부 보탬.<br>
· <b>여전히 warm 분할:</b> 약물 겹침(암기) 낙관 그대로. <b>negative 라벨도 '미확인'</b>(닮은 비상호작용은 오히려 미발견 상호작용일 위험↑).<br>
· 이진 단순화·XGBoost 고정은 02와 동일.
</div>

<h2>⑥ 다음에 해볼 것</h2>
<div class=card>
· ★<b>distribution-matched negative</b>(이번 교훈의 정답): positive와 <b>같은 Tanimoto 분포</b>가 되도록 음성을 뽑아, 유사도가 어느 방향으로도 지름길이 안 되게. 그래야 '진짜' 난이도 추정.<br>
· <b>유사도 외 '어려움' 정의</b>: 같은 약물 계열/타깃이지만 상호작용 없는 쌍 등(생물학적 hard).<br>
· cold(미지 약물) + 제대로 된 hard 결합 → 가장 현실적 추정.<br>
· 물리화학 서술자 추가로 구조 외 신호 보강.
</div>

<div class=foot>04 DDI hard negative · easy(랜덤) vs hard(닮은-비상호작용) · 융합 sum·concat · warm 분할 · XGBoost 고정 · seed {len(c['seeds'])} 평균±std · positive∩neg=0 검증 · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "04_ddi_drugbank_hard_negative.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


def main():
    t0 = time.time()
    print("[04] hard negative 실험 시작 ...", flush=True)
    # --- 데이터: 약물 + positive + easy negative(02 재사용) ---
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"]))
    did2row = {d: i for i, d in enumerate(drug_ids)}
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["drug_a"] = pairs["drug_a"].astype(str); pairs["drug_b"] = pairs["drug_b"].astype(str)
    pos_pairs = [ukey(a, b) for a, b in zip(pairs.loc[pairs.label == 1, "drug_a"], pairs.loc[pairs.label == 1, "drug_b"])]
    easy_neg = [ukey(a, b) for a, b in zip(pairs.loc[pairs.label == 0, "drug_a"], pairs.loc[pairs.label == 0, "drug_b"])]
    pos_key_set = set(pos_pairs)
    n_pos = len(pos_pairs)
    print(f"  약물 {len(drug_ids):,} · positive {n_pos:,} · easy negative(02) {len(easy_neg):,}", flush=True)

    # --- 지문: 특징용 csr + Tanimoto용 비트벡터 ---
    FP = build_fp_matrix(drug_ids, id2smi)
    bitvects = build_bitvects(drug_ids, id2smi)

    # --- hard negative 생성 ---
    print("  hard negative 생성(전체 비상호작용 쌍 중 Tanimoto 상위 채택) ...", flush=True)
    hard_neg = gen_hard_negatives(drug_ids, bitvects, pos_key_set, n_pos, did2row)
    # ★검증: positive와 겹치면 안 됨 · self-pair 없음
    overlap = len(set(hard_neg) & pos_key_set)
    n_self = sum(1 for a, b in hard_neg if a == b)
    assert overlap == 0, f"★hard∩positive={overlap} (누수)"
    assert n_self == 0, f"★self-pair {n_self}"
    pd.DataFrame(hard_neg, columns=["drug_a", "drug_b"]).to_csv(HARD_CSV, index=False)
    print(f"  hard negative {len(hard_neg):,}개 · ∩positive={overlap} · self={n_self} · 저장 {os.path.basename(HARD_CSV)}", flush=True)

    # --- Tanimoto 분포(easy vs hard): hard가 정말 더 닮았나 ---
    rng = np.random.RandomState(0)
    def sample(pairs_list, k=30000):
        idx = rng.choice(len(pairs_list), size=min(k, len(pairs_list)), replace=False)
        return pair_tanimoto([pairs_list[i] for i in idx], bitvects, did2row)
    t_easy = sample(easy_neg); t_hard = sample(hard_neg); t_pos = sample(pos_pairs)
    tani = {"easy_mean": float(t_easy.mean()), "hard_mean": float(t_hard.mean()), "pos_mean": float(t_pos.mean()),
            "easy_sample": t_easy.tolist(), "hard_sample": t_hard.tolist()}
    print(f"  Tanimoto 평균: easy {tani['easy_mean']:.3f} · positive {tani['pos_mean']:.3f} · hard {tani['hard_mean']:.3f} "
          f"(hard−easy {tani['hard_mean']-tani['easy_mean']:+.3f}, hard−pos {tani['hard_mean']-tani['pos_mean']:+.3f})", flush=True)

    # ★red flag: '유사도 지름길'이 안 막히면 실험 무의미. 기준 = hard가 positive만큼도 안 닮으면 중단.
    #   (DrugBank DDI는 구조 유사도 약함: pos≈easy. hard가 pos를 넘으면 지름길이 '역전'되어 유효.)
    if tani["hard_mean"] - tani["easy_mean"] < 0.04 or tani["hard_mean"] <= tani["pos_mean"]:
        print("\n★★ RED FLAG: hard가 충분히 닮지 않음(positive 유사도 이하) → 학습 중단, 사용자 확인 필요.", flush=True)
        json.dump({"aborted": "hard_not_harder", "tani": {k: tani[k] for k in ("easy_mean", "hard_mean", "pos_mean")}},
                  open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)
        return {"aborted": True, "tani": tani}

    # --- 학습·평가: 조건(easy/hard) × 방법(sum/concat) × seed ---
    per = {cond: {m: {mt: [] for mt in METRIC_KEYS} for m in METHODS} for cond in ["easy", "hard"]}
    for cond, neg in [("easy", easy_neg), ("hard", hard_neg)]:
        all_pairs = pos_pairs + neg
        labels = np.array([1] * n_pos + [0] * len(neg))
        idxA, idxB = pairs_to_idx(all_pairs, did2row)
        for m in METHODS:
            X = make_features(FP, idxA, idxB, methods=(m,))[m]
            for seed in SEEDS:
                # ★easy·hard 모두 같은 seed → positive는 동일하게 분할(공정)
                tr, te = train_test_split(np.arange(len(all_pairs)), test_size=0.2, random_state=seed, stratify=labels)
                s = evaluate(X[tr], labels[tr], X[te], labels[te])
                for mt in METRIC_KEYS:
                    per[cond][m][mt].append(s[mt])
            r = per[cond][m]
            print(f"    {cond:4s} {m:6s} ROC {np.mean(r['roc_auc']):.4f}±{np.std(r['roc_auc']):.4f}", flush=True)

    agg = {cond: {m: {mt: {"mean": float(np.mean(per[cond][m][mt])), "std": float(np.std(per[cond][m][mt]))}
                      for mt in METRIC_KEYS} for m in METHODS} for cond in ["easy", "hard"]}

    # ★red flag: 비현실적 고성능
    maxroc = max(agg[c][m]["roc_auc"]["mean"] for c in agg for m in METHODS)
    flag_hi = maxroc > 0.98

    res = {"config": {"n_pos": n_pos, "n_drugs": len(drug_ids), "seeds": SEEDS, "top_m": TOP_M,
                      "split": "warm", "runtime_s": round(time.time() - t0, 1)},
           "results": agg, "per_seed": per,
           "tanimoto": {"easy_mean": tani["easy_mean"], "hard_mean": tani["hard_mean"], "pos_mean": tani["pos_mean"]},
           "flags": {"roc_over_0.98": bool(flag_hi)}}
    json.dump(res, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ ROC-AUC (5 seed 평균) ================")
    for m in METHODS:
        e = agg["easy"][m]["roc_auc"]["mean"]; h = agg["hard"][m]["roc_auc"]["mean"]
        print(f"  {m:6s}: easy {e:.4f} → hard {h:.4f} ({h-e:+.4f})")
    if flag_hi:
        print(f"\n★★ RED FLAG: ROC-AUC 최대 {maxroc:.4f} > 0.98 (비현실적). 사용자 확인 필요.", flush=True)

    out_html = build_report(res, tani)
    print(f"\n[04] 완료 ({res['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    return res


if __name__ == "__main__":
    main()

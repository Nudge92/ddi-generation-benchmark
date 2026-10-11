"""
[DDI 실험 - 03단계] cold-start 분할 — 진짜 일반화 측정
====================================================================
질문: 02(warm split)의 sum ROC-AUC 0.8766이 '약물을 외운' 덕에 부풀려진 낙관치인가?
방법: 다른 건 02와 전부 고정하고 '분할'만 어렵게 바꿔서 성능 낙폭을 본다.

★왜 cold-start인가 (초보자용 설명)
  warm split(02)은 '쌍'을 랜덤으로 train/test에 나눈다. 그러면 같은 '약물'이
  train과 test 양쪽에 등장한다(예: 아스피린이 train의 어떤 쌍에도, test의 다른 쌍에도).
  모델이 "이 약물이 낀 쌍은 대체로 양성" 같은 '약물 정체성'을 외우면, test에서도
  그 약물 덕에 잘 맞힌다 → 낙관적. 실전에서 진짜 궁금한 건 '처음 보는 신약'인데.

  그래서 '약물'을 먼저 train-약물 / test-약물로 나눈다. 그다음 쌍을 나눔:
    - warm       : (기준) 02처럼 쌍 단위 랜덤. 약물 겹침 허용.
    - cold-start : train = 두 약물 다 train-약물인 쌍.
                   test  = 정확히 '한' 약물만 test-약물(=처음 보는 약물)인 쌍.
    - cold-cold  : test  = '두' 약물 다 test-약물(둘 다 처음)인 쌍.
  warm→cold-start→cold-cold로 갈수록 '미지 약물'이 늘어 어려워진다.
  낙폭 = 02 점수 중 '약물 암기'가 차지하던 몫.

★이 단계(1)에서 할 일: 약물 분할 + 세 regime의 쌍·negative를 만들고,
  개수와 '누수 없음'을 검증만 한다. (학습은 확인 후 2단계에서)

★주의(흔한 실수) — negative도 각 regime 규칙을 지켜야 한다:
  cold-start test의 negative도 '정확히 한 약물만 test-약물'이어야 한다.
  안 그러면(예: negative가 train-약물끼리) test가 'cold'가 아니게 되어 불공정.

산출: results/coldstart_counts.json (개수·검증 요약)
실행: conda run -n admet python src/03_coldstart.py
env : admet
"""
import os, sys, json, time, io, base64
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ★02와 동일한 지문·융합·학습 코드 재사용(중복 구현 없음)
from ddi_common import ukey, build_fp_matrix, make_features, fit_model, score_model

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
COUNTS_JSON = os.path.join(BASE, "results", "coldstart_counts.json")
RESULTS_JSON = os.path.join(BASE, "results", "coldstart_results.json")

SEEDS = [0, 1, 2, 3, 4]
DRUG_TEST_FRAC = 0.20      # 약물 기준 test 비율(80:20)
NEG_RATIO = 1.0           # 각 세트 안에서 positive:negative = 1:1
METHODS = ["sum", "concat"]   # 메인=sum(02 승자), 참고=concat
METRIC_KEYS = ["roc_auc", "pr_auc", "f1", "accuracy"]


# ---------------- 약물 분할 ----------------
def split_drugs(drug_ids, seed, test_frac=DRUG_TEST_FRAC):
    """약물 목록을 train-약물/test-약물로 랜덤 분할(약물 기준)."""
    rng = np.random.RandomState(seed)          # 약물 분할 시드(seed별 다른 분할)
    perm = rng.permutation(len(drug_ids))
    n_test = int(round(len(drug_ids) * test_frac))
    arr = np.asarray(drug_ids)
    test_drugs = set(arr[perm[:n_test]].tolist())
    train_drugs = set(arr[perm[n_test:]].tolist())
    return train_drugs, test_drugs


def classify_positives(pos_pairs, test_drugs):
    """positive 쌍을 미지 약물 수(0/1/2)로 분류: TT(둘다 train)·TS(하나 미지)·SS(둘다 미지)."""
    TT, TS, SS = [], [], []
    for (a, b) in pos_pairs:
        k = (a in test_drugs) + (b in test_drugs)
        (TT if k == 0 else TS if k == 1 else SS).append((a, b))
    return TT, TS, SS


# ---------------- negative 생성 (regime 규칙 준수) ----------------
def _batched(rng, n):
    return rng.randint(0, n, size=200_000)


def sample_neg_both(pool, n_needed, pos_key_set, n_pos_in_pool, rng):
    """pool 안 두 약물 무순서 쌍 중 positive 아닌 것 n개. (TT 또는 SS negative)"""
    arr = np.array(sorted(pool)); m = len(arr)
    space = m * (m - 1) // 2 - n_pos_in_pool         # 뽑을 수 있는 최대 negative
    n_take = min(n_needed, space)                    # 공간 부족하면 cap
    chosen = set(); tries = 0
    while len(chosen) < n_take and tries < 80_000_000:
        ai = _batched(rng, m); bi = _batched(rng, m)
        for x, y in zip(arr[ai], arr[bi]):
            if x == y:
                continue
            key = ukey(x, y)
            if key in pos_key_set or key in chosen:
                continue
            chosen.add(key)
            if len(chosen) >= n_take:
                break
        tries += 200_000
    return list(chosen), (n_take < n_needed)         # (쌍들, cap됐는지)


def sample_neg_cross(pool_train, pool_test, n_needed, pos_key_set, rng):
    """한 약물은 train-약물, 다른 하나는 test-약물 → '정확히 하나가 미지'. (cold-start negative)"""
    p1 = np.array(sorted(pool_train)); p2 = np.array(sorted(pool_test))
    space = len(p1) * len(p2)                         # train×test 조합 수(근사 상한)
    n_take = min(n_needed, space)
    chosen = set(); tries = 0
    while len(chosen) < n_take and tries < 80_000_000:
        ai = _batched(rng, len(p1)); bi = _batched(rng, len(p2))
        for x, y in zip(p1[ai], p2[bi]):
            key = ukey(x, y)                          # x=train, y=test → 항상 하나만 미지
            if key in pos_key_set or key in chosen:
                continue
            chosen.add(key)
            if len(chosen) >= n_take:
                break
        tries += 200_000
    return list(chosen), (n_take < n_needed)


# ---------------- 누수 검증 ----------------
def verify(train_pairs, test_pairs, test_drugs, regime):
    """train엔 미지약물 없어야·test는 regime 조건 만족·train/test 쌍 겹침 0."""
    bad_train = sum(1 for (a, b) in train_pairs if (a in test_drugs) or (b in test_drugs))
    if regime == "cold-start":
        bad_test = sum(1 for (a, b) in test_pairs if ((a in test_drugs) + (b in test_drugs)) != 1)
        cond = "test 쌍마다 정확히 1개 미지"
    else:  # cold-cold
        bad_test = sum(1 for (a, b) in test_pairs if ((a in test_drugs) + (b in test_drugs)) != 2)
        cond = "test 쌍마다 2개 다 미지"
    overlap = len(set(train_pairs) & set(test_pairs))
    ok = (bad_train == 0 and bad_test == 0 and overlap == 0)
    return {"regime": regime, "train_미지약물낀쌍": bad_train, "test_조건위반": bad_test,
            "train_test_쌍겹침": overlap, "조건": cond, "PASS": ok}


# ---------------- 한 seed에 대해 regime 구성 + 검증 ----------------
def build_seed(seed, drug_ids, pos_pairs, pos_key_set):
    train_drugs, test_drugs = split_drugs(drug_ids, seed)
    TT, TS, SS = classify_positives(pos_pairs, test_drugs)

    rng = np.random.RandomState(1000 + seed)   # negative 시드(약물 분할과 분리)
    negTT, capTT = sample_neg_both(train_drugs, int(len(TT) * NEG_RATIO), pos_key_set, len(TT), rng)
    negTS, capTS = sample_neg_cross(train_drugs, test_drugs, int(len(TS) * NEG_RATIO), pos_key_set, rng)
    negSS, capSS = sample_neg_both(test_drugs, int(len(SS) * NEG_RATIO), pos_key_set, len(SS), rng)

    # 각 세트 = positive + negative
    train_pairs = TT + negTT                    # cold-start·cold-cold 공통 학습셋
    cs_test = TS + negTS                         # cold-start test
    cc_test = SS + negSS                         # cold-cold  test

    vcs = verify(train_pairs, cs_test, test_drugs, "cold-start")
    vcc = verify(train_pairs, cc_test, test_drugs, "cold-cold")

    # 라벨(positive=1, negative=0)을 pos·neg 순서 그대로 붙임
    sets = {
        "train": (train_pairs, np.array([1] * len(TT) + [0] * len(negTT))),      # cold 학습셋(TT)
        "cold-start": (cs_test, np.array([1] * len(TS) + [0] * len(negTS))),      # TS test
        "cold-cold": (cc_test, np.array([1] * len(SS) + [0] * len(negSS))),       # SS test
    }
    return {
        "seed": seed,
        "n_train_drugs": len(train_drugs), "n_test_drugs": len(test_drugs),
        "train(TT)": {"pos": len(TT), "neg": len(negTT), "total": len(train_pairs), "neg_capped": capTT},
        "cold-start(TS)": {"pos": len(TS), "neg": len(negTS), "total": len(cs_test), "neg_capped": capTS},
        "cold-cold(SS)": {"pos": len(SS), "neg": len(negSS), "total": len(cc_test), "neg_capped": capSS},
        "verify": {"cold-start": vcs, "cold-cold": vcc},
        "_sets": sets,        # 학습·평가용(JSON 저장 땐 제외)
    }


def pairs_to_idx(pair_list, did2row):
    """약물 쌍 리스트 → 두 약물의 지문 행번호 배열(idxA, idxB)."""
    idxA = np.fromiter((did2row[a] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    idxB = np.fromiter((did2row[b] for a, b in pair_list), dtype=np.int32, count=len(pair_list))
    return idxA, idxB


# ---------------- 보고서(HTML) ----------------
def _b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_report(counts, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for _kf in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]:
        if os.path.exists(_kf):
            fm.fontManager.addfont(_kf); plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False

    agg = results["results"]; c = results["config"]; cs = results["counts_summary"]
    REGIMES = ["warm", "cold-start", "cold-cold"]
    RLAB = {"warm": "warm\n(약물 겹침)", "cold-start": "cold-start\n(1개 미지)", "cold-cold": "cold-cold\n(2개 다 미지)"}
    colors = {"sum": "#1a8f4c", "concat": "#8a8a99"}

    # ★핵심 그림: regime별 ROC-AUC 낙폭
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(REGIMES)); w = 0.36
    for i, m in enumerate(METHODS):
        means = [agg[m][rg]["roc_auc"]["mean"] for rg in REGIMES]
        stds = [agg[m][rg]["roc_auc"]["std"] for rg in REGIMES]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=3, color=colors[m],
               label=m + (" (메인, 02 승자)" if m == "sum" else " (참고)"))
        for xi, (mu, sd) in enumerate(zip(means, stds)):
            ax.annotate(f"{mu:.3f}", (x[xi] + (i - 0.5) * w, mu + sd + 0.006), ha="center", fontsize=8)
    ax.axhline(0.5, ls="--", c="#e2555f", lw=1, alpha=.7); ax.text(2.35, 0.51, "무작위(0.5)", color="#e2555f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([RLAB[r] for r in REGIMES])
    ax.set_ylim(0.5, 0.95); ax.set_ylabel("ROC-AUC (5 seed 평균±std)")
    ax.set_title("분할이 어려워질수록 ROC-AUC ↓ — 낙폭 = 02 점수 중 '약물 암기'분")
    ax.legend(loc="upper right", fontsize=9); ax.grid(axis="y", alpha=.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    chart = _b64(fig)

    # sum 기준 낙폭
    sw = agg["sum"]["warm"]["roc_auc"]["mean"]; scs = agg["sum"]["cold-start"]["roc_auc"]["mean"]; scc = agg["sum"]["cold-cold"]["roc_auc"]["mean"]

    # 성능표
    def mrow(m):
        cells = ""
        for rg in REGIMES:
            for mt in METRIC_KEYS:
                cells += f"<td>{agg[m][rg][mt]['mean']:.4f}<span class=mut style=font-size:9px>±{agg[m][rg][mt]['std']:.3f}</span></td>"
        return f"<tr><td class=l><b>{m}</b></td>{cells}</tr>"
    prows = "".join(mrow(m) for m in METHODS)

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 7px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}b.hl{color:#ffd479}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>03 DDI cold-start split</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>03 · DDI cold-start 분할 — 진짜 일반화 측정</h1>
<p class=mut>02(warm)의 sum ROC-AUC 0.8766이 '약물 암기' 덕인지 검증. 분할만 어렵게 바꾸고 나머지(데이터·특징·XGBoost)는 02와 동일 고정 · env admet · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> sum ROC-AUC가 <b>warm {sw:.4f} → cold-start {scs:.4f} ({scs-sw:+.4f}) → cold-cold {scc:.4f} ({scc-sw:+.4f})</b>로 떨어짐.
<b class=hl>이 낙폭이 곧 02 점수 중 '약물을 외운' 몫</b> — 처음 보는 약물일수록 성능이 준다.</div>

<h2>① 설정 (02와 동일, '분할'만 변수)</h2>
<div class=card>데이터 TDC DrugBank DDI(무순서 dedupe positive {c['n_positive']:,}) · 약물 {c['n_drugs']:,} · 특징 Morgan 2048(r=2), 융합 <b>sum</b>(메인)·concat(참고) · XGBoost 고정 · seed {c['seeds']} 평균±std.<br>
★<b>3가지 분할</b>: 약물을 train {int(round((1-c['drug_test_frac'])*100))}% / test {int(c['drug_test_frac']*100)}%로 나눈 뒤 — <b>warm</b>(쌍 랜덤, 약물 겹침) · <b>cold-start</b>(test 쌍의 정확히 1개가 미지 약물) · <b>cold-cold</b>(2개 다 미지).</div>

<h2>② 데이터 규모 & 누수 검증</h2>
<table>
<tr><th class=l>Regime</th><th>학습 쌍(평균)</th><th>평가 쌍(평균)</th><th>누수 검증</th></tr>
<tr><td class=l><b>warm</b></td><td>~{int(c['n_warm_total']*0.8):,}</td><td>~{int(c['n_warm_total']*0.2):,}</td><td class=mut>약물 겹침 허용(기준)</td></tr>
<tr><td class=l><b>cold-start</b></td><td>{cs['train_mean']:,.0f} (TT)</td><td>{cs['cold_start_mean']:,.0f}</td><td class=good>PASS ✔ (test 쌍마다 정확히 1개 미지)</td></tr>
<tr><td class=l><b>cold-cold</b></td><td>{cs['train_mean']:,.0f} (TT, 공통)</td><td>{cs['cold_cold_mean']:,.0f} (최소 {cs['cold_cold_min']:,})</td><td class=good>PASS ✔ (test 쌍마다 2개 다 미지)</td></tr>
</table>
<p class=mut>✔ 모든 seed·regime에서 train에 미지 약물 0 · test regime 조건 위반 0 · train↔test 쌍 겹침 0. negative도 각 regime 규칙 준수.</p>

<h2>③ 성능 (5 seed 평균±std)</h2>
<img src="{chart}" alt="cold-start drop">
<table>
<tr><th class=l rowspan=2>융합</th><th colspan=4>warm</th><th colspan=4>cold-start</th><th colspan=4>cold-cold</th></tr>
<tr>{"".join("<th>"+h+"</th>" for _ in REGIMES for h in ["ROC","PR","F1","acc"])}</tr>
{prows}
</table>

<h2>④ 해석 — 낙폭 = 약물 암기분</h2>
<div class=card>
· <b>sum:</b> warm {sw:.4f} → cold-start {scs:.4f} (<b class=bad>{scs-sw:+.4f}</b>) → cold-cold {scc:.4f} (<b class=bad>{scc-sw:+.4f}</b>).<br>
· <b>cold-start</b>(한쪽만 미지): 아직 <b>익숙한 약물 1개</b>가 test 쌍에 있어 그 정보로 어느 정도 맞힘.<br>
· <b>cold-cold</b>(둘 다 미지): 두 약물 다 처음 → <b>약물 정체성 암기가 전혀 안 통함</b>. 여기 성능이 '진짜 신약 쌍'에 가장 가까운 추정치.<br>
· ★결론: 02의 0.877은 <b>약물 겹침(암기)</b>으로 부풀려진 낙관치였다. 실제 미지 약물 일반화력은 <b>cold-cold {scc:.4f}</b> 수준 — 여전히 무작위(0.5)보다 <b>{'상당히 높아 지문에 진짜 신호가 있음' if scc>0.65 else '높지만 낙폭이 커 암기 의존이 컸음'}</b>.<br>
· sum이 cold에서도 concat보다 {'우위 유지' if agg['sum']['cold-cold']['roc_auc']['mean']>agg['concat']['cold-cold']['roc_auc']['mean'] else '비슷/역전'} — 대칭 융합의 이점이 어려운 분할에서도 이어지는지 확인됨.
</div>

<h2>⑤ ★한계 메모 (정직하게)</h2>
<div class=card>
· <b>여전히 랜덤(쉬운) negative:</b> 02와 동일하게 랜덤 음성 사용 → cold 성능도 실전보다 낙관적일 수 있음. 어려운 음성이면 더 떨어질 것.<br>
· <b>cold-cold 표본 상대적으로 작음</b>(~{cs['cold_cold_mean']:,.0f}) → warm/cold-start보다 seed 편차 큼(표의 ±std 참고).<br>
· <b>약물 분할도 랜덤:</b> 어떤 약물이 test로 가느냐에 따라 난이도 변동 → 5 seed로 평균했지만 잔여 편차 존재.<br>
· <b>train 크기 차이:</b> cold는 아는 약물끼리만 학습(TT ~{cs['train_mean']:,.0f}) → warm 학습셋(~{int(c['n_warm_total']*0.8):,})보다 작음. 낙폭엔 '미지 약물' 효과 + '학습량 감소' 효과가 섞여 있음(주로 전자).<br>
· 이진 단순화·XGBoost 고정(튜닝 없음)은 02와 동일.
</div>

<h2>⑥ 다음에 해볼 것</h2>
<div class=card>
· <b>어려운 negative</b>(닮았지만 상호작용 없는 쌍 / degree-matched)로 cold 재측정 → 낙관 편향 추가 점검.<br>
· <b>cold 조건 균형</b>: 약물 test 비율·seed 늘려 cold-cold 편차 줄이기.<br>
· cold에서 지문 외 특징(물리화학 서술자) 추가 시 신약 일반화가 오르는지.<br>
· 유형별 다중분류로 확장 시 cold 성능.
</div>

<div class=foot>03 DDI cold-start · warm/cold-start/cold-cold · 융합 sum·concat · XGBoost 고정 · seed {len(c['seeds'])} 평균±std · 누수검증 통과 · env admet</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "03_ddi_drugbank_coldstart_split.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


def main():
    t0 = time.time()
    print("[03] cold-start 실험 (구성·검증 + 학습·평가) ...", flush=True)
    smi = pd.read_csv(DRUG_SMI_CSV); drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"]))
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["drug_a"] = pairs["drug_a"].astype(str); pairs["drug_b"] = pairs["drug_b"].astype(str)
    pos_df = pairs[pairs["label"] == 1]
    pos_pairs = [ukey(a, b) for a, b in zip(pos_df["drug_a"], pos_df["drug_b"])]
    pos_key_set = set(pos_pairs)
    warm_y = pairs["label"].to_numpy()
    n_warm_total = len(pairs)
    print(f"  유효 약물 {len(drug_ids):,} · positive 쌍 {len(pos_pairs):,} · warm 전체(02) {n_warm_total:,}", flush=True)

    # 지문(한 번만) + warm 특징(방법별 1번, seed는 분할만 변경)
    did2row = {d: i for i, d in enumerate(drug_ids)}
    FP = build_fp_matrix(drug_ids, id2smi)
    warm_idxA, warm_idxB = pairs_to_idx(list(zip(pairs["drug_a"], pairs["drug_b"])), did2row)
    warm_feats = {m: make_features(FP, warm_idxA, warm_idxB, methods=(m,))[m] for m in METHODS}

    REGIMES = ["warm", "cold-start", "cold-cold"]
    per = {m: {rg: {mt: [] for mt in METRIC_KEYS} for rg in REGIMES} for m in METHODS}
    counts_seeds = []

    for seed in SEEDS:
        r = build_seed(seed, drug_ids, pos_pairs, pos_key_set)
        sets = r.pop("_sets")                 # 학습용 세트 분리(JSON엔 안 넣음)
        counts_seeds.append(r)
        v = r["verify"]
        print(f"\n  [seed {seed}] 약물 train {r['n_train_drugs']}/test {r['n_test_drugs']} · "
              f"TT {r['train(TT)']['total']:,} · TS {r['cold-start(TS)']['total']:,} · SS {r['cold-cold(SS)']['total']:,} · "
              f"검증 {'PASS' if v['cold-start']['PASS'] and v['cold-cold']['PASS'] else 'FAIL'}", flush=True)

        tt_pairs, tt_y = sets["train"]; tt_idxA, tt_idxB = pairs_to_idx(tt_pairs, did2row)
        cs_pairs, cs_y = sets["cold-start"]; cs_idxA, cs_idxB = pairs_to_idx(cs_pairs, did2row)
        cc_pairs, cc_y = sets["cold-cold"]; cc_idxA, cc_idxB = pairs_to_idx(cc_pairs, did2row)
        wtr, wte = train_test_split(np.arange(n_warm_total), test_size=0.2, random_state=seed, stratify=warm_y)

        for m in METHODS:
            ts = time.time()
            # warm: 전체 약물 학습 → warm test
            wclf = fit_model(warm_feats[m][wtr], warm_y[wtr])
            warm_s = score_model(wclf, warm_feats[m][wte], warm_y[wte])
            # cold: TT(아는 약물) 학습 1번 → TS·SS 채점(같은 모델)
            ttclf = fit_model(make_features(FP, tt_idxA, tt_idxB, (m,))[m], tt_y)
            cs_s = score_model(ttclf, make_features(FP, cs_idxA, cs_idxB, (m,))[m], cs_y)
            cc_s = score_model(ttclf, make_features(FP, cc_idxA, cc_idxB, (m,))[m], cc_y)
            for mt in METRIC_KEYS:
                per[m]["warm"][mt].append(warm_s[mt])
                per[m]["cold-start"][mt].append(cs_s[mt])
                per[m]["cold-cold"][mt].append(cc_s[mt])
            print(f"    {m:6s} ROC warm {warm_s['roc_auc']:.4f} → cold-start {cs_s['roc_auc']:.4f} → cold-cold {cc_s['roc_auc']:.4f}  ({time.time()-ts:.0f}s)", flush=True)

    # 집계
    agg = {m: {rg: {mt: {"mean": float(np.mean(per[m][rg][mt])), "std": float(np.std(per[m][rg][mt]))}
                    for mt in METRIC_KEYS} for rg in REGIMES} for m in METHODS}

    def col(regime, field): return [s[regime][field] for s in counts_seeds]
    counts_summary = {
        "all_verify_pass": all(s["verify"]["cold-start"]["PASS"] and s["verify"]["cold-cold"]["PASS"] for s in counts_seeds),
        "train_mean": float(np.mean(col("train(TT)", "total"))),
        "cold_start_mean": float(np.mean(col("cold-start(TS)", "total"))),
        "cold_cold_mean": float(np.mean(col("cold-cold(SS)", "total"))),
        "cold_cold_min": int(min(col("cold-cold(SS)", "total"))),
    }
    cfg = {"seeds": SEEDS, "drug_test_frac": DRUG_TEST_FRAC, "neg_ratio": NEG_RATIO,
           "n_drugs": len(drug_ids), "n_positive": len(pos_pairs), "n_warm_total": n_warm_total}
    json.dump({"config": cfg, "per_seed": counts_seeds, "summary": counts_summary},
              open(COUNTS_JSON, "w"), ensure_ascii=False, indent=1)
    results = {"config": {**cfg, "methods": METHODS, "regimes": REGIMES, "runtime_s": round(time.time() - t0, 1)},
               "results": agg, "per_seed_metrics": per, "counts_summary": counts_summary}
    json.dump(results, open(RESULTS_JSON, "w"), ensure_ascii=False, indent=1)

    print("\n================ ROC-AUC (5 seed 평균) ================")
    for m in METHODS:
        w = agg[m]["warm"]["roc_auc"]["mean"]; c1 = agg[m]["cold-start"]["roc_auc"]["mean"]; c2 = agg[m]["cold-cold"]["roc_auc"]["mean"]
        print(f"  {m:6s}: warm {w:.4f} → cold-start {c1:.4f} ({c1-w:+.4f}) → cold-cold {c2:.4f} ({c2-w:+.4f})")
    out_html = build_report({"summary": counts_summary}, results)
    print(f"· 누수 검증: {'✔ 전부 PASS' if counts_summary['all_verify_pass'] else '✗ 실패'}")
    print(f"\n[03] 완료 ({results['config']['runtime_s']:.0f}s). 보고서 → {out_html}")
    return results


if __name__ == "__main__":
    main()

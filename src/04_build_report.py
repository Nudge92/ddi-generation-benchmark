"""
[DDI 실험 - 04단계] 결과 보고서 생성 (HTML)
====================================================================
results/fusion_results.json 을 읽어 보고서를 만든다.
  - 실험 목적·설정 (데이터 크기, negative 생성법, 융합법)
  - 세 방법(concat/sum/diff) 성능 비교 표 + 막대그래프(오차막대 포함)
  - 관찰·해석 (어느 융합이 이겼나, baseline 이겼나)
  - ★한계 메모 (negative 신뢰성·데이터 편향)
  - 다음에 해볼 것

산출: results/02_ddi_drugbank_binary_morgan_fusion.html
실행: conda run -n admet python src/04_build_report.py
"""
import os, io, json, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _kf in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]:
    if os.path.exists(_kf):
        fm.fontManager.addfont(_kf); plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_IN = os.path.join(BASE, "results", "fusion_results.json")
OUT_HTML = os.path.join(BASE, "results", "02_ddi_drugbank_binary_morgan_fusion.html")

METHODS = ["concat", "sum", "diff"]
MLABEL = {"concat": "(a) concat\n[fpA,fpB] 4096", "sum": "(b) sum\nfpA+fpB 2048", "diff": "(c) diff\n|fpA-fpB| 2048"}
METRICS = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1"), ("accuracy", "Accuracy")]


def b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chart(agg):
    """4지표 × 3방법 묶음 막대(오차막대). baseline=concat 회색."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(METRICS)); w = 0.26
    colors = {"concat": "#8a8a99", "sum": "#1a8f4c", "diff": "#c0562f"}
    for i, m in enumerate(METHODS):
        means = [agg[m][mt]["mean"] for mt, _ in METRICS]
        stds = [agg[m][mt]["std"] for mt, _ in METRICS]
        bars = ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3,
                      color=colors[m], label=m + (" (baseline)" if m == "concat" else ""))
        for xi, (mu, sd) in enumerate(zip(means, stds)):
            ax.annotate(f"{mu:.3f}", (x[xi] + (i - 1) * w, mu + sd + 0.006), ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in METRICS])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("점수 (5 seed 평균±표준편차)")
    ax.set_title("Morgan 지문 융합법 비교 — concat(baseline) vs sum vs diff (XGBoost 고정)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return b64(fig)


def build():
    r = json.load(open(JSON_IN))
    c = r["config"]; agg = r["results"]
    fig = chart(agg)

    base_roc = agg["concat"]["roc_auc"]["mean"]
    # 승자 판정: ROC-AUC 최고 방법 + concat 대비 표준편차 넘는 차이인가
    best = max(METHODS, key=lambda m: agg[m]["roc_auc"]["mean"])
    best_roc = agg[best]["roc_auc"]["mean"]
    concat_std = agg["concat"]["roc_auc"]["std"] or 0.001
    beats_base = best != "concat" and (best_roc - base_roc) > 2 * concat_std

    # 결과표
    def row(m):
        rr = agg[m]; d = rr["roc_auc"]["mean"] - base_roc; is0 = m == "concat"
        dcol = "good" if d > 0.005 else ("bad" if d < -0.005 else "mut")
        cells = "".join(f"<td>{rr[mt]['mean']:.4f}<span class=mut style=font-size:10px>±{rr[mt]['std']:.3f}</span></td>" for mt, _ in METRICS)
        dtxt = "baseline" if is0 else f"<span class={dcol}>{d:+.4f}</span>"
        star = " ★" if m == best and beats_base else ""
        return f"<tr{' style=background:rgba(120,120,140,.08)' if is0 else ''}><td class=l><b>{m}</b>{star} <span class=mut>{c['methods'][m]}</span></td>{cells}<td class=l>{dtxt}</td></tr>"
    rows = "".join(row(m) for m in METHODS)

    if beats_base:
        verdict = f"<b class=good>{best}</b>가 baseline(concat)을 ROC-AUC {best_roc-base_roc:+.4f}로 <b>표준편차 넘어 이김</b>."
    elif best == "concat":
        verdict = f"<b>baseline(concat)</b>이 그대로 최고 — sum/diff는 concat을 못 이김(정보 손실)."
    else:
        verdict = f"{best}가 근소 우위이나 <b>표준편차 안(무승부급)</b> — baseline 대비 명확한 승리 아님."

    cp = c["xgb_params"]
    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1040px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:6px 9px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}b.hl{color:#ffd479}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>02 DDI DrugBank binary — Morgan fusion</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>02 · DrugBank DDI 이진 분류 — Morgan 지문 융합법 비교 (concat/sum/diff)</h1>
<p class=mut>첫 DDI 실험. '약물 쌍의 지문을 어떻게 합치나'만 바꿔 XGBoost로 공정 비교 · TDC DrugBank DDI · env admet · 실행 {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> {verdict}</div>

<h2>① 실험 설정</h2>
<table>
<tr><th class=l>항목</th><th class=l>내용</th></tr>
<tr><td class=l>데이터</td><td class=l>TDC DrugBank DDI · 학습 쌍 <b>{c['n_pairs']:,}</b> · 고유 약물 <b>{c['n_drugs']:,}</b> · 양성비율 {c['pos_frac']:.3f}(1:1)</td></tr>
<tr><td class=l>negative 생성</td><td class=l>무순서 유니크 positive를 <b>모두 제외</b>한 공간에서 랜덤 1:1 추출(시드 42). ★진짜 양성 제외 검증(∩=0)</td></tr>
<tr><td class=l>누수 차단</td><td class=l>(A,B)↔(B,A) <b>무순서 dedupe</b> → 같은 쌍이 train/test로 갈리지 않음</td></tr>
<tr><td class=l>특징(융합)</td><td class=l>Morgan {c['fp_bits']}bit(r=2). (a)concat 4096 · (b)sum 2048 · (c)diff|A-B| 2048</td></tr>
<tr><td class=l>분류기</td><td class=l>XGBoost 고정: trees {cp['n_estimators']}·depth {cp['max_depth']}·lr {cp['learning_rate']}·hist. 세 방법 동일</td></tr>
<tr><td class=l>평가</td><td class=l>seed {c['seeds']} × 80/20 stratified split(동일 분할 공유) → 평균±표준편차. 지표 ROC-AUC/PR-AUC/F1/acc</td></tr>
</table>

<h2>② 세 방법 성능 비교</h2>
<img src="{fig}" alt="fusion comparison">
<table>
<tr><th class=l>융합법</th>{"".join(f"<th>{lab}</th>" for _,lab in METRICS)}<th class=l>ΔROC vs baseline</th></tr>
{rows}
</table>
<p class=mut>±는 5 seed 표준편차. ★=baseline을 표준편차 넘어 이긴 방법. F1·accuracy는 임계 0.5 기준.</p>

<h2>③ 관찰·해석</h2>
<div class=card>
· <b>승자:</b> {verdict}<br>
· <b>concat(baseline):</b> ROC-AUC {agg['concat']['roc_auc']['mean']:.4f} — 두 지문을 '따로' 보존(A·B 구분 유지). 차원 2배(4096)라 정보 손실 없음.<br>
· <b>sum:</b> ROC-AUC {agg['sum']['roc_auc']['mean']:.4f} ({agg['sum']['roc_auc']['mean']-base_roc:+.4f}) — 대칭(순서 무시)·2048차원. A·B 구분 사라짐(어느 쪽이 무슨 비트인지 모름).<br>
· <b>diff|A-B|:</b> ROC-AUC {agg['diff']['roc_auc']['mean']:.4f} ({agg['diff']['roc_auc']['mean']-base_roc:+.4f}) — '두 약물이 다른 부분(대칭차)'만 남김. 공통 구조 정보 버림.<br>
· <b>해석 포인트:</b> concat은 정보를 가장 많이 보존(4096차원)하나 차원↑. sum/diff는 압축·대칭이라 (A,B) 순서 문제엔 강하지만 정보 손실. 어느 게 이겼는지가 'DDI에 A·B 구분이 중요한가'를 시사.
</div>

<h2>④ ★한계 메모 (정직하게)</h2>
<div class=card>
· <b>negative 신뢰성:</b> 우리 음성은 '진짜 상호작용 없음'이 아니라 <b>'아직 안 밝혀진 것'</b>일 수 있음(false negative 혼입). 성능이 과대/과소 될 여지.<br>
· <b>랜덤 negative 편향:</b> 랜덤 쌍은 대체로 서로 안 닮아 '쉬운 음성'이 되기 쉬움 → 성능이 <b>낙관적</b>일 수 있음(실전 난이도 과소평가).<br>
· <b>약물 겹침(허브):</b> 랜덤 쌍 분할이라 train·test에 같은 약물이 양쪽 등장 → 모델이 '약물'을 외웠을 수 있음. 진짜 일반화는 <b>cold-start(새 약물)</b>로 봐야 함.<br>
· <b>단순 baseline:</b> 지문+XGBoost 고정, 튜닝 없음. 절대 성능보다 <b>'융합법 상대 비교'</b>가 목적.<br>
· <b>이진 단순화:</b> 86개 상호작용 유형을 다 '있음=1'로 합침. 유형별 예측은 별개 과제.
</div>

<h2>⑤ 다음에 해볼 것</h2>
<div class=card>
· <b>cold-start 분할</b>: 학습에 없던 약물 쌍으로 평가 → 진짜 일반화 측정(이번 낙관 편향 점검).<br>
· <b>어려운 negative</b>: 랜덤 대신 '닮았지만 상호작용 없는' 쌍으로 난이도↑, 또는 degree-matched 음성.<br>
· <b>concat + prod</b> 등 융합 추가, 또는 concat에 |diff|·prod를 <b>이어붙인 하이브리드</b> 특징.<br>
· <b>유형별 다중분류</b>(86종) 또는 상위 몇 유형으로 확장.<br>
· 지문 외 특징(물리화학 서술자)·다른 분류기 비교.
</div>

<div class=foot>02 DDI DrugBank binary · Morgan fusion(concat/sum/diff) · XGBoost 고정 · seed {len(c['seeds'])} 평균±std · 값은 실제 결과 · env admet</div>
</div></body></html>"""
    with open(OUT_HTML, "w") as f:
        f.write(html)
    print("REPORT:", OUT_HTML)
    print(f"승자={best} · baseline이김={beats_base}")
    return OUT_HTML, best, beats_base


if __name__ == "__main__":
    build()

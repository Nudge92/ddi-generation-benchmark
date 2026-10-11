"""
[DDI 실험 - 01단계] EDA (탐색적 데이터 분석) + HTML 보고서
====================================================================
목적: 본격 실험(전처리·모델) 전에 데이터를 '눈으로' 이해한다.
      특히 초보자가 놓치기 쉬운 함정 = (A,B)와 (B,A) 방향성/중복 →
      train/test 누수(leakage) 위험을 미리 점검한다.

보는 것:
  1) 데이터 크기 (약물 쌍 수, 고유 약물 수)
  2) 라벨(상호작용 유형 Y) 분포 → 쏠림 확인 (막대그래프)
  3) 이진 관점: positive 몇 개? → negative 몇 개 만들지 근거
  4) 데이터 품질: SMILES 파싱 실패·결측·중복 쌍
  5) ★방향성/누수: (A,B)==(B,A)인가? 같은 쌍이 여러 번? self-pair?
  6) 약물별 상호작용 수(degree) 분포 → 허브 약물 구조
  7) 관찰 + 실험 설계 시사점

산출: results/01_ddi_drugbank_eda.html
실행: conda run -n admet python src/01_eda.py
env : admet
"""
import os, io, base64, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # 화면 없이 이미지 파일로만 그림
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")      # RDKit 경고 숨김(파싱 실패는 우리가 직접 셈)

# 한글 폰트(차트 깨짐 방지)
for _kf in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]:
    if os.path.exists(_kf):
        fm.fontManager.addfont(_kf)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE, "data", "drugbank_ddi_raw.csv")
OUT_HTML = os.path.join(BASE, "results", "01_ddi_drugbank_eda.html")


# ---------- 작은 도우미: matplotlib 그림 → base64 문자열(HTML에 그대로 박음) ----------
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    print("[01] EDA 시작 ...", flush=True)
    df = pd.read_csv(RAW_CSV)
    df["Drug1_ID"] = df["Drug1_ID"].astype(str)
    df["Drug2_ID"] = df["Drug2_ID"].astype(str)
    n_rows = len(df)

    # ===== (1) 크기 =====
    all_ids = pd.concat([df["Drug1_ID"], df["Drug2_ID"]])
    n_drugs = all_ids.nunique()

    # ===== (2) 라벨(Y) 분포 =====
    yvc = df["Y"].value_counts().sort_values(ascending=False)
    n_types = len(yvc)
    top_type, top_cnt = yvc.index[0], yvc.iloc[0]

    # ===== (5) ★방향성/중복 (누수 점검) — 이진 positive 수 결정에도 필요 =====
    # self-pair: 자기 자신과의 쌍(있으면 이상)
    n_self = int((df["Drug1_ID"] == df["Drug2_ID"]).sum())
    # 순서 있는(ordered) 유니크 쌍: (a,b)와 (b,a)를 다르게 셈
    ordered = set(zip(df["Drug1_ID"], df["Drug2_ID"]))
    n_ordered_unique = len(ordered)
    # (a,b)가 있을 때 (b,a)도 데이터에 있는 경우 수
    n_reverse_present = sum(1 for (a, b) in ordered if (b, a) in ordered)
    # 순서 없는(unordered) 유니크 쌍: {a,b}로 취급 → 이진 실험의 '진짜' positive 단위
    unordered_key = df.apply(lambda r: tuple(sorted((r["Drug1_ID"], r["Drug2_ID"]))), axis=1)
    n_unordered_unique = unordered_key.nunique()
    # 같은 무순서 쌍이 여러 행으로 = (양방향 중복) 또는 (같은 쌍 다른 상호작용유형)
    n_dup_unordered_rows = n_rows - n_unordered_unique

    # ===== (3) 이진 관점 & negative 공간 =====
    # 이진 실험 positive = 무순서 유니크 쌍 수(중복 제거). 1:1이면 같은 수의 negative 필요.
    n_pos = n_unordered_unique
    # 가능한 전체 무순서 쌍 = C(n_drugs, 2)
    n_all_possible = n_drugs * (n_drugs - 1) // 2
    n_neg_space = n_all_possible - n_pos          # positive를 제외한 negative 후보 공간
    pos_density = n_pos / n_all_possible           # 전체 쌍 중 positive 비율

    # ===== (4) 데이터 품질: 약물ID→SMILES 매핑 & 파싱 =====
    # 각 Drug_ID가 항상 같은 SMILES를 갖는지(일관성) + RDKit 파싱 성공 여부
    id2smi = {}
    conflict_ids = set()
    for col_id, col_smi in [("Drug1_ID", "Drug1"), ("Drug2_ID", "Drug2")]:
        for i, s in zip(df[col_id], df[col_smi]):
            if i in id2smi and id2smi[i] != s:
                conflict_ids.add(i)          # 같은 ID인데 SMILES 다름(데이터 품질 이슈)
            else:
                id2smi[i] = s
    # 고유 약물 SMILES 파싱(약물 1,706개만 파싱 → 빠름; 쌍마다 안 함)
    parse_fail = [i for i, s in id2smi.items() if Chem.MolFromSmiles(str(s)) is None]
    n_parse_fail = len(parse_fail)

    # ===== (6) 약물별 degree (몇 개 약물과 상호작용하나) =====
    deg = all_ids.value_counts()   # 각 약물이 등장한 횟수 ≈ 상호작용 파트너 수(중복 포함)

    # ---------------- 차트 A: Y 상호작용 유형 분포(쏠림) ----------------
    figA, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(range(len(yvc)), yvc.values, color="#4c78d8")
    ax.set_yscale("log")   # 쏠림이 커서 로그 스케일이 잘 보임
    ax.set_xlabel(f"상호작용 유형 (86종, 개수 내림차순)")
    ax.set_ylabel("쌍 개수 (log)")
    ax.set_title(f"라벨(Y) 분포 — 소수 유형에 쏠림 (최다: 유형 {top_type} = {top_cnt:,}쌍, {top_cnt/n_rows:.0%})")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    chartA = fig_to_b64(figA)

    # ---------------- 차트 B: 약물별 degree 분포(허브 구조) ----------------
    figB, ax = plt.subplots(figsize=(9, 3.2))
    ax.hist(deg.values, bins=60, color="#e45756")
    ax.set_xlabel("약물 1개가 참여한 상호작용 쌍 수 (degree)")
    ax.set_ylabel("약물 수")
    ax.set_title(f"약물별 상호작용 수 분포 — 소수 '허브' 약물이 많은 쌍에 등장 (최대 {int(deg.max()):,}, 중앙값 {int(deg.median())})")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    chartB = fig_to_b64(figB)

    # ---------------- HTML 조립 ----------------
    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1040px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:6px 9px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}
    .card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}
    img{max-width:100%;border-radius:6px;margin:6px 0}b.hl{color:#ffd479}"""

    dq_flag = lambda ok: "<b class=good>정상</b>" if ok else "<b class=bad>문제</b>"

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>01 DDI DrugBank EDA</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>01 · DrugBank DDI — 탐색적 데이터 분석 (EDA)</h1>
<p class=mut>목적: 본 실험(Morgan 지문 + XGBoost 이진 DDI 예측) 전에 데이터를 이해하고, 특히 <b>방향성/누수 위험</b>과 <b>negative 생성 근거</b>를 확보한다. · 출처: TDC DrugBank DDI · env admet</p>

<div class=card><b>한눈에:</b> 약물 쌍 <b class=hl>{n_rows:,}</b>개 · 고유 약물 <b class=hl>{n_drugs:,}</b>개 · 상호작용 유형 <b>{n_types}</b>종(쏠림 큼).
전부 <b>positive(상호작용 있음)</b> → 이진 실험의 실질 positive 단위는 <b>무순서 유니크 쌍 {n_pos:,}개</b>. negative는 이 쌍들을 제외한 약 <b>{n_neg_space:,}개</b> 공간에서 랜덤 추출(1:1이면 {n_pos:,}개).</div>

<h2>① 데이터 크기</h2>
<table>
<tr><th class=l>항목</th><th>값</th><th class=l>메모</th></tr>
<tr><td class=l>약물 쌍(행) 수</td><td>{n_rows:,}</td><td class=l>모두 상호작용 있음(positive)</td></tr>
<tr><td class=l>고유 약물 수</td><td>{n_drugs:,}</td><td class=l>이 약물들의 SMILES로 Morgan 지문 생성</td></tr>
<tr><td class=l>상호작용 유형(Y) 수</td><td>{n_types}</td><td class=l>이진 실험에선 '유형 무관, 있으면 1'로 합침</td></tr>
</table>

<h2>② 라벨(상호작용 유형) 분포 — 쏠림 확인</h2>
<img src="{chartA}" alt="Y distribution">
<div class=card><b>관찰:</b> 86개 유형 중 <b>유형 {top_type} 하나가 {top_cnt:,}쌍({top_cnt/n_rows:.0%})</b>으로 압도적. 상위 몇 유형에 심하게 쏠려 있음(로그 스케일).<br>
<b>시사점:</b> 이번은 <b>이진 분류</b>라 86유형을 전부 "상호작용 있음=1"로 합치므로 이 쏠림이 <b>이진 라벨 균형에는 직접 영향 없음</b>(어차피 다 1). 다만 나중에 '유형별 다중분류'로 확장하면 이 불균형이 큰 이슈가 됨(메모).</div>

<h2>③ 이진 관점 & negative 생성 근거</h2>
<table>
<tr><th class=l>항목</th><th>값</th><th class=l>의미</th></tr>
<tr><td class=l>positive(무순서 유니크 쌍)</td><td>{n_pos:,}</td><td class=l>이진 실험의 실제 양성 단위</td></tr>
<tr><td class=l>가능한 전체 쌍 C(1706,2)</td><td>{n_all_possible:,}</td><td class=l>약물 2개를 고르는 모든 조합</td></tr>
<tr><td class=l>positive 밀도</td><td>{pos_density:.1%}</td><td class=l>전체 쌍 중 알려진 상호작용 비율</td></tr>
<tr><td class=l>negative 후보 공간</td><td>{n_neg_space:,}</td><td class=l>positive 제외한 나머지(여기서 랜덤 추출)</td></tr>
</table>
<div class=card><b>시사점:</b> 1:1 비율이면 negative <b>{n_pos:,}개</b> 필요 → 후보 공간 {n_neg_space:,}개로 <b>충분</b>.
★<b>한계(보고서 필수):</b> 후보 공간의 {100-pos_density*100:.1f}%가 "상호작용 없음"으로 가정되지만, 실제로는 <b>아직 안 밝혀진 진짜 양성(false negative)</b>이 섞여 있을 수 있다. 즉 우리 negative는 "진짜 없음"이 아니라 "미확인"이다.</div>

<h2>④ 데이터 품질</h2>
<table>
<tr><th class=l>점검</th><th>결과</th><th class=l>판정</th></tr>
<tr><td class=l>결측치(모든 컬럼)</td><td>{int(df.isna().sum().sum())}개</td><td class=l>{dq_flag(df.isna().sum().sum()==0)}</td></tr>
<tr><td class=l>완전 중복 행</td><td>{int(df.duplicated().sum())}개</td><td class=l>{dq_flag(df.duplicated().sum()==0)}</td></tr>
<tr><td class=l>SMILES 파싱 실패(고유 약물 {n_drugs}개 중)</td><td>{n_parse_fail}개</td><td class=l>{dq_flag(n_parse_fail==0)} {'· '+','.join(parse_fail[:5]) if parse_fail else ''}</td></tr>
<tr><td class=l>같은 ID에 다른 SMILES(불일치)</td><td>{len(conflict_ids)}개</td><td class=l>{dq_flag(len(conflict_ids)==0)}</td></tr>
</table>

<h2>⑤ ★방향성 & 누수(leakage) 위험 — 이번 실험의 핵심 함정</h2>
<table>
<tr><th class=l>점검</th><th>값</th><th class=l>의미</th></tr>
<tr><td class=l>self-pair (A,A)</td><td>{n_self}</td><td class=l>자기 자신과의 쌍(있으면 제거 대상)</td></tr>
<tr><td class=l>순서 있는 유니크 쌍 (a,b)≠(b,a)</td><td>{n_ordered_unique:,}</td><td class=l>방향 구분 시 쌍 수</td></tr>
<tr><td class=l>(a,b)와 (b,a)가 <b>둘 다</b> 존재</td><td>{n_reverse_present:,}</td><td class=l>양방향 중복 여부</td></tr>
<tr><td class=l>순서 없는 유니크 쌍 {{a,b}}</td><td>{n_unordered_unique:,}</td><td class=l>방향 무시 시 진짜 쌍 수</td></tr>
<tr><td class=l>무순서 기준 중복 행</td><td>{n_dup_unordered_rows:,}</td><td class=l>같은 쌍이 여러 유형으로 반복 등장</td></tr>
</table>
<div class=card><b>관찰:</b> {'양방향 (a,b)/(b,a)가 함께 존재하거나 ' if n_reverse_present>0 else ''}같은 약물 쌍이 여러 상호작용 유형으로 여러 번 등장 → <b>행 {n_rows:,} ≠ 무순서 유니크 쌍 {n_unordered_unique:,}</b>.<br>
<b>★설계 시사점(누수 방지):</b><br>
&nbsp;&nbsp;1) 이진 실험 전에 <b>무순서 유니크 쌍으로 dedupe</b>({n_rows:,}→{n_unordered_unique:,}) 후 라벨=1.<br>
&nbsp;&nbsp;2) negative 뽑을 때 <b>무순서 키로</b> positive 전체를 제외(양방향 모두).<br>
&nbsp;&nbsp;3) ★<b>sum/avg·|diff|·prod 융합은 (A,B)와 (B,A)가 완전히 같은 특징</b>이 된다. 만약 같은 쌍의 양방향이 train/test로 갈리면 <b>동일 특징이 양쪽에 = 누수</b>. → 무순서 dedupe로 원천 차단(양방향이 애초에 하나가 됨).<br>
&nbsp;&nbsp;4) concat도 (A,B)=[fpA,fpB]와 (B,A)=[fpB,fpA]는 다른 벡터지만 <b>같은 쌍</b>이므로, dedupe 안 하면 사실상 같은 정보가 train·test에 = 누수. → 역시 dedupe 필수.</div>

<h2>⑥ 약물별 상호작용 수(degree) — 허브 구조</h2>
<img src="{chartB}" alt="degree distribution">
<div class=card><b>관찰:</b> 소수 '허브' 약물이 매우 많은 쌍에 등장(최대 {int(deg.max()):,}회), 대다수는 적음(중앙값 {int(deg.median())}). 전형적인 편향 네트워크.<br>
<b>시사점:</b> 랜덤 쌍 분할이라도 허브 약물은 train·test 양쪽에 등장할 수밖에 없음(쌍 예측의 본질). 이번엔 <b>랜덤 쌍 분할</b>로 감을 잡되, 이 '약물 겹침'은 낙관적 성능을 줄 수 있음을 한계로 명시. (다음 단계 아이디어: '한 번도 안 본 약물' 분할=cold-start.)</div>

<h2>⑦ 종합 관찰 → 실험 설계</h2>
<div class=card>
· 데이터는 깨끗(결측·중복행·파싱실패 거의 없음). SMILES 그대로 Morgan 지문 가능.<br>
· positive만 {n_pos:,}쌍(무순서) → negative를 후보 {n_neg_space:,}에서 1:1 랜덤(무순서 제외) 생성.<br>
· ★<b>누수 차단 3원칙</b>: (1) 무순서 dedupe, (2) 무순서 키로 positive 제외 후 negative, (3) 분할은 무순서 쌍 단위(양방향이 한 개라 자동 해결).<br>
· 융합 3방법(concat/sum/diff) 공정 비교: 같은 쌍·같은 분할·같은 seed, 분류기 XGBoost 고정.<br>
· <b>한계 메모</b>: negative는 '미확인'일 뿐 진짜 음성 보장 없음 / 허브 약물 겹침으로 낙관 편향 가능.
</div>

<div class=foot>01 DDI DrugBank EDA · TDC DrugBank DDI · 값은 실제 데이터 · 다음: 03단계(전처리·negative 생성) 전 사용자와 상의 · env admet</div>
</div></body></html>"""

    with open(OUT_HTML, "w") as f:
        f.write(html)

    # ===== 콘솔 요약(사용자 보고용) =====
    print("\n================ EDA 핵심 숫자 ================")
    print(f"· 행(쌍): {n_rows:,} · 고유 약물: {n_drugs:,} · 상호작용 유형: {n_types}종")
    print(f"· 최다 유형 {top_type} = {top_cnt:,}쌍 ({top_cnt/n_rows:.0%}) → 이진화하면 다 1이라 균형엔 무관")
    print(f"· self-pair: {n_self} · (a,b)&(b,a) 둘다: {n_reverse_present:,} · 무순서 유니크 쌍: {n_unordered_unique:,}")
    print(f"· 무순서 기준 중복 행: {n_dup_unordered_rows:,}  → 이진 dedupe 후 positive = {n_pos:,}")
    print(f"· negative 후보 공간: {n_neg_space:,} (positive 밀도 {pos_density:.1%}) → 1:1 충분")
    print(f"· 품질: 결측 {int(df.isna().sum().sum())} · 완전중복행 {int(df.duplicated().sum())} · SMILES 파싱실패 {n_parse_fail}/{n_drugs} · ID-SMILES 불일치 {len(conflict_ids)}")
    print(f"\n보고서 저장 → {OUT_HTML}")
    return dict(n_rows=n_rows, n_drugs=n_drugs, n_pos=n_pos, n_neg_space=n_neg_space,
               n_parse_fail=n_parse_fail, n_reverse=n_reverse_present, n_unordered=n_unordered_unique)


if __name__ == "__main__":
    main()

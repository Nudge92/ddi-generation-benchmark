# 공개본 구성 기록

원본 `HITS_portfolio/old/DDI/` 에서 이 저장소를 만들 때 **무엇을 뺐고 왜 뺐는지**를 적는다.
원본은 읽기만 했고 수정하지 않았다.

---

## 뺀 것

### 1. `data/` 전체 (9파일 · 81 MB)

TDC 가 배포하는 DrugBank DDI 원본과 그 파생물이다. 이 저장소에 싣지 않는다.
`src/00_load_data.py` 가 TDC 에서 다시 내려받고, `src/02_preprocess.py` 가 `SEED=42` 로
음성 샘플까지 같은 집합으로 재생성한다. 따라서 데이터를 빼도 재현 절차는 끊기지 않는다.

### 2. `results/ssiddi_warm_seed0.pt` (1.5 MB)

학습된 모델 가중치. `.gitignore` 에 `*.pt` 를 넣어 두었다.

### 3. `src/12_fetch_targets.py`

이 스크립트는 DrugBank 배포 파일(`structure_links5_1_9.csv`)을 절대경로로 직접 읽는다.
경로를 가려도 그 파일 없이는 돌지 않고, 실험 12 의 결과는 이 저장소의 결과 표에 쓰이지 않는다.
→ README 에도 같은 내용을 한 줄 적는다(Phase 3).

### 4. HTML 보고서 1개 — `results/01_ddi_drugbank_eda.html`

데이터 품질 표에 DrugBank 접근번호 1건(`DB11630`, SMILES 파싱 실패 약물)이
**렌더링된 값으로** 들어 있다. 코드 주석이나 식별자 사용이 아니라 데이터에서 뽑혀 나온 값이다.

---

## 고친 것

### 개인 홈 경로 마스킹 — 28파일 28줄

실행 안내 주석의 절대경로에서 홈 경로만 지우고 **환경 이름은 남겼다.**
그 줄이 담은 정보는 "개인 경로"가 아니라 "어느 환경에서 돌리는가"이기 때문이다.

```
전:  실행: <홈>/miniforge3/envs/admet/bin/python src/00_load_data.py
후:  실행: conda run -n admet python src/00_load_data.py
```
(위 "전" 줄의 `<홈>` 자리에 원래는 사용자 홈 절대경로가 있었다.
 이 기록 문서에도 그 경로를 그대로 옮기지 않는다.)

| 환경 | 파일 수 |
|---|---:|
| `admet` (주 환경) | 27 |
| `ADMET_AI` (`07_gen_cyp.py` 전용) | 1 |
| 해당 없음 (`ddi_common.py`) | 0 |

마스킹 후 저장소 전체에 사용자 홈 절대경로는 0건이다.

---

## 뺐다가 되돌린 것 — `results/20_ddi_coattention_interpret.html`

이 보고서는 co-attention 이 어느 원자를 봤는지를 분자 그림 위에 색칠해 보여 준다.
RDKit `rdMolDraw2D` 로 그린 분자 구조 12장이 base64 로 박혀 있다
(본문 텍스트에는 약물 ID·약물명·SMILES 가 0건이다 — 그림만 검사 대상이었다).

**처음에는 뺐다.** 그림의 원천인 `data/drug_smiles.csv` 가 TDC 가 함께 내려주는 SMILES 인지,
따로 받은 DrugBank 배포 파일에서 뽑은 것인지 특정하지 못했기 때문이다.

**그 뒤 00·02 를 따라가 확정했다 — TDC 다운로드에서 그대로 나온다.**

```
00_load_data.py:49   TDC DDI(name='DrugBank').get_data()
                     → data/drugbank_ddi_raw.csv
                       (컬럼 Drug1_ID · Drug1=SMILES · Drug2_ID · Drug2=SMILES · Y)
02_preprocess.py:48  그 두 SMILES 컬럼으로 id2smi 사전 구성
02_preprocess.py:105 → data/drug_smiles.csv
```

이 저장소가 다루는 다른 모든 것과 출처가 같다. 따로 받은 DrugBank 배포 파일
(`structure_links5_1_9.csv`)과는 무관하다 — 그 파일은 뺀 `12_fetch_targets.py` 만 썼다.

출처가 확정돼 **복원했다.** 수치 산출물 `coattn_gate2.json` · `coattn_interpret.json` 은
처음부터 포함돼 있었다.

---

## 그대로 둔 것

### DrugBank 접근번호 6건

| 위치 | 내용 |
|---|---|
| `src/16_ssiddi_gnn.py:330` | `samples = ["DB00006","DB00014","DB01059","DB00898","DB09143"]` — 그래프 변환이 제대로 되는지 보려고 찍는 대표 약물 5개 |
| `src/02_preprocess.py:7` | 주석에 `DB11630` 1건 — SMILES 파싱에 실패해 제외한 약물 |

접근번호 6건이 코드에 남아 있다. SMILES·약물명·상호작용 내용은 붙어 있지 않다.
두 곳에 같은 기준을 적용했다 — 한쪽을 가리면 다른 쪽도 함께 가린다.

### `results/*.json` 25개 전부

25개 전수 검사에서 DrugBank 접근번호 0건, SMILES 0건이다.
`coattn_gate2.json` 의 `drugA: 1415` 같은 값은 DrugBank ID 가 아니라 내부 행 번호다.

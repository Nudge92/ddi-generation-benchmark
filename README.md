# DDI 세대 벤치마크 — 1~4세대 방법을 같은 조건에서 비교

> **신약-신약 조건(cold-cold)에서 2~4세대 딥러닝과 트리 모델 6개 구성이 서로 구분되지 않았다.**
> macro-F1 이 **0.219 ~ 0.284** 에 모였다. 다수 클래스 기준선은 0.009 다.
> 아키텍처를 1세대에서 4세대까지 바꿔도 warm 대비 잔존율은 **30~34%** 로 같았다.

DrugBank 약물-약물 상호작용 86종 분류. 데이터는 이 저장소에 없고, 코드가 TDC 에서 내려받는다.

---

## 무엇을 했나

| | 내용 |
|---|---|
| **네 방법 구현** | 1세대 Vilar(유사도 전파·무학습) · 2세대 DeepDDI(SSP+PCA → DNN 8층×2048) · 3세대 DDIMDL(특징별 sub-model → 확률 평균) · 4세대 SSI-DDI(분자그래프 + GAT 4층 + co-attention + RESCAL) |
| **통일 조건 비교** | 같은 데이터·같은 분할·같은 지표. 다중분류(86종, macro-F1)와 이진(ROC-AUC) 두 과제로 각각 |
| **원인 분리** | 표현을 고정하고 모델만 바꾸기 / 모델을 고정하고 표현만 바꾸기 |
| **용량 탐색** | 4세대를 375K → 6.06M 파라미터까지 키워 "낮은 것이 설정 탓인지" 확인 |

데이터 191,392 쌍(positive) · 고유 약물 1,706 · 유형 86종 · 시드 0~4 고정 · 분할 warm / cold-start / cold-cold.

---

## 결과

### 1. 다중분류 macro-F1 (86종 · 5시드 평균 ± 표준편차)

| 모델 | warm | cold-start | **cold-cold** |
|---|---:|---:|---:|
| 3세대 DDIMDL (구조 단독) | **0.8365 ± 0.0123** | 0.5321 ± 0.0247 | **0.2837 ± 0.0363** |
| 3세대 DDIMDL (확률 평균) | 0.8346 ± 0.0133 | 0.5344 ± 0.0231 | 0.2821 ± 0.0382 |
| 2세대 DeepDDI | 0.7576 ± 0.0133 | 0.4643 ± 0.0154 | 0.2569 ± 0.0326 |
| XGBoost (구조+효소) | 0.7831 ± 0.0119 | **0.5444 ± 0.0347** | 0.2550 ± 0.0399 |
| XGBoost (구조) | 0.7847 ± 0.0103 | 0.5409 ± 0.0299 | 0.2515 ± 0.0300 |
| 4세대 SSI-DDI | 0.7206 ± 0.0166 | 0.4466 ± 0.0167 | **0.2185 ± 0.0263** |
| 3세대 효소 특징 단독 | 0.0397 ± 0.0028 | 0.0322 ± 0.0023 | 0.0265 ± 0.0066 |
| **다수 클래스 기준선** | 0.0056 | 0.0057 | 0.0085 |

세대 순서대로 좋아지지 않는다. cold-cold 에서 **3세대 > 2세대 > 4세대**이고, 4세대가 세 분할 모두 최하위다.

근거: [`results/ddimdl_results.json`](results/ddimdl_results.json) · [`results/deepddi_results.json`](results/deepddi_results.json) · [`results/ssiddi_results.json`](results/ssiddi_results.json) · [`results/multiclass_results.json`](results/multiclass_results.json)

### 2. 지표를 바꾸면 순위가 바뀐다 (accuracy)

| 모델 | warm | cold-start | cold-cold |
|---|---:|---:|---:|
| 3세대 DDIMDL (구조 단독) | **0.9000 ± 0.0030** | 0.6546 ± 0.0194 | 0.4250 ± 0.0157 |
| 3세대 DDIMDL (확률 평균) | 0.8979 ± 0.0029 | **0.6595 ± 0.0205** | 0.4375 ± 0.0173 |
| 2세대 DeepDDI | 0.8799 ± 0.0030 | 0.6352 ± 0.0231 | 0.4128 ± 0.0267 |
| 4세대 SSI-DDI | 0.7996 ± 0.0068 | 0.5654 ± 0.0172 | 0.3655 ± 0.0238 |
| XGBoost (구조+효소) | 0.7970 ± 0.0013 | 0.6217 ± 0.0229 | **0.4534 ± 0.0357** |
| XGBoost (구조) | 0.7942 ± 0.0004 | 0.6186 ± 0.0240 | 0.4477 ± 0.0397 |

warm 에서 macro-F1 은 XGBoost(0.7847) > DeepDDI(0.7576) 인데 accuracy 는 DeepDDI(0.8799) > XGBoost(0.7942) 다.
cold-cold 에서 macro-F1 1위는 DDIMDL 인데 accuracy 1위는 XGBoost 다.
**지표를 밝히지 않으면 "딥러닝이 트리를 이겼다"와 "졌다"가 둘 다 참이 된다.**

근거: 위와 같음

### 3. 이진 과제 ROC-AUC — 1~4세대를 한 저울에

모델 코드를 그대로 쓰고 출력 head 만 2-클래스로 바꿔 재학습했다. 6:2:2 층화 분할 · 5 fold.

| 세대 | warm | cold-start | **cold-cold** |
|---|---:|---:|---:|
| 1 Vilar | 0.8711 ± 0.0014 | 0.7999 ± 0.0092 | **0.5000 ± 0.0000** |
| 2 DeepDDI | **0.9552 ± 0.0021** | 0.8454 ± 0.0102 | 0.7091 ± 0.0144 |
| 3 DDIMDL | 0.9525 ± 0.0010 | **0.8460 ± 0.0090** | **0.7338 ± 0.0110** |
| 4 SSI-DDI | 0.9298 ± 0.0057 | 0.8112 ± 0.0024 | 0.6878 ± 0.0140 |

1세대는 cold-cold 에서 정확히 0.5000 ± 0.0000, F1 0.0000 이다. 표준편차가 0 인 것은 우연이 아니라
양쪽 약물이 모두 새로우면 알려진 이웃이 없어 유사도 전파가 구조적으로 degenerate 하기 때문이다.

근거: [`results/benchmark_binary_results.json`](results/benchmark_binary_results.json)

### 4. 원인 분리 — 모델 축과 표현 축 (cold-cold macro-F1)

**A. 표현 고정 → 모델만 교체**

| | XGBoost | DNN | 차이 |
|---|---:|---:|---:|
| SSP 표현 | 0.1212 ± 0.0318 | 0.2569 ± 0.0326 | +0.1357 |
| Morgan 표현 | 0.2583 ± 0.0307 | 0.2698 ± 0.0370 | +0.0115 |

트리는 Morgan(희소 이진)에서는 DNN 과 거의 같고, SSP(밀집 PCA 연속)에서만 크게 뒤진다.
모델 단독도 표현 단독도 아닌 **둘의 상호작용**이다.

**B. 모델 고정(SSI-DDI) → 표현만 교체**

| | warm | cold-start | cold-cold |
|---|---:|---:|---:|
| 분자그래프 | 0.7214 ± 0.0128 | 0.4471 ± 0.0253 | 0.2221 ± 0.0315 |
| 결합 엣지 제거 | 0.6335 ± 0.0093 | 0.4127 ± 0.0236 | 0.2013 ± 0.0232 |
| 그래프 기여 | +0.0879 | +0.0344 | +0.0208 |

결합 연결 정보의 기여가 과제가 어려워질수록 줄어든다.

근거: [`results/model_swap_results.json`](results/model_swap_results.json)

**★ 이 비교에서 유의성을 주장하지 않는다.** n=5 에서 양측 부호검정이 도달할 수 있는 최소 p 는
**0.0625** 이므로 어떤 결과도 p<0.05 가 될 수 없다([`src/18_model_swap.py:323`](src/18_model_swap.py)).
5 fold 는 상관 표본이라 t-검정을 주 근거로 쓰지 않고 방향일치(5/5)와 부호검정을 썼다.

### 5. 4세대 용량 탐색 — "낮은 것이 설정 탓인가"

실행 전에 해석 규칙을 정했다: 오르면 설정이 작았던 것, 안 오르면 데이터/표현 병목
([`src/19_ssiddi_tune.py:5-7`](src/19_ssiddi_tune.py)). 비교 목표도 실행 시점 config 에 고정했다
(기준 0.2185 · 3세대 0.2837).

| 설정 | 파라미터 | warm | cold-start | **cold-cold** |
|---|---:|---:|---:|---:|
| d64·L4 | 374,678 | 0.7263 ± 0.0086 | 0.4478 ± 0.0233 | 0.2320 ± 0.0261 |
| d128·L6 | 1,520,538 | **0.7658 ± 0.0157** | 0.4843 ± 0.0247 | **0.2495 ± 0.0319** |
| d256·L6 | 6,055,578 | 0.7586 ± 0.0245 | **0.4876 ± 0.0368** | 0.2494 ± 0.0386 |

파라미터를 16배 늘려도 3세대(0.2837)에 미달했고 1.52M 에서 포화됐다.
증가분(+0.031)은 시드 산포(d128·L6 에서 0.1956~0.2953) 안에 있어 올랐다고 단정할 수 없다.

근거: [`results/ssiddi_tune_stage1.json`](results/ssiddi_tune_stage1.json) · [`results/ssiddi_tune_stage2.json`](results/ssiddi_tune_stage2.json)

### 6. 4세대가 무엇을 보고 있었나 — co-attention 해석

[`results/20_ddi_coattention_interpret.html`](results/20_ddi_coattention_interpret.html)
— SSI-DDI 의 co-attention 가중치를 원자 단위로 되돌려, **모델이 두 약물의 어느 원자에 주목해
상호작용 유형을 골랐는지**를 분자 구조 위에 빨갛게 칠한 그림 12장이다.
성능 수치가 아니라 판단 근거를 본 것이고, 위 결과 표에는 쓰이지 않는다.

근거: [`results/coattn_gate2.json`](results/coattn_gate2.json) · [`results/coattn_interpret.json`](results/coattn_interpret.json)

---

## 이 저장소가 주장하지 않는 것

| 주장하지 않는 것 | 이유 |
|---|---|
| **"논문 성능을 재현했다"** | 원 논문이 보고한 수치가 이 저장소의 기록에 없다. 논문의 구조를 구현한 것과, 논문과 다르게 한 지점(아래 표)만 적는다. |
| **"판정 기준을 사전등재했다"** | 등재 파일이 없다. 각 스크립트가 자기 HTML 보고서를 생성하므로 결과를 본 뒤 해석 문장을 넣게 되고, 파일 수정시각으로는 기준이 결과보다 앞섰는지 보일 수 없다. |
| **warm 수치를 대표값으로 쓰는 것** | warm 분할은 같은 약물이 train·test 양쪽에 있어 자기 유형 암기분이 들어간다. 원본 코드가 "warm 절대값 과대"라고 적었다([`src/16_ssiddi_gnn.py:667`](src/16_ssiddi_gnn.py)). 핵심 지표는 cold 다. |
| **"DDIMDL 의 다특징 결합이 무효다"** | 원 논문은 4특징(구조·타깃·효소·경로)인데 여기서는 2특징(구조·효소)만 쓴 축소판이다. 이 조건에서 평균이 구조 단독보다 낮았다는 것까지만 말할 수 있다. |
| **"그래프 표현이 열등하다"** | 4세대는 미튜닝·소용량 설정이라 하한이다. 그 하한 주장을 §5 에서 직접 시험했다. 올랐는지는 시드 산포 안이라 단정할 수 없었다. |

---

## 논문과 다르게 한 지점

| 모델 | 다르게 한 것 | 위치 |
|---|---|---|
| DeepDDI | 논문은 multi-label + sigmoid + 임계 0.47 → **86종 softmax 다중분류**로 단순화(쌍당 1유형이 99.8%) | [`src/14_deepddi_dnn.py:18`](src/14_deepddi_dnn.py) |
| DeepDDI | BatchNorm 추가(8층 학습 안정화). 출력형식 세부는 재확인 필요로 남겨 둠 | `src/14_deepddi_dnn.py:401,403` |
| DDIMDL | 4특징 → **2특징(구조·효소)**. 논문은 가중결합, 여기서는 단순평균 | [`src/15_ddimdl_multimodal.py:377`](src/15_ddimdl_multimodal.py) |
| SSI-DDI | 논문은 관계별 sigmoid 스코어 → **86 softmax**. RESCAL 스코어 헤드는 논문 구조를 따름 | [`src/16_ssiddi_gnn.py:670`](src/16_ssiddi_gnn.py) |
| 공통(이진 과제) | head 만 2-클래스로 바꿔 재학습. 세부는 원 논문과 다름 | [`src/17_benchmark_binary.py:563`](src/17_benchmark_binary.py) |

1세대 Vilar 만 논문 보고값(민감도 0.68 / 특이도 0.96)과 같은 특이도에서 대조한다
([`src/22_vilar_threshold.py:6`](src/22_vilar_threshold.py) · [`results/vilar_threshold_results.json`](results/vilar_threshold_results.json)).

---

## 데이터

이 저장소에는 데이터 파일을 포함하지 않는다. 코드와 결과 표만 들어 있다.

### 출처

약물-약물 상호작용 데이터는 Therapeutics Data Commons(TDC)가 배포하는 DrugBank DDI 데이터셋이다.
코드는 TDC 패키지를 통해 내려받는다.

```python
from tdc.multi_pred import DDI
data = DDI(name="DrugBank", path="data")
```

TDC 와 DrugBank 는 각각 자체 이용 약관을 가진다. 이 저장소는 두 약관의 전문을 옮기지 않으며,
해석도 하지 않는다. 데이터를 쓰려면 각 배포처의 약관을 직접 확인해야 한다.
 - TDC: https://tdcommons.ai
 - DrugBank: https://go.drugbank.com

분자 SMILES 도 이 다운로드에 함께 들어 있다(`drugbank_ddi_raw.csv` 의 `Drug1` · `Drug2` 컬럼).
`src/02_preprocess.py` 가 거기서 `data/drug_smiles.csv` 를 만든다.

### 규모 (내려받은 판본 기준)

| | |
|---|---|
| 원본 | 191,808 쌍 · 고유 약물 1,706 · 상호작용 유형 86종 (전부 positive) |
| 무순서 중복 제거 후 positive | 191,392 쌍 |
| 이진 과제 | positive 191,392 + 무작위 negative 191,392 = 382,784 행 |

TDC 판본이 바뀌면 이 숫자가 달라질 수 있다. 결과 표의 수치는 위 판본에서 나온 것이다.

### negative 정의

DrugBank 에는 양성만 있으므로, 데이터에 없는 무순서 쌍을 무작위로 뽑아 1:1 로 음성을 만든다.
진짜 양성은 무순서 키로 전부 제외한다. **이 음성은 "상호작용이 없음"이 아니라 "아직 기록되지 않음"이다.**

구조 유사도로 '어려운' 음성을 만드는 방식도 시험했고, warm ROC-AUC 가 0.8766 → 0.9417 로 **올랐다**.
상호작용 쌍의 Tanimoto 가 0.106 으로 무작위(0.090)와 거의 같아서, '닮은' 음성이 양성보다 더 닮게 되고
"더 닮으면 음성"이라는 역방향 지름길이 생긴다. 그래서 이후 실험은 무작위 음성을 쓴다
([`src/17_benchmark_binary.py:16`](src/17_benchmark_binary.py)).

---

## 재현

```bash
conda create -n admet python=3.12.13
conda activate admet
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # CPU 는 /whl/cpu
pip install -r requirements.txt

python src/00_load_data.py     # TDC 에서 내려받아 data/ 생성
python src/02_preprocess.py    # negative 생성 (SEED=42 고정)
python src/14_deepddi_dnn.py   # 이후 번호 순서대로
```

무작위 요소는 모두 시드가 고정돼 있다(negative 추출 SEED=42 · 분할·초기화 seed 0~4).
따라서 데이터를 동봉하지 않아도 음성 샘플까지 같은 집합으로 재생성된다.

**계산 규모** — GPU(RTX 4060 Ti) 기준 실측. 각 `results/*.json` 의 `runtime_s` 에 있다.

| 단계 | 시간 |
|---|---:|
| 16 SSI-DDI 전체 | 1,980 s (약 33분) |
| 17 이진 통일 벤치마크 | 9,561 s (약 2.7시간) + 4세대 재학습 10,907 s (약 3시간) |
| 19 용량 탐색 stage2 | 12,992 s (약 3.6시간) |

CPU 로도 전 단계가 동작하지만 GNN 을 쓰는 16·17·18·19 는 위 시간보다 훨씬 오래 걸린다.

**별도 환경** — `src/07_gen_cyp.py` 하나만 ADMET-AI 가 필요하다(Python 3.11.15).
`requirements-admet-ai.txt` 참조.

---

## 이 저장소에 없는 것

- `data/` — 위 "데이터" 절 참조
- 학습된 모델 가중치(`.pt`)
- `src/12_fetch_targets.py` (타깃 특징 **수집**) — DrugBank 배포 파일을 직접 읽으므로 포함하지 않았다.
  같은 번호의 `12_targets.py`(분석)는 들어 있다. 그 실험의 결과는 이 저장소의 결과 표에 쓰이지 않는다.
- `results/01_ddi_drugbank_eda.html` — 데이터 품질 표에 약물 접근번호가 렌더링돼 있다

자세한 구성 기록은 [`PUBLICATION_NOTES.md`](PUBLICATION_NOTES.md) 에 있다.

"""
[DDI 실험 - 02단계] 전처리 + negative 생성 (누수 차단)
====================================================================
EDA(01)에서 확인한 함정을 반영해, 학습용 라벨 데이터셋을 만든다.

★누수 차단 3원칙 (EDA에서 사용자와 합의):
  1) 정제 : SMILES 파싱 실패 약물(DB11630) 낀 쌍 제거 + self-pair 제거
  2) dedupe: (A,B)와 (B,A)를 '무순서 {A,B}'로 합쳐 유니크화 → positive(=1)
             (양방향/다중유형 중복 제거 = train/test 누수 원천 차단)
  3) negative: '무순서 키'로 모든 positive를 제외한 공간에서 랜덤 1:1 추출(=0)

출력:
  data/drug_smiles.csv       : 유효 약물 ID → SMILES (지문 계산용 사전)
  data/ddi_pairs_labeled.csv : drug_a, drug_b, label(1/0)  ← 학습 입력

실행: conda run -n admet python src/02_preprocess.py
env : admet
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE, "data", "drugbank_ddi_raw.csv")
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")

SEED = 42                 # negative 추출 재현용 고정 시드
NEG_RATIO = 1.0           # positive : negative = 1 : 1


def ukey(a, b):
    """무순서 키: (A,B)와 (B,A)를 같은 것으로. 항상 정렬해 튜플로."""
    return (a, b) if a <= b else (b, a)


def main():
    print("[02] 전처리 시작 ...", flush=True)
    df = pd.read_csv(RAW_CSV)
    df["Drug1_ID"] = df["Drug1_ID"].astype(str)
    df["Drug2_ID"] = df["Drug2_ID"].astype(str)
    n_raw = len(df)

    # ---------- (1) 약물ID→SMILES 사전 + 파싱 유효성 ----------
    id2smi = {}
    for ci, cs in [("Drug1_ID", "Drug1"), ("Drug2_ID", "Drug2")]:
        for i, s in zip(df[ci], df[cs]):
            id2smi[i] = s
    # RDKit로 파싱되는 약물만 '유효'로. (파싱 실패 = 지문 계산 불가 → 버림)
    valid_ids = {i for i, s in id2smi.items() if Chem.MolFromSmiles(str(s)) is not None}
    invalid_ids = set(id2smi) - valid_ids
    print(f"  약물 {len(id2smi)}개 중 유효 {len(valid_ids)} · 무효(파싱실패) {len(invalid_ids)}: {sorted(invalid_ids)}")

    # ---------- (1) 정제: 무효 약물 낀 쌍 + self-pair 제거 ----------
    is_self = df["Drug1_ID"] == df["Drug2_ID"]
    has_invalid = df["Drug1_ID"].isin(invalid_ids) | df["Drug2_ID"].isin(invalid_ids)
    n_self = int(is_self.sum()); n_invalid_pairs = int(has_invalid.sum())
    df = df[~is_self & ~has_invalid].copy()
    print(f"  제거: self-pair {n_self} · 무효약물 낀 쌍 {n_invalid_pairs} → 남은 행 {len(df):,}")

    # ---------- (2) 무순서 dedupe → positive ----------
    pos_keys = set()
    for a, b in zip(df["Drug1_ID"], df["Drug2_ID"]):
        pos_keys.add(ukey(a, b))
    n_pos = len(pos_keys)
    print(f"  무순서 dedupe: 행 {len(df):,} → positive(유니크 쌍) {n_pos:,}")

    # ---------- (3) negative 생성 (무순서 키로 positive 제외, 랜덤 1:1) ----------
    valid_list = np.array(sorted(valid_ids))       # 유효 약물만 후보(무효 약물은 지문 불가)
    n_valid = len(valid_list)
    n_neg_target = int(round(n_pos * NEG_RATIO))
    rng = np.random.RandomState(SEED)
    neg_keys = set()
    batch = 1_000_000
    tries = 0
    while len(neg_keys) < n_neg_target:
        ai = rng.randint(0, n_valid, size=batch)
        bi = rng.randint(0, n_valid, size=batch)
        for x, y in zip(ai, bi):
            if x == y:
                continue                            # 같은 약물끼리 X
            k = ukey(valid_list[x], valid_list[y])
            if k in pos_keys or k in neg_keys:      # ★진짜 양성 제외 + 중복 방지
                continue
            neg_keys.add(k)
            if len(neg_keys) >= n_neg_target:
                break
        tries += batch
    print(f"  negative 생성: {len(neg_keys):,}개 (시도 {tries:,}회, 시드 {SEED})")

    # ---------- 누수 안전성 검사(양성∩음성 = 공집합이어야 함) ----------
    overlap = pos_keys & neg_keys
    assert len(overlap) == 0, f"★위험: positive와 negative가 겹침 {len(overlap)}"
    print(f"  ✔ 안전성 검사 통과: positive∩negative = 0 (진짜 양성을 음성으로 안 찍음)")

    # ---------- 합쳐서 라벨 데이터셋 ----------
    rows = [(a, b, 1) for (a, b) in pos_keys] + [(a, b, 0) for (a, b) in neg_keys]
    pairs = pd.DataFrame(rows, columns=["drug_a", "drug_b", "label"])
    # 셔플(양성/음성 섞기) — 재현 위해 시드 고정
    pairs = pairs.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # ---------- 저장 ----------
    smi_df = pd.DataFrame({"drug_id": sorted(valid_ids)})
    smi_df["smiles"] = smi_df["drug_id"].map(id2smi)
    smi_df.to_csv(DRUG_SMI_CSV, index=False)
    pairs.to_csv(PAIRS_CSV, index=False)

    # ---------- 보고 ----------
    print("\n================ 전처리 결과 요약 ================")
    print(f"· 원본 행: {n_raw:,} → 정제 후: {len(df):,}")
    print(f"· positive(무순서 유니크 쌍): {n_pos:,}  (label=1)")
    print(f"· negative(랜덤, positive 제외): {len(neg_keys):,}  (label=0)")
    print(f"· 최종 학습 쌍: {len(pairs):,} · 라벨 균형: {pairs['label'].mean():.3f}(=positive 비율)")
    print(f"· 유효 약물(지문 사전): {len(valid_ids):,}")
    print(f"· 저장: {os.path.basename(PAIRS_CSV)} , {os.path.basename(DRUG_SMI_CSV)}")
    print(f"· 예시 negative 쌍 3개: {[ (a,b) for (a,b) in list(neg_keys)[:3] ]}")
    print("\n[02] 완료. 다음: 04단계 융합 3방법(concat/sum/diff) 실험 전 사용자 확인.")
    return dict(n_pos=n_pos, n_neg=len(neg_keys), n_total=len(pairs), n_valid=len(valid_ids),
               n_invalid=len(invalid_ids), n_self=n_self, n_invalid_pairs=n_invalid_pairs)


if __name__ == "__main__":
    main()

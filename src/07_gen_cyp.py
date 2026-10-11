"""
[DDI 실험 - 07단계-A] 효소(CYP) 특징 생성 — ADMET-AI 예측
====================================================================
각 약물 SMILES에 ADMET-AI를 돌려 CYP 8종 예측을 얻고, 이진 벡터(있음1/없음0)로.
  - 억제(inhibitor) 5종: CYP1A2/2C19/2C9/2D6/3A4 (Veith)
  - 기질(substrate) 3종: CYP2C9/2D6/3A4 (Carbon-Mangels)
임계 0.5로 이진화(Morgan과 같은 이진 형식). 예측이라 전 약물 커버(결측 없음).

★주의: 이 CYP는 SMILES에서 '예측'한 것(구조 유래) — 진짜 DrugBank 주석 아님.
       따라서 struct+enzyme은 struct와 일부 중복. 순환 위험은 낮으나(구조→CYP),
       라벨(DDI)이 CYP기반일 수 있어 해석은 cold-cold를 주 근거로.

출력: data/drug_cyp.csv (drug_id + 8개 이진 CYP 컬럼)
실행: conda run -n ADMET_AI python src/07_gen_cyp.py   (★ADMET_AI env)
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
OUT_CSV = os.path.join(BASE, "data", "drug_cyp.csv")

# ADMET-AI가 내놓는 CYP 확률 컬럼(퍼센타일 말고 raw prob)
CYP_COLS = ["CYP1A2_Veith", "CYP2C19_Veith", "CYP2C9_Veith", "CYP2D6_Veith", "CYP3A4_Veith",
            "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels", "CYP3A4_Substrate_CarbonMangels"]
THRESH = 0.5


def main():
    smi = pd.read_csv(DRUG_SMI_CSV)
    drug_ids = smi["drug_id"].astype(str).tolist(); smiles = smi["smiles"].tolist()
    print(f"[07-A] ADMET-AI CYP 예측: 약물 {len(drug_ids):,}", flush=True)

    from admet_ai import ADMETModel
    model = ADMETModel()
    preds = model.predict(smiles=smiles)          # DataFrame, index=SMILES
    preds = preds.reset_index(drop=True)

    out = pd.DataFrame({"drug_id": drug_ids})
    prob = preds[CYP_COLS].to_numpy(dtype=np.float64)
    binv = (prob >= THRESH).astype(np.int8)        # 이진화
    # 짧은 컬럼명
    short = ["cyp1a2_inh", "cyp2c19_inh", "cyp2c9_inh", "cyp2d6_inh", "cyp3a4_inh",
             "cyp2c9_sub", "cyp2d6_sub", "cyp3a4_sub"]
    for j, s in enumerate(short):
        out[s] = binv[:, j]
    out.to_csv(OUT_CSV, index=False)

    print(f"  저장 → {OUT_CSV} ({len(short)}개 CYP 이진 특징)", flush=True)
    print("  CYP별 양성(1) 비율:")
    for s in short:
        print(f"    {s:12s} {out[s].mean():.3f}")
    n_any = (out[short].sum(axis=1) > 0).mean()
    print(f"  ★하나라도 CYP=1인 약물 비율: {n_any:.3f} · 평균 CYP 개수 {out[short].sum(axis=1).mean():.2f}/8")
    print("[07-A] 완료.")


if __name__ == "__main__":
    main()

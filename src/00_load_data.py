"""
[DDI 실험 - 00단계] DrugBank DDI 데이터 다운로드 & 로드
====================================================================
목적: TDC(Therapeutics Data Commons)에서 DrugBank 약물-약물 상호작용(DDI)
      데이터를 받아서, 내용을 확인하고, 로컬(DDI/data/)에 저장해 둔다.
      (이후 단계들은 인터넷 없이 이 로컬 파일만 읽는다 = 재다운로드 방지)

왜 TDC인가?
  - 이미 설치돼 있고(`pip install PyTDC`), 한 줄로 받을 수 있다.
  - 각 약물의 SMILES(분자 구조 문자열)를 '함께' 준다 → Morgan 지문에 바로 사용.
  - DrugBank DDI에는 '상호작용 있음(positive)'만 있다 → negative는 우리가 직접 만든다.

실행: conda run -n admet python src/00_load_data.py
env : admet
"""
import os
import pandas as pd

# --- 경로 설정 -------------------------------------------------------
# BASE = DDI 프로젝트 루트. 이 파일(src/00_load_data.py) 기준으로 한 칸 위.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")          # 원본/가공 데이터 저장 위치
os.makedirs(DATA_DIR, exist_ok=True)
RAW_CSV = os.path.join(DATA_DIR, "drugbank_ddi_raw.csv")   # 우리가 저장할 로컬 사본


def download_and_load():
    """TDC에서 DrugBank DDI를 받아 pandas DataFrame으로 돌려준다."""
    # TDC의 multi_pred(여러 개체를 입력으로 받는 예측) 모듈에서 DDI 클래스 사용.
    from tdc.multi_pred import DDI
    # name='DrugBank' : DrugBank 출처의 DDI.
    # path=DATA_DIR   : 받은 원본 파일을 DDI/data/ 안에 저장(홈 폴더 안 어지럽힘).
    data = DDI(name="DrugBank", path=DATA_DIR)
    # get_data() : 전체를 하나의 DataFrame으로. (train/test 분할은 우리가 나중에 직접)
    df = data.get_data()
    return df


def main():
    print("[00] DrugBank DDI 다운로드/로드 시작 ...", flush=True)

    # 이미 로컬 사본이 있으면 재다운로드하지 않는다(빠르게 재실행).
    if os.path.exists(RAW_CSV):
        print(f"  로컬 사본 발견 → 재사용: {RAW_CSV}", flush=True)
        df = pd.read_csv(RAW_CSV)
    else:
        df = download_and_load()
        # 로컬에 저장(다음 단계들이 이걸 읽음).
        df.to_csv(RAW_CSV, index=False)
        print(f"  저장 완료 → {RAW_CSV}", flush=True)

    # ===== 로드 결과 간단 보고(1단계에서 사용자에게 보여줄 핵심 숫자) =====
    print("\n================ 로드 결과 요약 ================")
    print(f"· 전체 행(약물 쌍) 수 : {len(df):,}")
    print(f"· 컬럼               : {list(df.columns)}")

    # TDC DDI 표준 컬럼: Drug1_ID, Drug1(SMILES), Drug2_ID, Drug2(SMILES), Y(상호작용 유형 1~86)
    # 고유 약물 수 = Drug1_ID와 Drug2_ID를 합쳐서 중복 제거.
    ids = pd.concat([df["Drug1_ID"], df["Drug2_ID"]]).astype(str)
    print(f"· 고유 약물 수       : {ids.nunique():,}")

    # 라벨(Y) = 상호작용 '유형' 번호. 몇 종류인지, 가장 흔한 유형 몇 개.
    print(f"· 상호작용 유형 수(Y): {df['Y'].nunique()} 종")
    print(f"· Y 상위 5개 유형(개수):")
    for y, cnt in df["Y"].value_counts().head(5).items():
        print(f"    유형 {y:>3} : {cnt:,} 쌍")

    # SMILES가 실제로 들어있는지(첫 행 예시).
    print(f"\n· 예시 행(첫 1개):")
    r = df.iloc[0]
    print(f"    Drug1_ID={r['Drug1_ID']}  SMILES={str(r['Drug1'])[:50]}...")
    print(f"    Drug2_ID={r['Drug2_ID']}  SMILES={str(r['Drug2'])[:50]}...")
    print(f"    Y(상호작용 유형)={r['Y']}")

    # 아주 기본적인 품질 신호(자세한 건 01 EDA에서).
    print(f"\n· 결측치 개수(컬럼별):\n{df.isna().sum().to_string()}")
    print(f"· 완전 중복 행 수    : {df.duplicated().sum():,}")

    print("\n[00] 완료. 다음 단계: 01 EDA (상세 탐색).")
    return df


if __name__ == "__main__":
    main()

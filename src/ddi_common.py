"""
[DDI 공용 모듈] 지문 생성 · 융합 · 평가 (02·03 실험이 공유 → 중복 구현 방지)
====================================================================
02(03_train_fusion.py)에서 쓴 함수를 여기로 모아 재사용한다.
로직은 02와 '완전히 동일'(결과 재현됨).
"""
import numpy as np
import scipy.sparse as sp
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
RDLogger.DisableLog("rdApp.*")

# 다중분류(10·11 공용) — multi:softprob. 10과 동일 하이퍼파라미터.
XGB_MC = dict(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist", objective="multi:softprob", n_jobs=8, eval_metric="mlogloss")


def fit_mc(Xtr, ytr, sample_weight=None):
    """다중분류 학습. ★train 유형이 0..K-1 부분집합(cold서 유형 구멍)이라 LabelEncoder로 압축.
    sample_weight=class weight 등 샘플 가중치(불균형 대응·11에서 사용). 반환:(clf, le)."""
    le = LabelEncoder(); yt = le.fit_transform(ytr)
    clf = xgb.XGBClassifier(**XGB_MC); clf.fit(Xtr, yt, sample_weight=sample_weight)
    return (clf, le)


def top3_acc(proba, classes, y_true):
    """top-3 정확도: 정답이 확률 상위 3개 클래스 안에 드나. classes=원 라벨(le.classes_)."""
    top3 = classes[np.argsort(proba, axis=1)[:, -3:]]
    return float(np.mean([y_true[i] in top3[i] for i in range(len(y_true))]))

FP_BITS = 2048
# XGBoost 설정 — 02와 동일(공정 비교). 튜닝 없음.
XGB_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                  eval_metric="logloss", n_jobs=8, random_state=0)
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)


def ukey(a, b):
    """무순서 키: (A,B)와 (B,A)를 같은 것으로. 항상 정렬."""
    return (a, b) if a <= b else (b, a)


def build_fp_matrix(drug_ids, id2smi):
    """약물별 Morgan 지문을 희소행렬(csr)로. 행=약물, 열=2048비트."""
    indptr = [0]; indices = []
    for did in drug_ids:
        m = Chem.MolFromSmiles(str(id2smi[did]))
        on = list(_GEN.GetFingerprint(m).GetOnBits())   # 켜진 비트 위치만
        indices.extend(on); indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.uint8)
    return sp.csr_matrix((data, indices, np.array(indptr)),
                         shape=(len(drug_ids), FP_BITS), dtype=np.uint8)


def make_features(FP, idxA, idxB, methods=("concat", "sum", "diff")):
    """융합법 특징 행렬(csr). methods로 필요한 것만 만들 수 있음(기본=셋 다, 02와 동일)."""
    A = FP[idxA]      # (쌍 수 × 2048)  약물 A들의 지문
    B = FP[idxB]      # (쌍 수 × 2048)  약물 B들의 지문
    out = {}
    if "concat" in methods:
        out["concat"] = sp.hstack([A, B], format="csr").tocsr()   # (a) 4096차원
    if "sum" in methods:
        out["sum"] = (A + B).tocsr()                              # (b) 값 0/1/2
    if "diff" in methods:
        diff = (A + B)                                            # (c) |A-B| = XOR
        diff.data = (diff.data % 2).astype(np.uint8)              #   2→0(공통비트 제거), 1→1
        diff.eliminate_zeros()
        out["diff"] = diff.tocsr()
    return out


def fit_model(Xtr, ytr):
    """XGBoost 모델 하나 학습해서 돌려준다(재사용 가능 — cold regime은 한 모델로 두 test 채점)."""
    clf = xgb.XGBClassifier(**XGB_PARAMS)
    clf.fit(Xtr, ytr)
    return clf


def score_model(clf, Xte, yte):
    """학습된 모델로 test 채점 → 4지표(ROC-AUC/PR-AUC/F1/accuracy)."""
    proba = clf.predict_proba(Xte)[:, 1]      # 양성 확률
    pred = (proba >= 0.5).astype(int)         # 0.5 임계 이진 예측
    return {
        "roc_auc": float(roc_auc_score(yte, proba)),
        "pr_auc": float(average_precision_score(yte, proba)),
        "f1": float(f1_score(yte, pred)),
        "accuracy": float(accuracy_score(yte, pred)),
    }


def evaluate(Xtr, ytr, Xte, yte):
    """학습+채점 한 번에(02에서 사용). = fit_model 후 score_model."""
    return score_model(fit_model(Xtr, ytr), Xte, yte)


# ===== cold 분할(03) · Tanimoto(05) 공용 — 06이 재사용(중복 방지) =====
def build_sim_matrix(bitvects):
    """전체 약물쌍 Tanimoto 유사도 행렬(n×n). bitvects=RDKit 비트벡터 리스트."""
    n = len(bitvects); S = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        S[i] = DataStructs.BulkTanimotoSimilarity(bitvects[i], bitvects)
    return S


def split_drugs(drug_ids, seed, test_frac=0.20):
    """약물을 train/test로 랜덤 분할(약물 기준). 03과 동일 로직 → 같은 seed면 같은 분할."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(drug_ids))
    n_test = int(round(len(drug_ids) * test_frac))
    arr = np.asarray(drug_ids)
    test_drugs = set(arr[perm[:n_test]].tolist())
    train_drugs = set(arr[perm[n_test:]].tolist())
    return train_drugs, test_drugs


def bin_match(cand_sims, pos_sims, n_bins, seed):
    """비상호작용 후보(cand_sims)를 positive(pos_sims)의 Tanimoto 분포에 맞춰 인덱스 선택.
    05의 매칭 로직과 동일: positive를 bin별로 세고, 후보를 같은 bin에서 그만큼 뽑음.
    후보가 부족한 고-유사도 bin은 남은 후보 중 가장 닮은 것으로 채움. 반환:(선택인덱스, 부족분)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    pos_hist, _ = np.histogram(pos_sims, bins=bins)
    cand_bin = np.clip(np.digitize(cand_sims, bins) - 1, 0, n_bins - 1)
    rng = np.random.RandomState(seed)
    chosen = []; deficit = 0
    for b in range(n_bins):
        need = int(pos_hist[b])
        if need == 0:
            continue
        in_bin = np.where(cand_bin == b)[0]
        if len(in_bin) >= need:
            sel = rng.choice(in_bin, size=need, replace=False)
        else:
            sel = in_bin; deficit += need - len(in_bin)
        chosen.extend(np.atleast_1d(sel).tolist())
    if deficit > 0:                              # 고-유사도 부족분: 남은 후보 중 가장 닮은 것으로
        used = np.zeros(len(cand_sims), bool); used[chosen] = True
        rem = np.where(~used)[0]; rem = rem[np.argsort(-cand_sims[rem])]
        chosen.extend(rem[:deficit].tolist())
    return np.array(chosen, dtype=np.int64), int(deficit)


def regime_matched(val, g_iu, g_ju, iu, ju, sims, is_pos, pos_r_sims, drug_ids, seed, n_bins):
    """regime(val: TT=0/TS=1/SS=2)의 미지-약물 조건 + regime positive 분포 동시 만족 negative.
    06과 동일 로직(06은 인라인 사본, 여기가 06·07 공용 정본)."""
    pair_unseen = g_iu + g_ju
    cmask = (pair_unseen == val) & (~is_pos)          # 조건1: 미지약물 조건 + 비상호작용
    c_iu = iu[cmask]; c_ju = ju[cmask]; c_sims = sims[cmask]
    if len(c_sims) == 0:
        return [], 0, np.array([])
    chosen, deficit = bin_match(c_sims, pos_r_sims, n_bins, seed)   # 조건2: 분포 매칭
    pairs = [ukey(drug_ids[int(c_iu[k])], drug_ids[int(c_ju[k])]) for k in chosen]
    return pairs, deficit, c_sims[chosen]


def verify_regime(train_pairs, test_pairs, test_drugs, regime):
    """train에 미지약물 0 · test는 regime 조건 · train↔test 겹침 0 (06과 동일)."""
    bad_train = sum(1 for a, b in train_pairs if (a in test_drugs) or (b in test_drugs))
    want = 1 if regime == "cold-start" else 2
    bad_test = sum(1 for a, b in test_pairs if ((a in test_drugs) + (b in test_drugs)) != want)
    overlap = len(set(train_pairs) & set(test_pairs))
    return {"bad_train": bad_train, "bad_test": bad_test, "overlap": overlap,
            "PASS": bad_train == 0 and bad_test == 0 and overlap == 0}


def match_2d(cand_tani, cand_burden, pos_tani, pos_burden, n_tbins, seed):
    """2축(Tanimoto bin × 정수 burden) 매칭 — 08과 동일 정본. 반환:(선택인덱스, 부족분)."""
    if len(cand_tani) == 0:
        return np.array([], dtype=np.int64), 0
    tb = np.linspace(0.0, 1.0, n_tbins + 1)
    maxB = int(max(int(pos_burden.max()), int(cand_burden.max())))
    nB = maxB + 1

    def cellid(tani, burden):
        ti = np.clip(np.digitize(tani, tb) - 1, 0, n_tbins - 1)
        return ti.astype(np.int64) * nB + np.clip(burden.astype(np.int64), 0, maxB)

    pos_hist = np.bincount(cellid(pos_tani, pos_burden), minlength=n_tbins * nB)
    ccell = cellid(cand_tani, cand_burden)
    order = np.argsort(ccell, kind="stable"); ccell_s = ccell[order]
    uniq, starts = np.unique(ccell_s, return_index=True)
    cell_pool = {int(u): order[starts[i]:(starts[i + 1] if i + 1 < len(starts) else len(order))] for i, u in enumerate(uniq)}
    rng = np.random.RandomState(seed)
    chosen = []; used = np.zeros(len(cand_tani), bool); deficit = 0
    for c in np.nonzero(pos_hist)[0]:
        need = int(pos_hist[c]); pool = cell_pool.get(int(c))
        if pool is None or len(pool) == 0:
            deficit += need; continue
        if len(pool) >= need:
            sel = rng.choice(pool, need, replace=False)
        else:
            sel = pool; deficit += need - len(pool)
        chosen.extend(sel.tolist()); used[sel] = True
    if deficit > 0:
        rem = np.where(~used)[0]
        rem = rem[np.argsort(np.abs(cand_burden[rem] - pos_burden.mean()))]
        chosen.extend(rem[:deficit].tolist())
    return np.array(chosen, dtype=np.int64), int(deficit)


def regime_burden_matched(val, g_iu, g_ju, iu, ju, sims, pair_burden, is_pos, pos_r_sims, pos_r_burden, drug_ids, seed, n_tbins):
    """regime의 미지-약물 조건 + Tanimoto·CYP-burden 2축 매칭 negative (08과 동일)."""
    cmask = ((g_iu + g_ju) == val) & (~is_pos)
    c_iu = iu[cmask]; c_ju = ju[cmask]
    chosen, deficit = match_2d(sims[cmask], pair_burden[cmask], pos_r_sims, pos_r_burden, n_tbins, seed)
    pairs = [ukey(drug_ids[int(c_iu[k])], drug_ids[int(c_ju[k])]) for k in chosen]
    return pairs, deficit


def pair_feature(FP, EXTRA, idxA, idxB, config, fusion):
    """한 쌍의 특징(07 build_feat과 동일 정본). config: struct/enzyme/struct+enzyme.
    Morgan(FP)·부가벡터(EXTRA=CYP 등)를 두 약물끼리 fusion(sum/concat)으로 결합."""
    parts = []
    if config in ("struct", "struct+enzyme"):
        parts.append(make_features(FP, idxA, idxB, methods=(fusion,))[fusion])
    if config in ("enzyme", "struct+enzyme"):
        A = EXTRA[idxA]; B = EXTRA[idxB]
        parts.append((A + B).tocsr() if fusion == "sum" else sp.hstack([A, B], format="csr").tocsr())
    return parts[0] if len(parts) == 1 else sp.hstack(parts, format="csr").tocsr()

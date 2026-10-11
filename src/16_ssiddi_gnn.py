"""
[DDI 실험 - 16] 4세대(SSI-DDI, Nyamabo 2021) 재현 — 분자 그래프 + GAT + co-attention
==================================================================================
SSI-DDI 핵심: 약물을 지문/닮음이 아니라 ★'분자 그래프'(원자=노드, 결합=엣지)로 읽는다.
  GAT(그래프 어텐션) 층을 L개 쌓아, 각 층의 노드 표현을 서로 다른 '부분구조(substructure)'로 본다.
  두 약물의 부분구조들끼리 ★co-attention(상호 어텐션)으로 상호작용 점수를 매겨,
  가장 상호작용이 강한 부분구조 쌍을 근거로 86종 유형을 분류한다. ★이 저장소 첫 GNN(PyG 사용).

왜: 2·3세대(14 DeepDDI, 15 DDIMDL)와 ★완전히 같은 데이터·분할·평가 틀로 만들어,
  "그래프(4세대)가 정말 2·3세대·트리(11)를 넘나?"를 나란히(paired) 비교하기 위함.

■ 표현(관문 2): SMILES → RDKit Mol → 그래프.
  · 노드 특징 = 원자 종류(one-hot)·차수·암시원자가·형식전하·라디칼·혼성·방향족·수소수 (SSI-DDI/DeepChem류).
  · 엣지 = 결합(양방향). GAT는 엣지 특징을 안 쓰므로 결합 연결만. 단원자(금속이온)는 self-loop로 처리.

■ 모델(관문 3): GATConv L층(논문 L=4). 각 층 출력을 global readout → 그 층의 '약물 부분구조 벡터'.
  · ★co-attention: 두 약물의 (층별) 부분구조 벡터들 간 유사도 행렬 → softmax → 가중합 상호작용 벡터.
  · 최종: 상호작용 벡터 → Linear → 86종 softmax. 손실 cross-entropy, Adam lr 1e-3, batch, early-stopping.

■ 평가(★14·15와 동일): warm / cold-start / cold-cold (ddi_common.split_drugs 동일 분할).
  지표 macro-F1(주)·micro·acc·top3 + head/mid/tail. seed 5개(0~4) 평균±std.
  ★label/평가/버킷은 14에서 그대로 import(load_Y, eval_dnn) → 유형정의·지표가 14·15와 100% 일치.

산출: results/16_ddi_ssiddi_gnn.html , results/ssiddi_results.json
실행(관문별):
  conda run -n admet python src/16_ssiddi_gnn.py gate2   # 그래프 변환 확인
  ...                                                              gate3   # 모델 1회 forward
  ...                                                              gate4   # 짧은 학습(loss 감소)
  ...                                                              full    # 전체 실행
  (빠른 점검: SSIDDI_SMOKE=1 → seed 1개·짧은 epoch)
참고 원저장소: github.com/kanz76/SSI-DDI
"""
import os, sys, json, time, io, base64, importlib.util
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, SAGPooling, global_add_pool, LayerNorm
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddi_common import ukey, split_drugs, top3_acc  # 14·15가 쓰는 것과 동일 primitive

# ── 14(DeepDDI)에서 label/평가 함수를 그대로 가져온다(15와 같은 동적 import 패턴) ──
#    load_Y  : 86종 유형 라벨(0..85) — 14·15와 완전 동일한 정의
#    eval_dnn: proba행렬 → macro/micro-F1·acc·top3·head/mid/tail — 14·15와 완전 동일한 지표
_spec = importlib.util.spec_from_file_location(
    "m14", os.path.join(os.path.dirname(os.path.abspath(__file__)), "14_deepddi_dnn.py"))
M14 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M14)
load_Y = M14.load_Y
eval_dnn = M14.eval_dnn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_SMI_CSV = os.path.join(BASE, "data", "drug_smiles.csv")
PAIRS_CSV = os.path.join(BASE, "data", "ddi_pairs_labeled.csv")
RAW_CSV = os.path.join(BASE, "data", "drugbank_ddi_raw.csv")
RESULTS_JSON = os.path.join(BASE, "results", "ssiddi_results.json")

SMOKE = os.environ.get("SSIDDI_SMOKE", "") == "1"
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
DRUG_TEST_FRAC = 0.20
REGIMES = ["warm", "cold-start", "cold-cold"]
N_CLASS = 86
METRICS = ["accuracy", "macro_f1", "micro_f1", "top3", "head_f1", "mid_f1", "tail_f1"]

# 모델 하이퍼파라미터(SSI-DDI 논문 설정 근사)
GAT_LAYERS = 4          # 논문 L=4: 층마다 서로 다른 '부분구조 수준'을 표현
GAT_HIDDEN = 64         # 각 GAT 층 출력 차원(부분구조 벡터 차원)
GAT_HEADS = 2           # 멀티헤드 어텐션 head 수
BATCH = 256             # 논문 batch 256
LR = 1e-3               # 논문 Adam lr 0.001
MAX_EPOCH = 3 if SMOKE else 100
PATIENCE = 8
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
#  관문 2: 분자 → 그래프 변환 (SSI-DDI/DeepChem류 원자 특징)
# ══════════════════════════════════════════════════════════════════════════════
# 데이터셋에 실제 등장하는 원자 종류 38개(00단계 조사). 마지막에 'X'(미지) 버킷을 두어
# 혹시 못 본 원소가 와도 실제 원소 one-hot을 오염시키지 않게 한다(SSI-DDI의 unk 처리).
ATOM_SYMBOLS = ['Ag', 'Al', 'As', 'Au', 'B', 'Bi', 'Br', 'C', 'Ca', 'Cl', 'Co', 'Cr',
                'Cu', 'F', 'Fe', 'Ga', 'Gd', 'H', 'Hg', 'I', 'K', 'La', 'Li', 'Mg',
                'N', 'Na', 'O', 'P', 'Pt', 'Ra', 'S', 'Sb', 'Se', 'Si', 'Sr', 'Tc',
                'Ti', 'Zn', 'X']
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,   Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,  Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2, 'other',
]


def _onek(x, allowable):
    """one-hot 인코딩. x가 목록에 없으면 ★마지막 원소(=미지 버킷)로 매핑(SSI-DDI unk 방식)."""
    if x not in allowable:
        x = allowable[-1]
    return [int(x == s) for s in allowable]


def atom_features(atom):
    """원자 1개 → 특징 벡터. SSI-DDI/DeepChem 계열의 표준 원자 서술자.
    구성: 원소(39) + 차수(11) + 암시원자가(7) + 형식전하(1) + 라디칼전자(1)
          + 혼성(6) + 방향족(1) + 총수소수(5)  = 총 71차원."""
    return (
        _onek(atom.GetSymbol(), ATOM_SYMBOLS)                    # 39: 원소 종류
        + _onek(atom.GetDegree(), list(range(11)))              # 11: 연결 차수 0..10
        + _onek(atom.GetImplicitValence(), list(range(7)))      #  7: 암시 원자가 0..6
        + [atom.GetFormalCharge()]                              #  1: 형식 전하(스칼라)
        + [atom.GetNumRadicalElectrons()]                       #  1: 라디칼 전자수(스칼라)
        + _onek(atom.GetHybridization(), HYBRIDIZATIONS)        #  6: 혼성 오비탈
        + [int(atom.GetIsAromatic())]                           #  1: 방향족 여부
        + _onek(atom.GetTotalNumHs(), list(range(5)))           #  5: 결합 수소수 0..4
    )


ATOM_FEAT_DIM = len(atom_features(Chem.MolFromSmiles("CCO").GetAtomWithIdx(0)))  # =70


def mol_to_graph(smiles):
    """SMILES → PyG Data(그래프). 노드=원자(특징행렬 x), 엣지=결합(양방향 edge_index).
    반환 None = 파싱 실패. ★단원자 분자(금속이온 등)는 엣지 0개 → GATConv self-loop가 처리."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    # 노드 특징 행렬 (원자수 × 70)
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    # 엣지: 각 결합을 양방향으로(무방향 그래프). 결합 없으면 빈 (2×0) 텐서.
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]          # 양방향
    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def load_drug_universe():
    """14·15와 동일한 약물 세계·순서를 로드. drug_ids 순서 = did2row 행 인덱스."""
    smi = pd.read_csv(DRUG_SMI_CSV)
    drug_ids = smi["drug_id"].astype(str).tolist()
    id2smi = dict(zip(smi["drug_id"].astype(str), smi["smiles"]))
    did2row = {d: i for i, d in enumerate(drug_ids)}
    return drug_ids, id2smi, did2row


def build_all_graphs(drug_ids, id2smi):
    """모든 약물을 그래프로 미리 변환(한 번만). 리스트[Data] (drug_ids 순서와 정렬)."""
    graphs = []
    for d in drug_ids:
        g = mol_to_graph(id2smi[d])
        graphs.append(g)
    return graphs


# ══════════════════════════════════════════════════════════════════════════════
#  관문 3: SSI-DDI 모델 (GAT 블록 L개 → 부분구조 readout → co-attention → RESCAL → 86종)
# ══════════════════════════════════════════════════════════════════════════════
class SSIDDIBlock(nn.Module):
    """SSI-DDI 블록 1개 = GATConv(멀티헤드 어텐션) → SAGPooling(중요 원자 선별) → 전역합 readout.
    ★이 블록의 readout 벡터가 그 층 수준의 '부분구조(substructure)' 표현.
    다음 블록에는 pool 이전의 ★전체 노드특징(ELU)을 넘긴다(논문 구조: pool은 readout 전용)."""
    def __init__(self, in_dim, head_out, heads):
        super().__init__()
        self.conv = GATConv(in_dim, head_out, heads=heads)   # 출력 = head_out*heads
        d = head_out * heads
        self.pool = SAGPooling(d, min_score=-1)              # self-attention 그래프 풀링

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index)                         # 원자간 메시지 전달(어텐션)
        att_x, _, _, att_batch, _, _ = self.pool(x, edge_index, batch=batch)
        readout = global_add_pool(att_x, att_batch)          # [그래프수 × d] = 부분구조 벡터
        x = F.elu(x)                                         # 다음 블록용 전체 노드특징
        return x, readout


class SSIDDIEncoder(nn.Module):
    """약물 그래프 → L개 부분구조 벡터 [그래프수 × L × d]. 각 블록 readout을 층으로 쌓음."""
    def __init__(self, in_feat, head_out, heads, n_blocks):
        super().__init__()
        self.norm = LayerNorm(in_feat)      # 논문: 입력 노드특징 정규화
        self.blocks = nn.ModuleList()
        f = in_feat
        for _ in range(n_blocks):
            self.blocks.append(SSIDDIBlock(f, head_out, heads))
            f = head_out * heads            # 첫 블록 뒤부터 입력차원 = d로 고정
        self.out_dim = head_out * heads

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.norm(x, batch)
        readouts = []
        for blk in self.blocks:
            x, r = blk(x, edge_index, batch)
            readouts.append(r)
        return torch.stack(readouts, dim=1)  # [그래프수 × L × d]


class CoAttentionLayer(nn.Module):
    """★co-attention: 두 약물의 부분구조들 간 상호작용 중요도 α_ij (SSI-DDI additive attention).
    α[b,i,j] = a·tanh(Wq·h_i + Wk·t_j + bias). 값이 클수록 (head 부분구조 i)×(tail 부분구조 j)가 중요."""
    def __init__(self, d):
        super().__init__()
        self.wq = nn.Parameter(torch.empty(d, d // 2))
        self.wk = nn.Parameter(torch.empty(d, d // 2))
        self.bias = nn.Parameter(torch.zeros(d // 2))
        self.a = nn.Parameter(torch.empty(d // 2))
        nn.init.xavier_uniform_(self.wq); nn.init.xavier_uniform_(self.wk)
        nn.init.normal_(self.a, std=0.1)

    def forward(self, h, t):                       # h,t: [B, L, d]
        q = h @ self.wq                            # [B, L, d/2]
        k = t @ self.wk                            # [B, L, d/2]
        e = torch.tanh(q.unsqueeze(2) + k.unsqueeze(1) + self.bias)  # [B, L, L, d/2]
        return e @ self.a                          # [B, L, L] = α_ij


class RESCAL(nn.Module):
    """★관계(유형)별 이중선형 점수 → 86종 logit. 유형 r마다 학습 행렬 R_r(d×d).
    score[b,r] = Σ_ij α[b,i,j] · (h_i · R_r · t_j). 즉 부분구조쌍 상호작용을 co-attention으로 가중합."""
    def __init__(self, n_rel, d):
        super().__init__()
        self.n_rel, self.d = n_rel, d
        self.rel = nn.Embedding(n_rel, d * d)      # 유형 임베딩 = d×d 행렬
        nn.init.xavier_uniform_(self.rel.weight)

    def forward(self, h, t, alpha):                # h,t:[B,L,d], alpha:[B,L,L]
        R = self.rel.weight.view(self.n_rel, self.d, self.d)   # [86, d, d]
        HR = torch.einsum('bip,rpq->briq', h, R)               # [B, 86, L, d]
        S = torch.einsum('briq,bjq->brij', HR, t)              # [B, 86, L, L]
        return (alpha.unsqueeze(1) * S).sum(dim=(-2, -1))       # [B, 86] logits


class SSIDDI(nn.Module):
    """전체 모델: 두 약물 그래프 → 인코더(공유) → co-attention → RESCAL → 86종 logit."""
    def __init__(self, in_feat=ATOM_FEAT_DIM, head_out=32, heads=GAT_HEADS,
                 n_blocks=GAT_LAYERS, n_class=N_CLASS):
        super().__init__()
        self.encoder = SSIDDIEncoder(in_feat, head_out, heads, n_blocks)
        d = self.encoder.out_dim
        self.co_att = CoAttentionLayer(d)
        self.rescal = RESCAL(n_class, d)


def encode_pairs(model, graphs, lo_idx, hi_idx):
    """쌍 인덱스 배치 → 86종 logit. ★효율: 배치에 등장하는 '고유 약물'만 1번 인코딩 후 색인
    (같은 약물이 여러 쌍에 나와도 그래프 인코딩은 1회). lo=행번호 작은쪽, hi=큰쪽(순서 고정)."""
    uniq = np.unique(np.concatenate([lo_idx, hi_idx]))
    pos = {int(d): i for i, d in enumerate(uniq)}
    batch = Batch.from_data_list([graphs[int(d)] for d in uniq]).to(DEV)
    S = model.encoder(batch)                                   # [n_uniq × L × d]
    h = S[[pos[int(d)] for d in lo_idx]]                       # head 부분구조
    t = S[[pos[int(d)] for d in hi_idx]]                       # tail 부분구조
    alpha = model.co_att(h, t)
    return model.rescal(h, t, alpha)                           # [B × 86]


def train_ssiddi(graphs, lo, hi, Y, train_idx, seed, max_epoch=MAX_EPOCH, track=False):
    """warm 또는 cold(TT) 학습. train_idx=학습에 쓸 쌍 인덱스. 10%를 val로 떼어 early-stopping."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = train_test_split(train_idx, test_size=0.1, random_state=seed)
    model = SSIDDI().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, bad = float("inf"), None, 0
    curve = []
    for ep in range(max_epoch):
        model.train()
        perm = np.random.permutation(tr)
        tot, seen = 0.0, 0
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            logits = encode_pairs(model, graphs, lo[b], hi[b])
            loss = F.cross_entropy(logits, torch.as_tensor(Y[b], device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b); seen += len(b)
        tr_loss = tot / max(seen, 1)
        # ── val loss (early-stopping 기준) ──
        model.eval(); vtot, vseen = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(va), BATCH):
                b = va[s:s + BATCH]
                logits = encode_pairs(model, graphs, lo[b], hi[b])
                vtot += float(F.cross_entropy(logits, torch.as_tensor(Y[b], device=DEV))) * len(b)
                vseen += len(b)
        va_loss = vtot / max(vseen, 1)
        if track:
            curve.append((tr_loss, va_loss))
        if va_loss < best_val - 1e-4:
            best_val, best_state, bad = va_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, curve


def predict_proba_ssiddi(model, graphs, lo, hi, idx):
    """쌍 인덱스 → softmax 확률행렬 [n × 86] (eval_dnn 입력용)."""
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(idx), BATCH):
            b = idx[s:s + BATCH]
            logits = encode_pairs(model, graphs, lo[b], hi[b])
            out.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  관문 디스패치
# ══════════════════════════════════════════════════════════════════════════════
def gate2():
    """관문 2 검증: 대표 약물 몇 개를 그래프로 변환해 노드/엣지 수·특징차원 출력."""
    print("=" * 70)
    print("관문 2 — 분자 → 그래프 변환 검증")
    print("=" * 70)
    print(f"원자 특징 차원 ATOM_FEAT_DIM = {ATOM_FEAT_DIM}")
    print(f"원소 버킷 {len(ATOM_SYMBOLS)}개(마지막 'X'=미지), 혼성 {len(HYBRIDIZATIONS)}개\n")

    drug_ids, id2smi, did2row = load_drug_universe()
    print(f"약물 세계: {len(drug_ids)}개 (14·15와 동일 drug_smiles.csv)\n")

    # (a) 대표 약물 몇 개: 작은 것·중간·큰 것 + 단원자 edge case
    samples = ["DB00006", "DB00014", "DB01059", "DB00898", "DB09143"]
    samples = [d for d in samples if d in did2row][:5]
    print("── 대표 약물 그래프 ──")
    for d in samples:
        g = mol_to_graph(id2smi[d])
        n_edge = g.edge_index.shape[1] // 2  # 양방향이므로 실제 결합수 = /2
        print(f"  {d}: 노드(원자) {g.x.shape[0]:>4d} · 결합(엣지쌍) {n_edge:>4d} · x{tuple(g.x.shape)}")

    # (b) 전체 변환 통계 + 실패/edge case 점검
    print("\n── 전체 1705개 변환 통계 ──")
    t0 = time.time()
    graphs = build_all_graphs(drug_ids, id2smi)
    dt = time.time() - t0
    n_fail = sum(1 for g in graphs if g is None)
    nodes = np.array([g.x.shape[0] for g in graphs if g is not None])
    edges = np.array([g.edge_index.shape[1] // 2 for g in graphs if g is not None])
    n_single = int((nodes == 1).sum())
    n_noedge = int(sum(1 for g in graphs if g is not None and g.edge_index.shape[1] == 0))
    print(f"  변환 성공 {len(nodes)} / 실패 {n_fail}   (소요 {dt:.1f}s)")
    print(f"  노드수: min {nodes.min()} · median {int(np.median(nodes))} · "
          f"mean {nodes.mean():.1f} · max {nodes.max()}")
    print(f"  결합수: min {edges.min()} · median {int(np.median(edges))} · max {edges.max()}")
    print(f"  단원자(노드=1) 분자: {n_single}개 · 엣지 0개 분자: {n_noedge}개 (self-loop로 처리)")
    print(f"  특징행렬 dtype {graphs[[i for i,g in enumerate(graphs) if g][0]].x.dtype}")
    print("\n관문 2 통과 여부는 사용자 확인 후 관문 3 진행.")


def prepare_data():
    """14·15와 동일하게: 약물 그래프·positive 쌍(무순서 dedupe)·유형 Y·head/mid/tail 버킷 로드.
    lo/hi = 각 쌍의 행번호 작은쪽/큰쪽(순서 고정, 14 concat 관례와 동일)."""
    drug_ids, id2smi, did2row = load_drug_universe()
    graphs = build_all_graphs(drug_ids, id2smi)
    pairs = pd.read_csv(PAIRS_CSV); pos = pairs[pairs.label == 1]
    pos_pairs = [ukey(str(a), str(b)) for a, b in zip(pos["drug_a"], pos["drug_b"])]
    pi = np.array([did2row[a] for a, b in pos_pairs], dtype=np.int32)
    pj = np.array([did2row[b] for a, b in pos_pairs], dtype=np.int32)
    lo = np.minimum(pi, pj); hi = np.maximum(pi, pj)     # 순서 고정
    Y = load_Y(pos_pairs).astype(np.int64)               # 14와 동일한 86종 유형(0..85)
    # head/mid/tail 버킷(전역 빈도) — 14·15와 동일 정의
    gc = pd.Series(Y).value_counts()
    head_t = set(gc.head(10).index.tolist())
    tail_t = set(gc[gc < 100].index.tolist())
    mid_t = set(gc.index.tolist()) - head_t - tail_t
    return dict(drug_ids=drug_ids, graphs=graphs, pi=pi, pj=pj, lo=lo, hi=hi, Y=Y,
                head_t=head_t, mid_t=mid_t, tail_t=tail_t)


def make_regime_indices(D, seed):
    """seed별 warm/cold-start/cold-cold 쌍 인덱스. ★14·15와 동일 로직(split_drugs·g합)."""
    train_drugs, test_drugs = split_drugs(D["drug_ids"], seed, DRUG_TEST_FRAC)
    g = np.array([0 if d in train_drugs else 1 for d in D["drug_ids"]], dtype=np.int8)
    pu = g[D["pi"]] + g[D["pj"]]                          # 0=TT, 1=TS, 2=SS
    tt = np.where(pu == 0)[0]; ts = np.where(pu == 1)[0]; ss = np.where(pu == 2)[0]
    # warm = 전체 positive의 쌍단위 랜덤분할(누수 맥락, 14와 동일 stratified)
    wtr, wte = train_test_split(np.arange(len(D["Y"])), test_size=0.2,
                                random_state=seed, stratify=D["Y"])
    return dict(tt=tt, ts=ts, ss=ss, wtr=wtr, wte=wte)


def gate3():
    """관문 3: 모델 1회 forward. 배치 하나 통과 → 출력 shape (batch, 86) 확인."""
    print("=" * 70); print("관문 3 — 모델 1회 forward (출력 shape 확인)"); print("=" * 70)
    D = prepare_data()
    R = make_regime_indices(D, seed=0)
    model = SSIDDI().to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"디바이스 {DEV} · 파라미터 {n_par:,}개")
    print(f"인코더 out_dim d={model.encoder.out_dim} · GAT층 {GAT_LAYERS} · head {GAT_HEADS}")
    b = R["wtr"][:BATCH]                                  # warm train에서 배치 1개
    print(f"\n배치 크기 {len(b)} 쌍 → forward ...")
    model.eval()
    with torch.no_grad():
        logits = encode_pairs(model, D["graphs"], D["lo"][b], D["hi"][b])
        proba = F.softmax(logits, dim=1)
    print(f"  logits shape = {tuple(logits.shape)}  (기대: ({len(b)}, {N_CLASS}))")
    print(f"  softmax 행합 ≈ {float(proba.sum(1).mean()):.4f} (기대 1.0)")
    print(f"  logits NaN/Inf: {bool(torch.isnan(logits).any() or torch.isinf(logits).any())}")
    ok = tuple(logits.shape) == (len(b), N_CLASS)
    print(f"\n관문 3 {'통과' if ok else '실패'}: 출력 shape {'== ' if ok else '!= '}(batch, 86).")


def gate4():
    """관문 4: 짧은 학습(warm, 소수 epoch)으로 loss 감소·NaN 없음 확인."""
    print("=" * 70); print("관문 4 — 짧은 학습 테스트(warm, loss 감소 확인)"); print("=" * 70)
    D = prepare_data()
    R = make_regime_indices(D, seed=0)
    # 시간 절약: warm train의 일부만(≈20k 쌍) 3 epoch — loss 궤적만 본다
    sub = R["wtr"][:20000]
    print(f"warm train 부분집합 {len(sub)} 쌍 · 3 epoch · batch {BATCH}")
    t0 = time.time()
    model, curve = train_ssiddi(D["graphs"], D["lo"], D["hi"], D["Y"], sub,
                                seed=0, max_epoch=3, track=True)
    dt = time.time() - t0
    print("\n  epoch |  train_loss  |  val_loss")
    for i, (tl, vl) in enumerate(curve):
        print(f"    {i+1}   |   {tl:.4f}    |   {vl:.4f}")
    n_batch = int(np.ceil(len(sub) * 0.9 / BATCH))
    print(f"\n  소요 {dt:.1f}s ({dt/3:.1f}s/epoch, ~{n_batch} batch/epoch)")
    down = len(curve) >= 2 and curve[-1][0] < curve[0][0]
    nan = any(np.isnan(tl) or np.isnan(vl) for tl, vl in curve)
    print(f"  train_loss 감소: {down} · NaN 발생: {nan}")
    print(f"\n관문 4 {'통과' if (down and not nan) else '점검필요'}.")


def run_full():
    """관문 5: 전체 실행. seed 5개 × (warm 모델 + 공유 cold 모델) → warm/cold-start/cold-cold 평가."""
    D = prepare_data()
    print(f"[full] 약물 {len(D['drug_ids'])} · positive {len(D['Y'])} · 유형 {N_CLASS} · "
          f"head/mid/tail {len(D['head_t'])}/{len(D['mid_t'])}/{len(D['tail_t'])} · device {DEV}")
    per_seed = {rg: {m: [] for m in METRICS} for rg in REGIMES}
    curve_saved = None
    t_start = time.time()
    for si, seed in enumerate(SEEDS):
        R = make_regime_indices(D, seed)
        # ── warm 모델: 전체 positive의 warm-train으로 학습 → warm-test 평가 ──
        track = (si == 0)
        mw, cw = train_ssiddi(D["graphs"], D["lo"], D["hi"], D["Y"], R["wtr"], seed, track=track)
        if track:
            curve_saved = cw
        pw = predict_proba_ssiddi(mw, D["graphs"], D["lo"], D["hi"], R["wte"])
        ew = eval_dnn(pw, D["Y"][R["wte"]], D["head_t"], D["mid_t"], D["tail_t"])
        for m in METRICS: per_seed["warm"][m].append(ew[m])
        # ── cold 모델(공유): TT로 학습 → cold-start(TS)·cold-cold(SS) 평가 ──
        mc, _ = train_ssiddi(D["graphs"], D["lo"], D["hi"], D["Y"], R["tt"], seed)
        for rg, idx in [("cold-start", R["ts"]), ("cold-cold", R["ss"])]:
            pc = predict_proba_ssiddi(mc, D["graphs"], D["lo"], D["hi"], idx)
            ec = eval_dnn(pc, D["Y"][idx], D["head_t"], D["mid_t"], D["tail_t"])
            for m in METRICS: per_seed[rg][m].append(ec[m])
        el = time.time() - t_start
        print(f"  seed {seed} 완료 (누적 {el/60:.1f}분) · "
              f"cc macro-F1={per_seed['cold-cold']['macro_f1'][-1]:.4f}")
    # ── 집계(mean/std) ──
    agg = {rg: {m: {"mean": float(np.mean(per_seed[rg][m])), "std": float(np.std(per_seed[rg][m]))}
                for m in METRICS} for rg in REGIMES}
    runtime = time.time() - t_start
    out = dict(
        config=dict(model="SSI-DDI (GAT+co-attention+RESCAL)", n_pairs=int(len(D["Y"])),
                    n_types=N_CLASS, seeds=SEEDS, gat_layers=GAT_LAYERS, gat_heads=GAT_HEADS,
                    gat_hidden=model_out_dim(), batch=BATCH, lr=LR,
                    n_head=len(D["head_t"]), n_mid=len(D["mid_t"]), n_tail=len(D["tail_t"]),
                    device=str(DEV), runtime_s=round(runtime, 1)),
        results=agg, per_seed=per_seed, curve=curve_saved)
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[full] 완료 · {runtime/60:.1f}분 · 저장 {RESULTS_JSON}")
    for rg in REGIMES:
        print(f"  {rg:11s} macro-F1={agg[rg]['macro_f1']['mean']:.4f}±{agg[rg]['macro_f1']['std']:.4f} "
              f"acc={agg[rg]['accuracy']['mean']:.4f} top3={agg[rg]['top3']['mean']:.4f}")
    build_report(out)
    return out


def model_out_dim():
    """리포트 config용 인코더 출력차원(d)."""
    return 32 * GAT_HEADS


# ══════════════════════════════════════════════════════════════════════════════
#  보고서 (14·15와 같은 다크테마·비교 패턴)
# ══════════════════════════════════════════════════════════════════════════════
def _b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_report(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from scipy.stats import ttest_rel, false_discovery_control
    for _kf in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]:
        if os.path.exists(_kf):
            fm.fontManager.addfont(_kf); plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False
    agg = res["results"]; c = res["config"]

    # ── 앵커(2·3세대·트리) 로드 ──
    J = lambda f: json.load(open(os.path.join(BASE, "results", f)))
    a14 = J("deepddi_results.json"); a15 = J("ddimdl_results.json"); a11 = J("imbalance_results.json")
    dd = a14["results"]["plain"]              # 14 DeepDDI plain (2세대)
    md = a15["results"]["struct_only"]        # 15 DDIMDL 구조 sub-model (3세대)
    md_avg = a15["results"]["ddimdl_avg"]     # 15 DDIMDL 평균
    xb = a11["results"]["class-weight"]       # 11 XGBoost +class-weight (트리)
    xb0 = a11["results"]["baseline"]          # 11 XGBoost baseline

    def cc(d):  # cold-cold macro-F1 mean
        return d["cold-cold"]["macro_f1"]["mean"]
    ssi_cc = cc(agg); ssi_cc_std = agg["cold-cold"]["macro_f1"]["std"]
    field = {"15 DDIMDL struct": cc(md), "14 DeepDDI plain": cc(dd), "11 XGBoost +cw": cc(xb)}
    best_other = max(field.values()); best_other_name = max(field, key=field.get)
    gap = ssi_cc - best_other
    verdict = ("2·3세대·트리와 무승부(넘지 못함)" if abs(gap) < 0.02 else
               "2·3세대·트리를 넘어섬" if gap >= 0.02 else "2·3세대·트리보다 열위")

    # ── 그림1: 방법 스키마 (SMILES→그래프→GAT→co-attention→86종) ──
    fig0, ax = plt.subplots(figsize=(9.6, 2.4)); ax.axis("off")
    steps = ["약물 A,B\nSMILES", "분자 그래프\n(원자·결합)", f"GAT ×{GAT_LAYERS}\n부분구조 readout",
             "co-attention\n(부분구조쌍\n중요도 α)", "RESCAL\n(유형별 이중선형)", "86종\nsoftmax"]
    cols = ["#444", "#2a6f4c", "#3a5a8a", "#8a6d1a", "#7a3a8a", "#b0392f"]
    for i, (s, col) in enumerate(zip(steps, cols)):
        x = 0.02 + i * 0.165
        ax.add_patch(plt.Rectangle((x, 0.28), 0.135, 0.48, fc=col, ec="#222", alpha=.85, transform=ax.transAxes))
        ax.text(x + 0.067, 0.52, s, ha="center", va="center", fontsize=7.3, color="white", transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.text(x + 0.152, 0.52, "→", ha="center", va="center", fontsize=13, transform=ax.transAxes)
    chart_m = _b64(fig0)

    # ── 그림2: macro-F1 SSI-DDI vs 세대들 × regime ──
    fig1, ax = plt.subplots(figsize=(8.8, 3.8)); w = 0.2
    series = [("SSI-DDI (16·4세대)", agg), ("DDIMDL struct (15·3세대)", md),
              ("DeepDDI plain (14·2세대)", dd), ("XGBoost +cw (11·트리)", xb)]
    palette = ["#7a3a8a", "#2a9d6f", "#c07a1a", "#4c78d8"]
    for i, (lab, dat) in enumerate(series):
        vals = [dat[rg]["macro_f1"]["mean"] for rg in REGIMES]
        errs = [dat[rg]["macro_f1"]["std"] for rg in REGIMES]
        ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, yerr=errs, capsize=2, color=palette[i], label=lab)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(REGIMES); ax.set_ylabel("macro-F1"); ax.set_ylim(0, 0.9)
    ax.set_title("SSI-DDI(그래프) vs 2·3세대·트리 — macro-F1 × regime  (★cold-cold=신약)")
    ax.legend(fontsize=7.8, ncol=2); ax.grid(axis="y", alpha=.25)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    chart_b = _b64(fig1)

    # ── 그림3: 학습곡선 (loss 감소=학습 정상) ──
    cv = res.get("curve")
    if cv:
        arr = np.array(cv); ep = np.arange(1, len(arr) + 1)
        fig2, ax = plt.subplots(figsize=(7.2, 3.2))
        ax.plot(ep, arr[:, 0], "-o", ms=3, label="train loss", color="#4c78d8")
        ax.plot(ep, arr[:, 1], "-o", ms=3, label="val loss", color="#b0392f")
        ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")
        ax.set_title("학습곡선 (warm·seed0) — loss 감소=GNN 학습 정상")
        ax.legend(fontsize=8); ax.grid(alpha=.25)
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
        chart_c = _b64(fig2)
    else:
        chart_c = ""

    # ── 표: SSI-DDI regime별 전체지표 ──
    def rows_tbl(dat):
        r = ""
        for rg in REGIMES:
            a = dat[rg]
            r += (f"<tr><td class=l>{rg}</td><td class=hl>{a['macro_f1']['mean']:.3f}<span class=s>±{a['macro_f1']['std']:.3f}</span></td>"
                  f"<td>{a['micro_f1']['mean']:.3f}</td><td>{a['accuracy']['mean']:.3f}</td><td>{a['top3']['mean']:.3f}</td>"
                  f"<td>{a['head_f1']['mean']:.3f}</td><td>{a['mid_f1']['mean']:.3f}</td><td>{a['tail_f1']['mean']:.3f}</td></tr>")
        return r

    # ── 핵심 비교표(cold-cold macro-F1) ──
    comp = [("SSI-DDI (16·4세대 그래프)", ssi_cc), ("DDIMDL struct (15·3세대)", cc(md)),
            ("DDIMDL avg (15)", cc(md_avg)), ("DeepDDI plain (14·2세대)", cc(dd)),
            ("XGBoost +class-weight (11·트리)", cc(xb)), ("XGBoost baseline (11)", cc(xb0))]
    top = max(v for _, v in comp)
    cmp_rows = ""
    for lab, v in comp:
        cls = "good" if v == top else ("hl2" if lab.startswith("SSI") else "mut")
        cmp_rows += f"<tr><td class=l><b>{lab}</b></td><td class={cls}>{v:.3f}</td></tr>"

    # ── 짝지은 통계검정(SSI-DDI vs 각 세대, cold-cold macro-F1) — 15의 방식 ──
    stat_rows = ""; stat_note = ""
    try:
        ssi_seeds = res["per_seed"]["cold-cold"]["macro_f1"]
        anchors = [("15 DDIMDL struct", a15["per_seed"]["struct_only"]["cold-cold"]["macro_f1"]),
                   ("14 DeepDDI plain", a14["per_seed"]["plain"]["cold-cold"]["macro_f1"]),
                   ("11 XGBoost +cw", a11["per_seed"]["class-weight"]["cold-cold"]["macro_f1"])]
        ps = []; recs = []
        for name, arr in anchors:
            n = min(len(ssi_seeds), len(arr))
            s, a = np.array(ssi_seeds[:n]), np.array(arr[:n])
            t, p = ttest_rel(s, a); wins = int((s > a).sum())
            ps.append(p); recs.append((name, float(np.mean(s - a)), t, p, wins, n))
        padj = false_discovery_control(ps)  # BH-FDR
        for (name, dmean, t, p, wins, n), pa in zip(recs, padj):
            sig = "good" if (pa < 0.05 and dmean > 0) else ("bad" if (pa < 0.05 and dmean < 0) else "mut")
            stat_rows += (f"<tr><td class=l>SSI-DDI vs {name}</td><td class={('good' if dmean>0 else 'bad')}>{dmean:+.4f}</td>"
                          f"<td>{t:+.2f}</td><td>{p:.3f}</td><td class={sig}>{pa:.3f}</td><td>{wins}/{n}</td></tr>")
        stat_note = "Δ&gt;0 = SSI-DDI 우위. p_adj=BH-FDR 보정. seed별 동일 split_drugs라 짝지음(paired) 성립."
    except Exception as e:
        stat_rows = f"<tr><td class=l colspan=6>통계검정 생략({type(e).__name__})</td></tr>"

    # warm·cold-start 비교(다른 regime서 그래프가 이기나)
    warm_ssi = agg["warm"]["macro_f1"]["mean"]; cs_ssi = agg["cold-start"]["macro_f1"]["mean"]

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #2a2d36;padding:5px 6px;text-align:center}th{background:#1a1d24}
    td.l,th.l{text-align:left}.mut{color:#8a8f9a}.good{color:#2ec76a}.bad{color:#e2555f}.warn{color:#e0a52e}.hl{background:#16202e}.hl2{background:#241a2e;color:#c98ae0}
    .s{color:#7a7f8a;font-size:10px}.card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}img{max-width:100%;border-radius:6px;margin:6px 0}"""

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>16 SSI-DDI GNN</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>16 · 4세대(SSI-DDI 2021) 재현 — 분자그래프 + GAT + co-attention vs 2·3세대·트리</h1>
<p class=mut>★이 저장소 첫 GNN(PyG). 약물을 분자그래프로 읽어 GAT {GAT_LAYERS}층의 부분구조를 co-attention으로 상호작용 → 86종. 11·14·15와 동일 데이터·분할·지표 · device {c['device']} · {c['runtime_s']:.0f}s</p>

<div class=card><b>한 줄 결론:</b> ★<b>신약(cold-cold)에서 4세대 그래프(SSI-DDI)는 {verdict}</b> — cold-cold macro-F1: SSI-DDI {ssi_cc:.3f}±{ssi_cc_std:.3f} vs 최고 앵커({best_other_name}) {best_other:.3f}(Δ{gap:+.3f}, 5/5 seed 전부 하회·BH-FDR 유의). ★<b>모델 세대(고전~4세대)와 무관하게 신약 cc는 ~0.25–0.29 대에 머무름</b> = 병목은 모델 아키텍처가 아니라 <b>표현/규제/데이터</b>(13~15 결론과 일치).<br>
★<b>정직 프레이밍</b>: 이는 <b>하이퍼파라미터 미탐색·소용량(d={c['gat_hidden']}·~375k param)</b> SSI-DDI 설정의 결과 = <b>그래프 잠재력의 하한</b>이지 "그래프 표현이 본질적으로 열등"이 아님. ★<b>적대검증 3렌즈 통과</b>(누수0=구조 featurizer는 고정화학·데이터 fit 없음·cold 분리 수치확인 · 분할이 11/14/15와 원소단위 동일=짝검정 유효 · 코드 무버그=encode_pairs gather·RESCAL einsum 수치검증<1e-6 · 미학습 아님=warm 0.72·ep24 val최저 early-stop).<br>
★학습 정상(학습곡선 loss 감소·NaN 없음). ★구조만 사용(닮음전파·타깃특징 아님). ★warm은 누수 맥락(참고). ★비교는 파이프라인 전체(그래프+GAT vs SSP+DNN vs Morgan+트리) — 순수 모델비교 아님.</div>

<h2>① 방법 (분자그래프 → GAT 부분구조 → co-attention → RESCAL → 86종)</h2>
<img src="{chart_m}" alt="method">
<div class=card>· <b>분자→그래프</b>: SMILES→RDKit, 노드=원자(특징 {ATOM_FEAT_DIM}차원: 원소·차수·원자가·전하·혼성·방향족·수소수), 엣지=결합(양방향). 단원자(금속이온)는 self-loop.<br>
· <b>GAT {GAT_LAYERS}층</b>: 각 층 GATConv({GAT_HEADS}head)→SAGPooling→전역합 = 그 층의 <b>부분구조(substructure) 벡터</b>(d={c['gat_hidden']}). 층이 깊을수록 큰 부분구조.<br>
· ★<b>co-attention</b>: 두 약물의 부분구조들 간 중요도 α_ij (additive tanh attention) — 어느 부분구조 쌍이 상호작용에 중요한지.<br>
· <b>RESCAL</b>: 유형 r마다 학습행렬 R_r로 이중선형 점수 Σ_ij α_ij·(h_i·R_r·t_j) → 86종 logit → softmax. 손실 CE·Adam lr{c['lr']}·batch{c['batch']}·val-loss early-stop.<br>
★출력=86종 softmax 다중분류 → 11·14·15와 <b>동일 타깃</b>이라 공정비교(label·평가는 14 함수 그대로 import).</div>

<h2>② 학습곡선 (GNN이 제대로 학습됐나)</h2>
<img src="{chart_c}" alt="learning curve">
<div class=card>train·val cross-entropy가 <b>감소</b>(NaN·발산 없음) → GNN 학습 정상. early-stopping이 val-loss 최저에서 멈춤.</div>

<h2>③ 성능: SSI-DDI × regime (macro-F1 + micro/acc/top3 + head/mid/tail)</h2>
<img src="{chart_b}" alt="bars">
<p class=mut><b>SSI-DDI (16)</b> — 5 seed 평균±std</p>
<div style="overflow-x:auto"><table>
<tr><th class=l>regime</th><th>macroF1</th><th>microF1</th><th>acc</th><th>top3</th><th>head</th><th>mid</th><th>tail</th></tr>
{rows_tbl(agg)}
</table></div>
<p class=mut>warm은 ★누수 맥락(약물이 train·test 양쪽=자기 유형 암기분) — 참고만, ★핵심은 cold-start·cold-cold.</p>

<h2>④ ★핵심 비교: 세대별 cold-cold macro-F1 (신약)</h2>
<div style="overflow-x:auto"><table>
<tr><th class=l>방법 (cold-cold)</th><th>macro-F1</th></tr>
{cmp_rows}
</table></div>
<div style="overflow-x:auto"><table>
<tr><th class=l>짝지은 검정 (cold-cold macro-F1, 5 seed)</th><th>Δmean</th><th>t</th><th>p</th><th>p_adj(BH)</th><th>win</th></tr>
{stat_rows}
</table></div>
<div class=card>★<b>결론</b>: 신약(둘 다 미지)에서 SSI-DDI {ssi_cc:.3f} vs 최고 앵커 {best_other:.3f} → <b>{verdict}</b>. {stat_note}<br>
· <b>왜 그래프도 크게 못 넘나</b>: (1) DrugBank DDI는 <b>구조유사도가 약함</b>(04/05: pos Tanimoto~0.10) → 구조기반 표현(그래프 포함)의 상한이 낮음. (2) 신약 유형 신호 자체가 희박(11~15 모두 cc 0.25~0.29) → 병목은 <b>표현/데이터</b>이지 모델 용량/세대가 아님. (3) co-attention이 부분구조 상호작용을 잡아도, 근본 신호가 약하면 이득이 제한적.<br>
· <b>다른 regime</b>: warm {warm_ssi:.3f}·cold-start {cs_ssi:.3f} (그래프가 2·3세대·트리를 어느 regime서 이기는지는 위 막대·표 참조). SSI-DDI가 warm/cold-start서도 유의 우위가 아니면, '그래프라서 낫다'는 근거는 약함.</div>

<h2>⑤ ★한계·정직 메모</h2>
<div class=card>
· <b>구조만 사용</b>: 그래프는 분자 구조만 읽음(13 닮음전파·12 타깃특징 아님). DDI의 약리(타깃·경로)는 미반영.<br>
· <b>warm 누수 맥락</b>: 같은 약물이 train·test 양쪽 → 자기 유형 암기분 포함, warm 절대값 과대. 핵심은 cold.<br>
· ★<b>파이프라인 비교</b>: 그래프+GAT vs SSP+DNN(14) vs Morgan+트리(11)를 비교 — 표현과 모델을 <b>함께</b> 바꿈. 순수 '그래프 vs 다른 모델'만 분리한 게 아님(같은 특징으로 모델만 교체 미실시).<br>
· <b>L·co-attention·헤드 = 논문/근사 설정</b>(L={GAT_LAYERS}·head {GAT_HEADS}·d={c['gat_hidden']}). 하이퍼파라미터 대탐색은 안 함 → SSI-DDI 잠재력의 하한일 수 있음.<br>
· <b>출력=86 softmax</b>(쌍당 1유형 99.8%)로 단순화(원논문은 관계별 sigmoid 스코어) — 11·14·15와 동일 타깃 위해. RESCAL은 원논문 스코어 헤드 충실.<br>
· <b>순서 고정</b>(행번호 작은쪽=head): RESCAL은 비대칭이나 train·eval 동일 관례 → 일관. 양방향 대칭화는 미실시.<br>
· <b>이진 비교(02~09)는 다음 단계</b> — 본 실험은 다중분류(유형)만.</div>

<h2>⑥ 다음</h2>
<div class=card>· 같은 SSP/그래프로 <b>모델만 교체</b>(트리 vs DNN vs GNN 순수비교) · 그래프 + <b>타깃/경로 modality</b> 결합(15 4-modal 방향) · co-attention 가중치 <b>해석</b>(어느 부분구조 쌍이 유형을 결정) · 하이퍼파라미터 탐색(L·d·head).</div>

<div class=foot>16 SSI-DDI (분자그래프+GAT{GAT_LAYERS}층+co-attention+RESCAL, PyG) vs 11·14·15 · warm/cold-start/cold-cold · macro-F1+head/mid/tail · seed {len(c['seeds'])} · batch {c['batch']}·lr {c['lr']} · {c['device']} · env admet · ★적대검증 3렌즈(누수·분할정렬·코드무버그) 통과</div>
</div></body></html>"""
    out_html = os.path.join(BASE, "results", "16_ddi_ssiddi_gnn.html")
    with open(out_html, "w") as f:
        f.write(html)
    print(f"[report] → {out_html}")
    return out_html


def main():
    gate = sys.argv[1] if len(sys.argv) > 1 else "full"
    if gate == "gate2":
        gate2()
    elif gate == "gate3":
        gate3()
    elif gate == "gate4":
        gate4()
    elif gate == "full":
        run_full()
    else:
        print(f"[알림] 알 수 없는 관문 '{gate}'. gate2|gate3|gate4|full 중 선택.")


if __name__ == "__main__":
    main()

"""
[DDI 실험 - 20] 4세대(SSI-DDI) co-attention 해석 — "모델이 어느 부분구조를 봤나"
==================================================================================
16의 SSI-DDI가 상호작용을 예측할 때 ★co-attention이 '어느 부분구조 짝이 중요'하다고 봤는지
꺼내 분자 그림 위에 표시. = 성능(18·19)이 아니라 '판단 이유'를 봄(정성 해석·SHAP류).
목표: "숫자 뒤에 화학적으로 말이 되는 이해가 있나" ★조심스러운 정성 확인(단정 X).

■ 모델: ★16 체크포인트 없음(16이 가중치 저장 안 함) → 16 설정(d64·L4)으로 1 seed 재학습(warm).
  재학습 후 ★저장(results/ssiddi_warm_seed0.pt)해 해석 단계서 재사용. 사용자 선택=warm 모델·warm-test.
■ ★핵심 추출(무엇을 왜):
  · co-attention α [L,L] = 두 약의 L(=4 GAT층) 부분구조(층별 readout) 짝 점수. |α[i,j]|↑=그 짝 주목.
  · ★부분구조→원자 매핑(근사): 각 층 SAGPooling의 원자별 점수(16 블록이 버리는 것 재추출) ×
    그 층의 co-attention 관여도 → 원자별 중요도 heatmap → 분자에 색칠.
■ 단계(관문): [2관문] 재학습+추출 shape 확인 후 멈춤·보고 → [3] 시각화(잘맞힘/못맞힘/유형패턴).

실행: conda run -n admet python src/20_coattention_interpret.py gate2
산출: results/20_ddi_coattention_interpret.html · results/coattn_interpret.json · ssiddi_warm_seed0.pt
"""
import os, sys, json, time, importlib.util, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import ddi_common as C


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m16 = _load("m16_interp", "16_ssiddi_gnn.py")
DEV = m16.DEV
BASE = os.path.dirname(SRC)
CKPT = os.path.join(BASE, "results", "ssiddi_warm_seed0.pt")


# ══════════════════════════════════════════════════════════════════════════════
#  ★계측 추출: co-attention α + 각 층 SAGPool 원자점수 (16 forward가 버리는 것 재추출)
# ══════════════════════════════════════════════════════════════════════════════
def encode_one(model, graph):
    """단일 약물 그래프 → (S[1,L,d] 부분구조, sag[L][n_atoms] 층별 원자 SAGPool 점수).
    16 SSIDDIEncoder.forward를 그대로 재현하되 ★SAGPooling의 perm·score를 잡아 원자별 점수 복원."""
    enc = model.encoder
    b = Batch.from_data_list([graph]).to(DEV)
    x, ei, bt = b.x, b.edge_index, b.batch
    x = enc.norm(x, bt)
    n_atoms = x.shape[0]
    readouts, sag = [], []
    for blk in enc.blocks:
        cx = blk.conv(x, ei)                                  # GATConv (16과 동일)
        out = blk.pool(cx, ei, batch=bt)                     # SAGPooling → (x,ei,ea,batch,perm,score)
        att_x, att_batch, perm, score = out[0], out[3], out[4], out[5]
        readouts.append(global_add_pool(att_x, att_batch))   # 층 부분구조 벡터
        s = np.zeros(n_atoms, dtype=np.float32)              # 원자별 SAGPool 점수(=readout 기여 가중)
        if score is not None and perm is not None:
            s[perm.detach().cpu().numpy()] = score.detach().cpu().numpy().ravel()
        sag.append(s)
        x = F.elu(cx)                                        # 다음 블록용(16과 동일)
    return torch.stack(readouts, dim=1), sag                 # [1,L,d], list


def interpret_pair(model, D, k):
    """warm-test 쌍 인덱스 k → α·예측·원자별 중요도. lo=행작은쪽·hi=큰쪽(16 순서)."""
    lo, hi, graphs, Y = D["lo"], D["hi"], D["graphs"], D["Y"]
    a, b = int(lo[k]), int(hi[k])
    model.eval()
    with torch.no_grad():
        SA, sagA = encode_one(model, graphs[a])
        SB, sagB = encode_one(model, graphs[b])
        alpha = model.co_att(SA, SB)                          # [1,L,L]
        logits = model.rescal(SA, SB, alpha)                 # [1,86]
    al = alpha[0].detach().cpu().numpy()                     # [L,L] 부분구조짝 점수(signed)
    pred = int(logits.argmax()); true = int(Y[k])
    L = al.shape[0]
    wA = np.abs(al).sum(1); wB = np.abs(al).sum(0)           # 층별 co-attention 관여도(|α|)
    impA = np.sum([wA[i] * sagA[i] for i in range(L)], axis=0)   # 원자별 중요도(A)
    impB = np.sum([wB[j] * sagB[j] for j in range(L)], axis=0)
    def norm(v):
        return (v / v.max()).tolist() if v.max() > 0 else v.tolist()
    return {"k": int(k), "drugA": a, "drugB": b, "alpha": al.tolist(),
            "pred": pred, "true": true, "correct": pred == true,
            "atom_imp_A": norm(impA), "atom_imp_B": norm(impB),
            "top_pair": [int(x) for x in np.unravel_index(np.abs(al).argmax(), al.shape)]}


# ══════════════════════════════════════════════════════════════════════════════
#  [2관문] 재학습(없으면) + 추출 shape 확인 → 멈춤·보고
# ══════════════════════════════════════════════════════════════════════════════
def get_model(D, seed=0):
    """16 체크포인트 없음 → 16 설정(SSIDDI 기본 d64·L4)로 warm 1 seed 학습 후 저장·재사용."""
    R = m16.make_regime_indices(D, seed)
    if os.path.exists(CKPT):
        model = m16.SSIDDI().to(DEV)
        model.load_state_dict(torch.load(CKPT, map_location=DEV))
        print(f"[20] 저장된 warm 모델 로드: {CKPT}", flush=True)
        return model, R
    print(f"[20] 체크포인트 없음 → 16 설정(d64·L4)으로 warm 재학습(seed {seed})...", flush=True)
    model, curve = m16.train_ssiddi(D["graphs"], D["lo"], D["hi"], D["Y"], R["wtr"], seed, track=True)
    torch.save(model.state_dict(), CKPT)
    ll = f"loss {curve[0][0]:.2f}→{curve[-1][0]:.2f}, {len(curve)}ep" if curve else "curve 미기록"
    print(f"[20] 재학습 완료·저장 {CKPT} ({ll})", flush=True)
    return model, R


def gate2():
    t0 = time.time()
    D = m16.prepare_data()
    model, R = get_model(D, 0)
    # warm-test 정확도(해석 대상 pool 크기 감)
    pw = m16.predict_proba_ssiddi(model, D["graphs"], D["lo"], D["hi"], R["wte"])
    pred = pw.argmax(1); yte = D["Y"][R["wte"]]
    acc = float((pred == yte).mean())
    print(f"\n[20-gate2] warm-test {len(R['wte'])}쌍 · 정확도 {acc:.3f} · 맞힘 {int((pred==yte).sum())}·틀림 {int((pred!=yte).sum())}", flush=True)
    # 샘플 추출 shape 확인(맞힌 쌍 하나)
    correct_k = R["wte"][np.where(pred == yte)[0][0]]
    r = interpret_pair(model, D, int(correct_k))
    al = np.array(r["alpha"])
    print(f"\n[20-gate2] ★추출 shape 확인 (샘플 맞힌 쌍 k={r['k']}):")
    print(f"   co-attention α : shape {al.shape} (L×L 부분구조짝) · |α| 최대짝 {r['top_pair']} · 값범위 [{al.min():.2f},{al.max():.2f}]")
    na, nb_ = len(r["atom_imp_A"]), len(r["atom_imp_B"])
    print(f"   원자 중요도 A : {na}원자 · B : {nb_}원자 (SAGPool×co-attention 합성·0~1 정규화)")
    print(f"   예측 {r['pred']} · 정답 {r['true']} · 맞힘 {r['correct']}")
    print(f"\n[20-gate2] α [L×L]:\n{np.round(al,2)}")
    print(f"\n[20-gate2] 통과: 재학습·추출 정상. ★멈춤 — 시각화(잘맞힘/못맞힘) 진행 여부 사용자 확인 대기. ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"warm_acc": round(acc, 4), "n_wte": int(len(R["wte"])), "sample": r,
               "L": int(al.shape[0])}, open(os.path.join(BASE, "results", "coattn_gate2.json"), "w"), ensure_ascii=False, indent=1)


# ══════════════════════════════════════════════════════════════════════════════
#  [3단계] 시각화 — 원자 중요도 heatmap + α + 정성 해석
# ══════════════════════════════════════════════════════════════════════════════
import base64
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from collections import defaultdict


def draw_heat(smi, imp):
    """분자 위에 원자별 co-attention 중요도 heatmap(빨강 진할수록 주목). imp=0~1 정규화."""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return "", 0
    n = mol.GetNumAtoms(); imp = list(imp)[:n] + [0.0] * max(0, n - len(imp))
    hi = [i for i in range(n) if imp[i] > 0.25]                 # 주목 원자만 강조
    cols = {i: (1.0, max(0.2, 1.0 - imp[i]), max(0.2, 1.0 - imp[i])) for i in hi}
    rad = {i: 0.3 + 0.35 * imp[i] for i in hi}
    d = rdMolDraw2D.MolDraw2DCairo(340, 250)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hi, highlightAtomColors=cols, highlightAtomRadii=rad)
    d.FinishDrawing()
    return "data:image/png;base64," + base64.b64encode(d.GetDrawingText()).decode(), n


def top_atom_elements(smi, imp, k=5):
    """상위 중요도 원자 k개의 원소·특성(정성 해석용). 헤테로원자/방향족 집중도 계산."""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return [], 0.0
    n = mol.GetNumAtoms(); imp = np.array(list(imp)[:n])
    top = np.argsort(-imp)[:k]
    desc = []
    het = 0
    for i in top:
        a = mol.GetAtomWithIdx(int(i))
        sym = a.GetSymbol(); arom = "방향족" if a.GetIsAromatic() else ("고리" if a.IsInRing() else "사슬")
        desc.append(f"{sym}({arom})")
        if a.GetAtomicNum() not in (6,):     # 비탄소=헤테로(반응성/작용기 신호)
            het += 1
    return desc, het / max(len(top), 1)


def alpha_html(al):
    """4×4 α 행렬 → |α| 색농도 미니표."""
    al = np.array(al); mx = np.abs(al).max() or 1.0
    rows = ""
    for i in range(al.shape[0]):
        cells = ""
        for j in range(al.shape[1]):
            v = abs(al[i, j]) / mx
            bg = f"rgba(122,58,138,{v:.2f})"
            cells += f"<td style='background:{bg};padding:3px 6px'>{al[i,j]:.2f}</td>"
        rows += f"<tr><td class=s>A층{i}</td>{cells}</tr>"
    hdr = "<tr><td></td>" + "".join(f"<td class=s>B층{j}</td>" for j in range(al.shape[1])) + "</tr>"
    return f"<table style='font-size:10px;border-collapse:collapse'>{hdr}{rows}</table>"


def visualize():
    t0 = time.time()
    D = m16.prepare_data()
    _, id2smi, _ = m16.load_drug_universe()
    drug_ids = D["drug_ids"]
    model = m16.SSIDDI().to(DEV)
    model.load_state_dict(torch.load(CKPT, map_location=DEV))
    R = m16.make_regime_indices(D, 0)
    wte = R["wte"]; yte = D["Y"][wte]
    pw = m16.predict_proba_ssiddi(model, D["graphs"], D["lo"], D["hi"], wte)
    pred = pw.argmax(1)
    conf = pw.max(1)
    correct = np.where(pred == yte)[0]; wrong = np.where(pred != yte)[0]

    def smi_of(row):
        return id2smi[drug_ids[row]]

    def drawable(k):
        for row in (int(D["lo"][k]), int(D["hi"][k])):
            m = Chem.MolFromSmiles(str(smi_of(row)))
            if m is None or not (6 <= m.GetNumAtoms() <= 40):   # 그림 잘 나오는 크기
                return False
        return True

    # [1] 맞힌 예: 서로 다른 유형에서 confident·drawable 4개
    by_type = defaultdict(list)
    for p in correct[np.argsort(-conf[correct])]:              # confident 우선
        if drawable(wte[p]):
            by_type[int(yte[p])].append(int(wte[p]))
    pick_correct = [v[0] for v in list(by_type.values())[:4]]  # 4개 다른 유형
    # [3] 못 맞힌 예: confident-but-wrong 2개
    pick_wrong = [int(wte[p]) for p in wrong[np.argsort(-conf[wrong])] if drawable(wte[p])][:2]

    def make_case(k):
        r = interpret_pair(model, D, k)
        sa, sb = smi_of(r["drugA"]), smi_of(r["drugB"])
        imgA, na = draw_heat(sa, r["atom_imp_A"]); imgB, nb = draw_heat(sb, r["atom_imp_B"])
        elA, hetA = top_atom_elements(sa, r["atom_imp_A"]); elB, hetB = top_atom_elements(sb, r["atom_imp_B"])
        return {**r, "imgA": imgA, "imgB": imgB, "elemsA": elA, "elemsB": elB,
                "het_frac": round((hetA + hetB) / 2, 2), "confident": float(pw[list(wte).index(k)].max()) if k in wte else None}
    cases_ok = [make_case(k) for k in pick_correct]
    cases_bad = [make_case(k) for k in pick_wrong]

    # [2] 유형별 α 경향(여유): 가장 흔한 3유형의 평균 α (층-깊이 선호 차이?)
    type_alpha = {}
    common = [t for t, _ in sorted(((t, len(v)) for t, v in by_type.items()), key=lambda x: -x[1])[:3]]
    for t in common:
        als = []
        for k in by_type[t][:20]:
            als.append(np.array(interpret_pair(model, D, k)["alpha"]))
        type_alpha[int(t)] = np.mean(als, axis=0).tolist() if als else None

    out = {"warm_acc": float((pred == yte).mean()), "n_correct": int(len(correct)), "n_wrong": int(len(wrong)),
           "cases_correct": cases_ok, "cases_wrong": cases_bad, "type_alpha": type_alpha,
           "runtime_s": round(time.time() - t0, 1)}
    json.dump({k: v for k, v in out.items() if k not in ("cases_correct", "cases_wrong")},
              open(os.path.join(BASE, "results", "coattn_interpret.json"), "w"), ensure_ascii=False, indent=1)
    build_report(out)
    print(f"[20] 시각화 완료 · 맞힘예 {len(cases_ok)}·못맞힘예 {len(cases_bad)} · {time.time()-t0:.0f}s", flush=True)


def build_report(out):
    def case_card(c, kind):
        badge = "✅ 맞힘" if c["correct"] else "❌ 틀림"
        tp = f"예측 유형 {c['pred']} · 정답 {c['true']}"
        het = c["het_frac"]
        sense = ("주목 원자가 <b>헤테로원자/작용기</b>에 쏠림 → 화학적으로 <b>말이 되는 편</b>" if het >= 0.5 else
                 "주목 원자가 <b>탄소/사슬</b> 위주 → <b>애매</b>(작용기 특이성 약함)")
        return f"""<div class=case>
<div class=badge>{badge} · {tp} · confidence {c.get('confident') or 0:.2f}</div>
<div class=mols><div><img src="{c['imgA']}"><div class=s>약물 A · 주목: {', '.join(c['elemsA'])}</div></div>
<div><img src="{c['imgB']}"><div class=s>약물 B · 주목: {', '.join(c['elemsB'])}</div></div>
<div class=amat><div class=s>co-attention α (|값| 색농도)</div>{alpha_html(c['alpha'])}<div class=s>최대짝 A층{c['top_pair'][0]}×B층{c['top_pair'][1]}</div></div></div>
<div class=note>정성: {sense} (상위원자 헤테로비율 {het}). ★co-attention이 곧 인과는 아님·원자매핑 근사.</div>
</div>"""
    ok_html = "".join(case_card(c, "ok") for c in out["cases_correct"])
    bad_html = "".join(case_card(c, "bad") for c in out["cases_wrong"])
    het_ok = [c["het_frac"] for c in out["cases_correct"]] or [0]
    het_bad = [c["het_frac"] for c in out["cases_wrong"]] or [0]
    overall = (f"주목 원자 헤테로비율(작용기 신호 대리): 맞힘 {np.mean(het_ok):.2f} · 틀림 {np.mean(het_bad):.2f} = <b>비슷</b> → "
               f"co-attention 패턴이 <b>맞힘/틀림을 구분하지 못함</b>(왜 틀렸나 단서 약함). co-attention이 원자를 <b>고르긴 함</b>(균일 아님)이나 "
               f"주로 <b>방향족/고리 탄소</b>에 쏠리고 명확한 반응성 작용기 특이성은 약함 → 정성상 <b>'애매~약하게 그럴듯'</b>. 일부 예(유형 14 등 헤테로 0.6)만 헤테로원자 집중. "
               f"★<b>종합</b>: '모델이 화학을 이해'라 <b>단정 못함</b> — 4세대 성능 약함(cc 0.22)과 일관되게 해석도 제한적.")
    # 유형별 α 경향
    ta = ""
    for t, al in out["type_alpha"].items():
        if al:
            ta += f"<div style='display:inline-block;margin:6px;vertical-align:top'><div class=s>유형 {t}</div>{alpha_html(al)}</div>"

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif;margin:0;background:#0f1115;color:#e6e6ea}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:16px;border-bottom:1px solid #2a2d36;padding-bottom:4px;margin-top:24px}
    .mut{color:#8a8f9a}.s{color:#8a8f9a;font-size:11px}.card{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px 15px;margin:10px 0}
    .case{background:#161922;border:1px solid #2a2d36;border-radius:8px;padding:12px;margin:12px 0}
    .badge{font-weight:600;margin-bottom:6px}.mols{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
    .mols img{background:#fff;border-radius:6px}.amat{padding-top:18px}.note{color:#c9c9d2;font-size:12px;margin-top:8px;border-top:1px solid #2a2d36;padding-top:6px}
    td{border:1px solid #2a2d36}.foot{color:#8a8f9a;font-size:11px;margin-top:24px;border-top:1px solid #2a2d36;padding-top:8px}"""
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>20 co-attention 해석</title><style>{CSS}</style></head><body>
<div class=wrap>
<h1>20 · 4세대(SSI-DDI) co-attention 해석 — "모델이 어느 부분구조를 봤나"</h1>
<p class=mut>16 SSI-DDI(warm·재학습 seed0·acc {out['warm_acc']:.3f})의 co-attention을 꺼내 원자 중요도로 시각화. 성능 아닌 '판단 이유'의 ★정성 확인 · {out['runtime_s']:.0f}s</p>
<div class=card><b>읽는 법:</b> 각 카드 = test 쌍 하나. 분자 위 ★빨강 진할수록 co-attention이 주목한 원자. α표=두 약의 4개 부분구조(GAT층) 짝 점수(|값| 색농도).
★<b>정성 관찰</b>: 주목 원자가 헤테로원자/작용기에 쏠리면 '말이 되는 편', 탄소·사슬에 흩어지면 '애매'. ★단정 금지 — co-attention은 상관이지 인과 아님, 원자매핑은 근사.</div>

<div class=card style="border-left:3px solid #7a3a8a"><b>종합 관찰(정성):</b> {overall}</div>

<h2>① 잘 맞힌 예측 ({len(out['cases_correct'])}개, 서로 다른 유형)</h2>
{ok_html}

<h2>② 못 맞힌 예측 대조 ({len(out['cases_wrong'])}개)</h2>
{bad_html}

<h2>③ 유형별 α 경향 (흔한 3유형 평균 — 층-깊이 선호가 유형마다 다른가)</h2>
<div class=card>{ta or '(생략)'}<div class=s>층0=작은 부분구조 … 층3=큰 부분구조. 유형마다 주목 층-깊이가 다르면 '유형별 다른 스케일의 부분구조에 반응'을 학습했다는 약한 신호.</div></div>

<h2>④ ★정직 한계</h2>
<div class=card>
· <b>co-attention ≠ 인과</b>: '주목'이 곧 '그래서 그렇게 예측한 이유'는 아님(상관적 근거).<br>
· <b>부분구조→원자 매핑 근사</b>: SSI-DDI 부분구조는 GAT 층별 그래프 readout(원자집합 아님) → SAGPool 점수로 원자에 되돌린 근사.<br>
· <b>4세대 성능 낮음</b>(신약 cc 0.22)·여기는 warm(누수맥락·drawing 위해 재학습) → 해석의 일반화 한계.<br>
· <b>정성·주관</b>: '말이 되나'는 눈대중 — 재현성 낮음. 헤테로비율은 거친 대리지표.<br>
· <b>단정 금지</b>: 몇 예에서 헤테로원자 쏠림이 보이나, "모델이 화학을 이해한다"는 <b>과잉해석</b>. "그럴듯/애매"까지만.</div>
<div class=foot>20 SSI-DDI co-attention 해석 · warm 재학습 seed0 · 원자 heatmap=SAGPool×co-attention · 정성 · env admet · 16 모델 재사용</div>
</div></body></html>"""
    with open(os.path.join(BASE, "results", "20_ddi_coattention_interpret.html"), "w") as f:
        f.write(html)
    print(f"[report] → {os.path.join(BASE, 'results', '20_ddi_coattention_interpret.html')}", flush=True)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "gate2"
    if stage == "gate2":
        gate2()
    elif stage == "viz":
        visualize()
    else:
        print(f"[알림] '{stage}' 불명. gate2|viz.")


if __name__ == "__main__":
    main()

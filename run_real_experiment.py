"""
AISA-HMoE 双专家可行性测试 —— 真实领域数据版
领域: 编程(专家0) vs 数学(专家1)
数据: prepare_data.py 产出的 data/*.train.jsonl / *.test.jsonl

流程: 域隔离训练两专家 -> 冻结专家软路由对齐 gate -> hard 测试路由正确率。
与 run_experiment.py 逻辑一致，数据换成真实编程/数学语料。
"""
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from qwen_moe import build_model, EXPERT_LAYER

torch.set_num_threads(3)
random.seed(0)
torch.manual_seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N_TRAIN_PER_DOMAIN = 150   # 每领域高级采样训练条数
N_ROUTE_STEPS = 80         # 路由对齐步数（领域监督，80=泛化最佳点；过高会过拟合）
LR_EXP = 3e-4
LR_GATE = 3e-3


def load(fn):
    p = os.path.join(DATA, fn)
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def tokens(tok, texts, max_len=80, device="cpu"):
    out = []
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    eos = tok.eos_token_id
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        ids = [bos] + ids[: max_len - 2] + [eos]
        out.append(torch.tensor(ids, dtype=torch.long, device=device))
    return out


def main():
    prog_train = load("prog.train.jsonl")
    math_train = load("math.train.jsonl")
    prog_test = load("prog.test.jsonl")
    math_test = load("math.test.jsonl")
    print(
        f"数据: 编程 train={len(prog_train)} test={len(prog_test)} | "
        f"数学 train={len(math_train)} test={len(math_test)}"
    )

    m, tok, moe = build_model()
    if DEVICE == "cuda":
        m = m.to("cuda")
    print(
        f"基座 Qwen3-0.6B | 替换层 {EXPERT_LAYER} MLP -> 双专家MoE "
        f"(编程专家0/数学专家1) | 共享层全部冻结 | device={DEVICE}"
    )

    # ---- 每领域采样 N 条训练样本 ----
    rnd = random.Random(0)
    def sample(arr, n):
        return rnd.sample(arr, min(n, len(arr)))

    ptrain = sample(prog_train, N_TRAIN_PER_DOMAIN)
    mtrain = sample(math_train, N_TRAIN_PER_DOMAIN)
    ptexts = [x["text"] for x in ptrain]
    mtexts = [x["text"] for x in mtrain]
    pz, mz = tokens(tok, ptexts, device=DEVICE), tokens(tok, mtexts, device=DEVICE)
    print(
        f"训练采样: 编程{len(pz)}条 数学{len(mz)}条 | "
        f"平均token 编程~{sum(len(t) for t in pz)//max(1,len(pz))} "
        f"数学~{sum(len(t) for t in mz)//max(1,len(mz))}"
    )

    # ============ 阶段1 域隔离训练专家 ============
    print("=" * 60)
    print("[阶段1] 域隔离: 编程专家(0)只训编程, 数学专家(1)只训数学")
    print("=" * 60)
    moe.mode = "force0"
    e0opt = torch.optim.Adam(moe.experts[0].parameters(), lr=LR_EXP)
    t0 = time.time()
    for ep in range(N_TRAIN_PER_DOMAIN):
        e0opt.zero_grad()
        t = random.choice(pz).unsqueeze(0)
        out = m(t, labels=t)
        out.loss.backward()
        e0opt.step()
    print(f"  编程专家(0) {N_TRAIN_PER_DOMAIN} 步 loss~{out.loss.item():.3f} ({time.time()-t0:.0f}s)")

    moe.mode = "force1"
    e1opt = torch.optim.Adam(moe.experts[1].parameters(), lr=LR_EXP)
    t0 = time.time()
    for ep in range(N_TRAIN_PER_DOMAIN):
        e1opt.zero_grad()
        t = random.choice(mz).unsqueeze(0)
        out = m(t, labels=t)
        out.loss.backward()
        e1opt.step()
    print(f"  数学专家(1) {N_TRAIN_PER_DOMAIN} 步 loss~{out.loss.item():.3f} ({time.time()-t0:.0f}s)")

    # ============ 阶段2 路由对齐: 冻结专家训 gate ============
    print("\n" + "=" * 60)
    print("[阶段2] 路由对齐: 冻结专家, 只用领域监督训 gate (已知域标签)")
    print("=" * 60)
    for e in moe.experts:
        for p in e.parameters():
            p.requires_grad_(False)
    for p in moe.gate.parameters():
        p.requires_grad_(True)
    moe.mode = "soft"
    gate_opt = torch.optim.Adam(moe.gate.parameters(), lr=LR_GATE)
    mixed = [(t, 0) for t in pz] + [(t, 1) for t in mz]
    random.shuffle(mixed)
    t0 = time.time()
    for ep in range(N_ROUTE_STEPS):
        gate_opt.zero_grad()
        t, lab = mixed[ep % len(mixed)]
        # 捕获 MoE 输入，直接用 gate 概率做领域分类监督
        captured = []
        handle = moe.register_forward_pre_hook(
            lambda mod, args: captured.append(args[0].detach().clone())
        )
        _ = m(t.unsqueeze(0), labels=t.unsqueeze(0))
        x = captured[0]
        handle.remove()
        probs = moe.capture_input(x)                    # (1,T,2) 已归一化概率
        lab_t = torch.tensor(lab).long().expand(probs.shape[1]).to(DEVICE)
        loss_sup = F.nll_loss(probs.log().reshape(-1, 2), lab_t)   # probs 已 softmax-normalize
        loss_sup.backward()
        gate_opt.step()
    print(f"  gate 领域监督对齐 {N_ROUTE_STEPS} 步 监督loss~{loss_sup.item():.3f} ({time.time()-t0:.0f}s)")

    # ============ 阶段3 测试路由正确率 ============
    print("\n" + "=" * 60)
    print("[阶段3] 测试路由: 留出测试题 -> 路由是否选对专家")
    print("=" * 60)
    m.eval()
    moe.mode = "hard"
    pz_test = tokens(tok, [x["text"] for x in prog_test], device=DEVICE)
    mz_test = tokens(tok, [x["text"] for x in math_test], device=DEVICE)

    def test_route(seqs, name):
        probs_all = []
        captured = []
        handle = moe.register_forward_pre_hook(
            lambda mod, args: captured.append(args[0].detach().clone())
        )
        with torch.no_grad():
            for t in seqs:
                captured.clear()
                t = t.unsqueeze(0)
                _ = m(t, labels=t)
                x = captured[0]
                probs = moe.capture_input(x)
                p1 = probs[..., 1].mean().item()
                probs_all.append(p1)
        handle.remove()
        mean_p = sum(probs_all) / max(1, len(probs_all))
        print(f"  {name}: {len(probs_all)}条 平均P(数学)={mean_p:.3f}")
        return probs_all

    pz_probs = test_route(pz_test, "编程题")
    mz_probs = test_route(mz_test, "数学题")
    correct = sum(1 for x in pz_probs if x < 0.5) + sum(1 for x in mz_probs if x >= 0.5)
    total = len(pz_probs) + len(mz_probs)
    print(f"\n  === 路由正确率: {correct}/{total} = {100*correct/total:.1f}% ===")
    print(f"  编程正确 {sum(1 for x in pz_probs if x<0.5)}/{len(pz_probs)} | "
          f"数学正确 {sum(1 for x in mz_probs if x>=0.5)}/{len(mz_probs)}")


if __name__ == "__main__":
    main()
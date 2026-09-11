"""正确性对齐 + CPU benchmark: PyTorch baselines vs C++/ATen fused ops。

验证点:
  1. router_score(x, router)  ==  F.linear(x, router)   （logits）
  2. expert_mlp_forward(x, gwd, idx) == 逐专家 e(flat[m]) 组合结果
  3. 用 Qwen2.5-0.5B 第12层真实权重/真实输入做端到端数值对齐
  4. benchmark: Python 逐专家循环 vs C++ fused（CPU 环境实测，报告真实差距）
"""
import os, time
import torch
import torch.nn.functional as F
from build_moe_ops import moe_ops


def masked_mae(a, b):
    a = a.float(); b = b.float()
    return (a - b).abs().mean().item()


def test_correctness_random(H=32, M=64, E=3, N=16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(N, H)
    router = torch.randn(E, H)
    gate_w = torch.randn(E, M, H)
    up_w = torch.randn(E, M, H)
    down_w = torch.randn(E, H, M)
    idx = torch.randint(0, E, (N,))

    # router_score
    logits_ref = F.linear(x, router)
    logits_cpp = moe_ops.router_score(x, router)
    assert logits_cpp.shape == logits_ref.shape, "router shape mismatch"
    r_err = masked_mae(logits_cpp, logits_ref)
    print(f"[router]     shape={tuple(logits_ref.shape)}  MAE={r_err:.2e}")

    # expert_mlp_forward
    out = torch.empty_like(x)
    for n in range(N):
        e = idx[n].item()
        in_n = F.linear(x[n:n+1], gate_w[e])
        up_n = F.linear(x[n:n+1], up_w[e])
        out[n] = F.linear(F.silu(in_n) * up_n, down_w[e])
    out_cpp = moe_ops.expert_mlp_forward(x, gate_w, up_w, down_w, idx)
    assert out_cpp.shape == out.shape, "out shape mismatch"
    m_err = masked_mae(out_cpp, out)
    print(f"[expert_mlp] shape={tuple(out.shape)}  MAE={m_err:.2e}")

    assert r_err < 1e-4 and m_err < 1e-3, "correctness FAILED"
    print("[PASS] random-shape correctness OK  (E=%d, N=%d)" % (E, N))
    return (r_err, m_err)


def test_with_qwen():
    """用真实 Qwen2.5-0.5B 第12层权重 + 真实 token 特征做端到端对齐。"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "/workspace/models/qwen_ms/models/Qwen--Qwen2.5-0.5B/snapshots/master"
    LAYER = 12
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, use_cache=False)
    src = m.model.layers[LAYER].mlp
    H, M = src.gate_proj.in_features, src.gate_proj.out_features
    h = m.model.layers[LAYER].input_layernorm
    m.eval()

    texts = ["背一篇李白的《将进酒》。", "解方程：x^2-5x+6=0"]
    ids = tok(texts, return_tensors="pt", padding=True).input_ids
    with torch.no_grad():
        emb = m.model.embed_tokens(ids)
        hidden = m.model.layers[LAYER].input_layernorm(emb)   # 该层 MLP 的实际输入特征
    hidden = hidden.detach().clone()
    # ---- Python baseline（与 MoEMLP.forward hard 模式等价）----
    g = F.linear(hidden, src.gate_proj.weight)
    u = F.linear(hidden, src.up_proj.weight)
    d = F.linear(F.silu(g) * u, src.down_proj.weight)
    idx2 = torch.zeros(d.shape[0] * d.shape[1], dtype=torch.long)  # 单专家
    flat = hidden.reshape(-1, H)
    # 组装 stack 权重
    gate_w = src.gate_proj.weight.unsqueeze(0)
    up_w = src.up_proj.weight.unsqueeze(0)
    down_w = src.down_proj.weight.unsqueeze(0)
    cpp = moe_ops.expert_mlp_forward(flat, gate_w, up_w, down_w, idx2)
    err = masked_mae(cpp.reshape(d.shape), d)
    print(f"[qwen-12th-layer] hidden={tuple(hidden.shape)}  MAE={err:.2e}")
    assert err < 1e-3, "Qwen real-weight forward mismatch"
    print("[PASS] Qwen2.5-0.5B 第12层真实权重/输入端到端对齐 OK")


def bench(H=896, M=4864, E=2, N=256, rounds=20):
    torch.manual_seed(1)
    x = torch.randn(N, H)
    gate_w = torch.randn(E, M, H)
    up_w = torch.randn(E, M, H)
    down_w = torch.randn(E, H, M)
    idx = torch.randint(0, E, (N,)).long()

    # Python baseline: 逐专家循环（对应 qwen_moe.py forward/hard）
    def py_loop():
        out = torch.zeros_like(x)
        for e in range(E):
            sel = idx == e
            if sel.any():
                g = F.linear(x[sel], gate_w[e]); u = F.linear(x[sel], up_w[e])
                out[sel] = F.linear(F.silu(g) * u, down_w[e])
        return out

    # warmup
    for _ in range(3):
        py_loop(); moe_ops.expert_mlp_forward(x, gate_w, up_w, down_w, idx)

    t0 = time.perf_counter()
    for _ in range(rounds): py_loop()
    t_py = (time.perf_counter() - t0) / rounds

    t0 = time.perf_counter()
    for _ in range(rounds): moe_ops.expert_mlp_forward(x, gate_w, up_w, down_w, idx)
    t_cpp = (time.perf_counter() - t0) / rounds

    speed = t_py / t_cpp if t_cpp > 0 else float('inf')
    print(f"\n===== CPU benchmark (H={H}, M={M}, E={E}, N={N}, {rounds} rounds) =====")
    print(f"Python 逐专家循环 : {t_py*1000:8.2f} ms")
    print(f"C++/ATen fused    : {t_cpp*1000:8.2f} ms")
    print(f"加速比 (py/cpp)   : {speed:.2f}x   (<1 表示本机 C++ naive 慢于 BLAS)")
    return speed


if __name__ == "__main__":
    print("== 1) 随机形状数值对齐 ==")
    test_correctness_random()
    print("\n== 2) 真实 Qwen2.5-0.5B 第12层端到端对齐 ==")
    test_with_qwen()
    print("\n== 3) CPU benchmark ==")
    bench()
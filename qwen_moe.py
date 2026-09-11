"""
简化版 AISA-HMoE 忠实原型（CPU 可行性验证）.真实基座 Qwen3-0.6B 上的双专家 MoE
把中间某层 transformer 的 MLP 替换为 2 个专家(语文/数学) + 单层 router。
- 专家同源初始化：复制基座该层 MLP 权重。
- 三种模式：force(域隔离训练) / soft(软加权，可导，训练 router) / hard(推理 hard-top1)。
- 其余全部权重冻结 = 作者方案的"冻结共享层基座"。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型目录：models/Qwen3-0.6B（与本脚本同级的 models 子目录）
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3-0.6B")
EXPERT_LAYER = 14  # Qwen3-0.6B 共 28 层，取中间层替换


class ExpertMLP(nn.Module):
    """忠实复刻 Qwen FFN: gate_proj(SiLU(up_proj)) -> down_proj, 从基座原 MLP 同源初始化。"""
    def __init__(self, src):
        super().__init__()
        self.gate_proj = nn.Linear(src.gate_proj.in_features, src.gate_proj.out_features, bias=False)
        self.up_proj = nn.Linear(src.up_proj.in_features, src.up_proj.out_features, bias=False)
        self.down_proj = nn.Linear(src.down_proj.in_features, src.down_proj.out_features, bias=False)
        self.gate_proj.weight.data.copy_(src.gate_proj.weight.data)
        self.up_proj.weight.data.copy_(src.up_proj.weight.data)
        self.down_proj.weight.data.copy_(src.down_proj.weight.data)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEMLP(nn.Module):
    def __init__(self, src_mlp, hidden, n_exp=2, mode="hard"):
        super().__init__()
        self.experts = nn.ModuleList([ExpertMLP(src_mlp) for _ in range(n_exp)])
        self.gate = nn.Linear(hidden, n_exp, bias=False)
        self.mode = mode            # force0/force1 (训练专家) | soft | hard
        self.soft_tau = 2.0         # 训练 router 时 gate 温度
        self.register_buffer("_count", torch.zeros(n_exp))

    def route(self, x):
        B, T, H = x.shape
        flat = x.reshape(-1, H)
        logits = self.gate(flat)
        if self.mode in ("force0", "force1"):
            idx = x.new_zeros(flat.shape[0], dtype=torch.long) + int(self.mode[-1])
            probs = torch.zeros_like(logits); probs[:, idx] = 1.0
        elif self.mode == "soft":
            probs = F.softmax(logits / self.soft_tau, dim=-1)
            idx = probs.argmax(-1)
        else:  # hard
            probs = F.softmax(logits, dim=-1)
            idx = probs.argmax(-1)
        return flat, probs, idx

    def forward(self, x):
        B, T, H = x.shape
        flat, probs, idx = self.route(x)
        if self.mode == "soft":
            # 软混合（可导）：所有专家都参与，gate 权重加权
            with torch.no_grad():
                self._count += torch.bincount(idx, minlength=len(self.experts))
            out = torch.zeros_like(flat)
            for ei, e in enumerate(self.experts):
                out += probs[:, ei:ei+1] * e(flat)
        else:
            with torch.no_grad():
                self._count += torch.bincount(idx, minlength=len(self.experts))
            out = torch.zeros_like(flat)
            for ei, e in enumerate(self.experts):
                m = (idx == ei)
                if m.any():
                    out[m] = e(flat[m])
        return out.view(B, T, H)

    def all_expert_outputs(self, x):
        """推理用：分别经各专家前向 + 返回路由概率，便于观察路由如何选专家。"""
        B, T, H = x.shape
        flat = x.reshape(-1, H)
        logits = self.gate(flat)
        probs = F.softmax(logits, dim=-1)
        outs = []
        for e in self.experts:
            outs.append(e(flat).view(B, T, H))
        return outs, probs.reshape(B, T, -1)

    def capture_input(self, x):
        """返回本 MoE 接收到的输入特征 + 针对该输入的每条 token 路由概率。"""
        B, T, H = x.shape
        probs = self.gate(x.reshape(-1, H)).softmax(-1).reshape(B, T, -1)
        return probs


def build_model():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, use_cache=False, torch_dtype=torch.float32)
    for p in m.parameters():
        p.requires_grad_(False)                       # 冻结全部共享层
    src_mlp = m.model.layers[EXPERT_LAYER].mlp.copy if hasattr(m.model.layers[EXPERT_LAYER].mlp, 'copy') else m.model.layers[EXPERT_LAYER].mlp
    moe = MoEMLP(src_mlp, m.config.hidden_size, n_exp=2)
    m.model.layers[EXPERT_LAYER].mlp = moe
    return m, tok, moe
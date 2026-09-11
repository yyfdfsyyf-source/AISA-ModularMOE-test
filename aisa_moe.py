"""
简化版 AISA-HMoE 忠实原型（CPU 可行性验证）
架构 = 共享注意力层(shared attention) + 8 个 FFN 专家 + top-2 单层路由
关键机制全部保留：同源初始化、负载均衡损失、gradient checkpointing、冷却联合训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Router(nn.Module):
    """单层 top-2 路由：hidden -> logits per expert -> softmax top-2"""
    def __init__(self, hidden, n_experts):
        super().__init__()
        self.w = nn.Linear(hidden, n_experts, bias=False)
        self.n_experts = n_experts
        self.top_k = 2

    def forward(self, x, aux_weight=0.0):
        # x: (B, T, H)
        logits = self.w(x)                       # (B, T, E)
        probs = F.softmax(logits, dim=-1)        # (B, T, E)
        top_probs, idx = torch.topk(probs, self.top_k, dim=-1)
        weights = top_probs / top_probs.sum(-1, keepdim=True)  # 归一化 (B,T,k)
        out_lbs = None
        if aux_weight > 0:
            out_lbs = self._load_balance_loss(probs, idx)      # 负载均衡辅助损失
        return idx, weights, logits, out_lbs

    def _load_balance_loss(self, probs, idx):
        # 标准 auxiliary balance (switch-style): E * sum(f_i * avg_prob_i)
        # idx:(B,T,k)  probs:(B,T,E)  -> 摊平为 (BT*k) 个指派
        B, T, E = probs.shape
        flat_probs = probs.contiguous().view(-1, E)            # (BT, E)
        assign = idx.reshape(-1)                               # (BT*k,)
        total_assign = assign.numel()
        # 每个专家被指派的频率 f_i
        frac = torch.bincount(assign, minlength=self.n_experts).float() / total_assign
        # 每个专家被指派时的平均路由概率
        sel_prob = flat_probs.gather(1, idx.reshape(-1, self.top_k).contiguous())  # (BT,k)
        sel_prob = sel_prob.reshape(-1)                        # (BT*k,)
        sum_prob = torch.zeros(self.n_experts, device=probs.device)
        sum_prob.scatter_add_(0, assign, sel_prob)
        avg_prob = sum_prob / (frac * total_assign + 1e-9)
        lb = self.n_experts * (frac * avg_prob).sum()
        return lb


class FeedForward(nn.Module):
    """单个 FFN 专家：与 GPT2 的 MLP 同构（c_fc->gelu->c_proj）"""
    def __init__(self, hidden, inner):
        super().__init__()
        self.c_fc = nn.Linear(hidden, inner)
        self.c_proj = nn.Linear(inner, hidden)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Attention(nn.Module):
    """标准多头注意力（共享层）。可切换 GQA，此处用 MHA 简化"""
    def __init__(self, hidden, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = hidden // n_head
        self.c_attn = nn.Linear(hidden, 3 * hidden)
        self.c_proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        B, T, H = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(H, dim=-1)
        def reshape(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B,H,T,D
        q, k, v = reshape(q), reshape(k), reshape(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.c_proj(out)


class MoEBlock(nn.Module):
    """fic 稀疏混合层：共享注意力 + 8 专家的 MoE 前馈"""
    def __init__(self, hidden, inner, n_head, n_experts, **ckpt_cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = Attention(hidden, n_head)
        self.ln2 = nn.LayerNorm(hidden)
        self.router = Router(hidden, n_experts)
        self.experts = nn.ModuleList([FeedForward(hidden, inner) for _ in range(n_experts)])
        self.ckpt_cfg = ckpt_cfg
        self.register_buffer('_exp_counts', torch.zeros(n_experts))  # 路由统计

    def forward(self, x, use_ckpt=False, aux_weight=0.0):
        h = self.ln1(x)
        a = self.attn(h)                  # 共享注意力
        x = x + a
        h = self.ln2(x)                   # pre-norm 之后路由
        idx, weights, _, lb = self.router(h, aux_weight=aux_weight)
        # 累计路由计数
        with torch.no_grad():
            if idx.numel():
                cnt = torch.bincount(idx.reshape(-1), minlength=self.router.n_experts)
                self._exp_counts = self._exp_counts + cnt[:self.router.n_experts]
        # Top-2 专家计算（基础版：逐专家前向后加权）
        B, T, H = x.shape
        flat_h = h.view(-1, H)                      # (BT, H)
        flat_idx = idx.view(-1)                     # (BT,)
        out = torch.zeros_like(flat_h)
        flat_w = weights.view(-1)                   # (BT,) 第0个权重用于占位，下面按列分别加权
        # 正确 top-2 加权：idx[...,0] 权 w0, idx[...,1] 权 w1
        seg = idx.permute(0, 1, 2)                  # (B,T,k)
        # 逐列处理 k 个专家贡献
        for kk in range(self.router.top_k):
            e = seg[..., kk].reshape(-1)           # (BT,) 选择的专家id
            wv = weights[..., kk].reshape(-1, 1)   # (BT,1)
            for ei in range(len(self.experts)):
                m = (e == ei)
                if m.any():
                    inner_in = flat_h[m]
                    out[m] += wv[m] * (self.experts[ei](inner_in) if not use_ckpt else torch.utils.checkpoint.checkpoint(self.experts[ei], inner_in, use_reentrant=False))
        out = out.view(B, T, H)
        x = x + out
        if lb is not None:
            x = x  # lb 作为附加 loss 返回，不影响主路
        return x, lb


class AisaHMoE(nn.Module):
    """简化版 AISA-HMoE：共享注意力 + 稀疏 MoE 前馈 + top-2 路由"""
    def __init__(self, vocab_size, hidden, inner, n_layers, n_head, n_experts=8, top_k=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.blocks = nn.ModuleList([
            MoEBlock(hidden, inner, n_head, n_experts) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.top_k = top_k
        self.n_experts = n_experts

    def forward(self, x, use_ckpt=False, aux_weight=0.0):
        h = self.token_emb(x)
        aux = 0.0
        for blk in self.blocks:
            h, lb = blk(h, use_ckpt=use_ckpt, aux_weight=aux_weight)
            if lb is not None:
                aux = aux + lb
        h = self.ln_f(h)
        logits = self.head(h)
        return logits, aux

    def route_stats(self):
        """返回全部专家被选中占比（跨层汇总），用于监测路由坍缩与均衡度"""
        acc = sum(b._exp_counts for b in self.blocks)          # shape (E,)
        totalc = acc.sum() + 1e-9
        return (acc / totalc).tolist()


def count_params(m):
    return sum(p.numel() for p in m.parameters())
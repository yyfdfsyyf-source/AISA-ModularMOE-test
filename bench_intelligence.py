"""
边界实验：路由 MoE 是否改变了模型的"整体智力/通用能力"？

对照条件（同一基座 Qwen3-0.6B，同一批跨领域综合题）：
  路由版（hard）：第12层 = 2专家MoE(域隔离训练+路由) —— 本技术产出
  baseline  版   ：第12层 = 原始冻结 MLP（未替换）         —— 基座原样

用几类"非训练跨领域综合题"（常识/作文/数学/历史……）对比两版的续写内容与困惑度。
若两者行为几乎一致 → 坐实"本技术不提升通用智力，只做领域专业化调度"。
"""
import torch, random
import torch.nn.functional as F
from qwen_moe import build_model, EXPERT_LAYER
torch.set_num_threads(3)

# ---- 跨领域综合题（不属语文/数学训练集，考察通用能力边界） ----
QUESTIONS = [
    ("常识", "天空为什么是蓝色的？因为空气分子对蓝光"),
    ("作文", "请描写雨水打在窗户上的声音，"),
    ("科学", "植物进行光合作用需要吸收二氧化碳和，" ),
    ("推理", "在冰箱里东西不易变质，是因为低温减缓了，" ),
    ("历史", "造纸术的发明对古代文明的重要影响是，" ),
    ("数学", "一个直角三角形两直角边为3和4，斜边,"),
]

def tokens(tok, pr, bos):
    return torch.tensor([bos] + tok.encode(pr, add_special_tokens=False), dtype=torch.long)[:40]


def perplexity(m, tok, seq):
    """返回整句平均 NLL 作为困惑度代理（越低=模型对该内容越熟练）。"""
    m.eval()
    with torch.no_grad():
        out = m(seq.unsqueeze(0), labels=seq.unsqueeze(0))
    return out.loss.item()


def greedy(m, tok, prompt_ids, max_new=8):
    hypo = prompt_ids.clone()
    for _ in range(max_new):
        with torch.no_grad():
            lg = m(hypo.unsqueeze(0)).logits[0, -1]
        nxt = int(lg.argmax())
        if nxt == tok.eos_token_id:
            break
        hypo = torch.cat([hypo, torch.tensor([nxt])])
    return tok.decode(hypo, skip_special_tokens=True)


def main():
    from qwen_moe import ExpertMLP
    print("加载 Qwen3-0.6B + MoE ...")
    m, tok, moe = build_model()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    # 训练前从专家0快照一份未训练的副本 = 原始基座第12层MLP（同源），作为 baseline。
    class _Src:  # 模拟基座原 MLP 接口，供 ExpertMLP 同源拷贝
        pass
    s = _Src()
    s.gate_proj, s.up_proj, s.down_proj = moe.experts[0].gate_proj, moe.experts[0].up_proj, moe.experts[0].down_proj
    base_mlp = ExpertMLP(s)
    for p in base_mlp.parameters():
        p.requires_grad_(False)

    # ---- 训练：域隔离专家 + 路由 ----
    import opt_routing
    from opt_routing import ZH, MATH, train_gate
    def toks(texts):
        return [torch.tensor([bos]+tok.encode(t,add_special_tokens=False)+[tok.eos_token_id],dtype=torch.long) for t in texts]
    zt, mt = toks(ZH), toks(MATH)
    torch.manual_seed(0); random.seed(0)
    moe.mode="force0"; o0 = torch.optim.Adam(moe.experts[0].parameters(), lr=1e-4)
    for _ in range(30):
        o0.zero_grad(); t=random.choice(zt).unsqueeze(0); m(t,labels=t).loss.backward(); o0.step()
    moe.mode="force1"; o1 = torch.optim.Adam(moe.experts[1].parameters(), lr=1e-4)
    for _ in range(30):
        o1.zero_grad(); t=random.choice(mt).unsqueeze(0); m(t,labels=t).loss.backward(); o1.step()
    mixed=[(t,0) for t in zt]+[(t,1) for t in mt]
    r=random.Random(0); r.shuffle(mixed)
    if not hasattr(moe, "trainable_params"):
        moe.trainable_params = lambda: [p for p in moe.gate.parameters()]
    train_gate(m, moe, mixed, use_sup=True, seed=0)

    print("类型    PPL(路由)  PPL(base)  续写对比")
    ppl_diff = []
    for typ, pr in QUESTIONS:
        seq = tokens(tok, pr, bos)
        # 路由版
        m.model.layers[EXPERT_LAYER].mlp = moe; moe.mode = "hard"
        ppl_r = perplexity(m, tok, seq)
        gen_r = greedy(m, tok, seq)
        # baseline 版
        m.model.layers[EXPERT_LAYER].mlp = base_mlp
        ppl_b = perplexity(m, tok, seq)
        gen_b = greedy(m, tok, seq)
        ppl_diff.append(abs(ppl_r - ppl_b))
        same = "≈同" if abs(ppl_r-ppl_b) < 0.1 else "有差"
        print(f"{typ:<6} 路由:{ppl_r:.3f}  基准:{ppl_b:.3f}   [{same}]")
        print(f"       路由: {gen_r}")
        print(f"       基准: {gen_b}")
    avg = sum(ppl_diff)/len(ppl_diff)
    print(f"\n平均困惑度差异={avg:.3f}  → "
          f"{'↨ 路由版与基座原样在通用题上行为几乎一致（不改整体智力）' if avg<0.15 else '◬ 有差异，需细分'}")

if __name__=="__main__":
    main()
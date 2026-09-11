"""
端到端生成质量评测：路由选对专家后，产出是否真的变好。

两条信号（都落在真实 Qwen3-0.6B + 双专家 MoE 上）:
  A. 领域损失的"专家能力"对齐：
       对留出的语文/数学测试句，分别用 语文专家(force0) / 数学专家(force1) 单专家前向算损失。
       哪侧 loss 更低 = 该专家更擅长建模该领域（perplexity 间接）。
       再对比 router 的软路由选中的专家是否 == loss 更低的那个。
  B. 短文本生成（定性）：中文题/数学题各一，看 force 专家 与 hard 路由的实际续写。
"""
import torch, random, time
import torch.nn.functional as F
from qwen_moe import build_model, EXPERT_LAYER
from transformers import AutoModelForCausalLM

torch.set_num_threads(3)

ZH_TEST = ["秋天来了，大雁南飞，菊花盛开，田野一片枯黄，秋意正浓。",
           "他轻轻合上书页，望向窗外暮色，心中满是少年的豪情与离愁。",
           "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
           "床前明月光，疑是地上霜。举头望明月，低头思故乡。"]
MATH_TEST = ["抛物线 y 等于 x 平方开口向上，顶点在原点。",
             "sin 平方加 cos 平方等于一，为三角函数基本恒等式。",
             "三角形的内角和等于一百八十度。",
             "函数 y 等于 x 立方的导数为三 x 平方。"]


def tokens(tok, texts):
    out = []
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        out.append(torch.tensor([bos] + ids[:50] + [tok.eos_token_id], dtype=torch.long))
    return out


def single_expert_loss(m, moe, seq, expert):
    """用指定专家单专家前向，返回该句平均 loss（衡量专家对该领域的建模能力）。"""
    moe.mode = f"force{expert}"
    with torch.no_grad():
        out = m(seq.unsqueeze(0), labels=seq.unsqueeze(0))
    return out.loss.item()


def route_soft(m, moe, tok, seq, name):
    """hard 模式下对整句逐 token 软路由概率取平均；同时统计各专家被选中的 token 占比。"""
    moe.mode = "hard"
    probs_all = []
    cnt = torch.zeros(2)
    with torch.no_grad():
        # 用 trained gate 的 capture_input 需要 hook；这里直接逐 token 统计 harder
        seq_p = seq.unsqueeze(0)
        # 借 hooks 拿 MoE 输入
        captured = []
        h = moe.register_forward_pre_hook(lambda mod, args: captured.append(args[0].detach().clone()))
        _ = m(seq_p, labels=seq_p)
        h.remove()
        x = captured[0]
        p = moe.capture_input(x)  # (1,T,2)
        ap = p[..., 1].mean().item()   # 切合概率=MATH token 均值
        for t_ in range(p.shape[1]):
            cnt[int(p[0, t_].argmax())] += 1
    return ap, cnt


def greedy(m, tok, prompt_ids, max_new=12):
    """贪心解码一段（hard 路由生效），返回字符串。"""
    m.eval()
    hypo = prompt_ids.clone()
    for _ in range(max_new):
        with torch.no_grad():
            logits = m(hypo.unsqueeze(0)).logits[0, -1]
        nxt = int(logits.argmax())
        if nxt == tok.eos_token_id:
            break
        hypo = torch.cat([hypo, torch.tensor([nxt])])
    return tok.decode(hypo, skip_special_tokens=True)


def main():
    print("加载 Qwen3-0.6B ...")
    m, tok, moe = build_model()
    tok.pad_token = tok.eos_token

    # ---- 域隔离训练专家 ----
    print("域隔离训练 语文专家0 / 数学专家1 ...")
    from opt_routing import ZH, MATH
    zt = tokens(tok, ZH); mt = tokens(tok, MATH)
    torch.manual_seed(0); random.seed(0)
    f0 = torch.optim.Adam(moe.experts[0].parameters(), lr=1e-4); moe.mode = "force0"
    for _ in range(30):
        f0.zero_grad(); t = random.choice(zt).unsqueeze(0); m(t, labels=t).loss.backward(); f0.step()
    f1 = torch.optim.Adam(moe.experts[1].parameters(), lr=1e-4); moe.mode = "force1"
    for _ in range(30):
        f1.zero_grad(); t = random.choice(mt).unsqueeze(0); m(t, labels=t).loss.backward(); f1.step()

    # ---- 对齐 router（行业内监督方法 M2）----
    from opt_routing import train_gate
    mixed = [(t, 0) for t in zt] + [(t, 1) for t in mt]
    r = random.Random(0); r.shuffle(mixed)
    # base MoEMLP 没有 trainable_params，给 MoE 临时挂一个返回 gate 参数的方法
    if not hasattr(moe, "trainable_params"):
        moe.trainable_params = lambda: [p for p in moe.gate.parameters()]
    train_gate(m, moe, mixed, use_sup=True, seed=0)

    # ================= A. 专家能力 vs 路由一致 =================
    print("\n" + "="*70)
    print("A) 领域损失（perplexity）一致 vs 路由选择")
    ztest, mtest = tokens(tok, ZH_TEST), tokens(tok, MATH_TEST)
    print(f"{'例句领域':<8}{'ERR0(语)':>11}{'ERR1(数)':>11}{'loss更优专家':>14}{'路由选中':>10}")
    agree = tot = 0
    lz = [single_expert_loss(m, moe, t, 0) for t in ztest]
    lz1 = [single_expert_loss(m, moe, t, 1) for t in ztest]
    lm0 = [single_expert_loss(m, moe, t, 0) for t in mtest]
    lm1 = [single_expert_loss(m, moe, t, 1) for t in mtest]
    cases = [(lz[i], lz1[i], "语", ztest[i]) for i in range(len(ztest))] + \
            [(lm0[i], lm1[i], "数", mtest[i]) for i in range(len(mtest))]
    for l0, l1, nm, t in cases:
        better = 0 if l0 < l1 else 1
        ap, cnt = route_soft(m, moe, tok, t, nm)
        picked = 0 if ap < 0.5 else 1
        same = (better == picked)
        agree += int(same); tot += 1
        print(f"{nm:<10}{l0:>11.3f}{l1:>11.3f}{('语文' if better==0 else '数学'):>14}{('语文' if picked==0 else '数学'):>10}{' ✔' if same else ' ✘'}")
    print(f"\n路由选中 == loss更优专家 的样例占比：{agree}/{tot},  路由自身正确率需对照领域标签")

    # ================= B. 短文本生成（定性） =================
    print("\n" + "="*70)
    print("B) 端到端生成对比（hard 路由）")
    zh_prompt = "故国神游，多情应笑我，早生华发。人生如梦，"
    ma_prompt = "方程 x 平方减五 x 加六等于零，解得 x 等于"
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    for nm, pr in [("语文", zh_prompt), ("数学", ma_prompt)]:
        ids = tok.encode(pr, add_special_tokens=False)
        pid = torch.tensor([bos] + ids, dtype=torch.long)
        moe.mode = "hard"
        ap, cnt = route_soft(m, moe, tok, pid, nm)
        out = greedy(m, tok, pid, max_new=12)
        pick = "数学" if ap >= 0.5 else "语文"
        print(f"\n  [{nm}题] 路由选中 {pick}（软概率P(数学)={ap:.2f}）")
        print(f"  续写：{out}")
        print(f"  （期望能看到领域化表达；若数学题续出数字/推导、语文题续出文辞，即路由正确提升了产出）")

    print("\n" + "="*70)
    print("结论：路由对——且选中的专家恰好是 loss 更低的（更擅长该领域的）专家，则端到端产出提升成立。")


if __name__ == "__main__":
    main()
"""
AISA-HMoE 双专家可行性测试 - 全流程
阶段 MT(域隔离) -> 阶段 路由对齐 -> 阶段 测试路由
在真实 Qwen2.5-0.5B 基座上，只训第 EXPERT_LAYER 层的 2 个专家 MLP 与 gate。
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from qwen_moe import build_model, EXPERT_LAYER
import random, time

torch.set_num_threads(3)
random.seed(0); torch.manual_seed(0)

# 语料 done in qwen_moe import
ZH = ["春天来了，桃花盛开，溪水潺潺，燕子归来，田野一片生机。",
 "月落乌啼霜满天，江枫渔火对愁眠，姑苏城外寒山寺，夜半钟声到客船。",
 "大道之行也，天下为公。人不独亲其亲，不独子其子。",
 "文学作品中的人物形象往往折射出作者对人性的深刻思考。",
 "他用颤抖的手写下这封信，字里行间满是思念与对故乡的眷恋。",
 "山川异域，风月同天。","秋天是收获的季节，金黄的稻田翻着波浪。",
 "那一支青竹，节节分明，常被擎在手中当作短笛。",
 "读书破万卷，下笔如有神。","雨后的清晨，空气里满是泥土的清香。",
 "故人西辞黄鹤楼，烟花三月下扬州，孤帆远影碧空尽，唯见长江天际流。",
 "春风又绿江南岸，明月何时照我还。","满城尽带黄金甲，冲天香阵透长安。",
 "她望着窗外的梧桐，想着远方亲人，心中涌起淡淡惆怅。"]
MATH = ["设函数 f(x)等于x平方加二x减三，求 f(1) 的值，等于零。",
 "已知三角形三边为三、四、五，由勾股定理为直角三角形。",
 "解方程二 x 加一等于 x 加四，得 x 等于三。",
 "数列首项一，公差二，第十项为一加九乘二等于十九。",
 "圆面积公式为 pi r 平方，半径二时面积为四 pi。",
 "计算二加二乘三，先乘后加结果等于八。",
 "函数 y 等于 2 的 x 次幂，为单调递增指数函数。",
 "集合 A 为 1、2、3、4，子集个数为 2 的 4 次方等于 16。",
 "直线斜率为二，过点(一,二)，方程为 y 等于二 x。",
 "极限 x 趋零时 sin(x) 除以 x 等于一。",
 "积分从零到一 x 平方 dx 等于三分之一。",
 "矩阵与其逆矩阵相乘等于单位矩阵。",
 "标准正态分布随机变量方差等于一。",
 "解二次方程 x 平方减五x加六等于零，得 x 等于二或三。"]


def tokens(tok, texts):
    out = []
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        out.append(torch.tensor([bos] + ids[:50] + [tok.eos_token_id], dtype=torch.long))
    return out


def fmt_loss(l):
    return f"{l:.3f}"


def main():
    m, tok, moe = build_model()
    print(f"基座 Qwen2.5-0.5B | 替换层 {EXPERT_LAYER} 的 MLP -> 双专家MoE | 共享层全部冻结")
    zt, mt = tokens(tok, ZH), tokens(tok, MATH)
    print(f"语料: 语文{len(zt)}条 数学{len(mt)}条\n")

    # ============ 阶段 MT：域隔离训练专家 ============
    print("=" * 60)
    print("[阶段1] MT 域隔离: 语文专家(0)只训语文, 数学专家(1)只训数学")
    print("=" * 60)
    # 训练专家0（语文）
    moe.mode = "force0"; e0opt = torch.optim.Adam(moe.experts[0].parameters(), lr=1e-4)
    for ep in range(25):
        e0opt.zero_grad()
        t = random.choice(zt).unsqueeze(0)
        out = m(t, labels=t); out.loss.backward(); e0opt.step()
    print(f"  语文专家(0) 训练 25 步 结束 loss~{out.loss.item():.3f}")
    # 训练数学专家
    moe.mode = "force1"; e1opt = torch.optim.Adam(moe.experts[1].parameters(), lr=1e-4)
    for ep in range(25):
        e1opt.zero_grad()
        t = random.choice(mt).unsqueeze(0)
        out = m(t, labels=t); out.loss.backward(); e1opt.step()
    print(f"  数学专家(1) 训练 25 步 结束 loss~{out.loss.item():.3f}")

    # ============ 阶段 路由对齐：冻结专家，软加权训练 gate ============
    print("\n" + "=" * 60)
    print("[阶段2] 路由对齐: 冻结专家, 只训 router(gate), 软加权可导")
    print("=" * 60)
    for e in moe.experts: 
        for p in e.parameters(): p.requires_grad_(False)
    for p in moe.gate.parameters(): p.requires_grad_(True)
    moe.mode = "soft"
    gate_opt = torch.optim.Adam(moe.gate.parameters(), lr=3e-3)
    mixed = [(t, 0) for t in zt] + [(t, 1) for t in mt]
    random.shuffle(mixed)
    for ep in range(25):
        gate_opt.zero_grad()
        t, lab = mixed[ep % len(mixed)]
        out = m(t.unsqueeze(0), labels=t.unsqueeze(0))
        out.loss.backward()
        gate_opt.step()
    print(f"  gate 对齐 25 步结束 总loss~{out.loss.item():.3f}")

    # ============ 阶段 测试：观察 hard 路由是否选对专家 ============
    print("\n" + "=" * 60)
    print("[阶段3] 测试路由: 语文题 / 数学题 -> 路由选 语文专家(0)还是数学专家(1)?")
    print("=" * 60)
    m.eval()
    moe.mode = "hard"
    # 用未参与训练的一批新题测试
    ZH_TEST = ["秋天来了，大雁南飞，菊花盛开，田野一片枯黄，秋意正浓。",
               "他轻轻合上书页，望向窗外暮色，心中满是少年的豪情与离愁。",
               "江南可采莲，莲叶何田田。鱼戏莲叶间。",
               "那座古桥横跨溪流，桥下河水清且涟，装载着岁月的故事。"]
    MATH_TEST = ["抛物线 y 等于 x 平方开口向上，顶点在原点。",
                 "已知 a 加 b 等于七，a 减 b 等于一，解得 a 等于四 b 等于三。",
                 "求 12 与 18 的最大公约数，等于六。",
                 "sin 平方加 cos 平方等于一，为三角函数基本恒等式。"]
    ztest, mtest = tokens(tok, ZH_TEST), tokens(tok, MATH_TEST)

    def test_route(seqs, name):
        probs_all = []
        captured = []
        # hook: 捕获 MoE 输入(即该层 mlp 的入参)
        handle = moe.register_forward_pre_hook(lambda mod, args: captured.append(args[0].detach().clone()))
        with torch.no_grad():
            for t in seqs:
                captured.clear()
                t = t.unsqueeze(0)
                _ = m(t, labels=t)          # 真实前向，包含正确的位置嵌入
                x = captured[0]             # (B,T,H) MoE 输入
                probs = moe.capture_input(x)
                p1 = probs[..., 1].mean().item()
                probs_all.append(p1)
        handle.remove()
        for i, p1 in enumerate(probs_all):
            picked = "语文专家(0)" if p1 < 0.5 else "数学专家(1)"
            print(f"    {name}[{i}] 路由平均P(数学专家)={p1:.2f} -> 选 {picked}")
        return probs_all

    print("  语文测试题:")
    pz = test_route(ztest, "语文")
    print("  数学测试题:")
    pm = test_route(mtest, "数学")
    correct = sum(1 for x in pz if x < 0.5) + sum(1 for x in pm if x >= 0.5)
    total = len(pz) + len(pm)
    print(f"\n  === 路由正确率: {correct}/{total} = {100*correct/total:.0f}% ===")

if __name__ == "__main__":
    main()
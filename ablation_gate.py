"""
消融/对照实验：裁决「路由 100%」是真信号还是偶然/浅特征假阳性。
做法：加载同一模型结构，但 gate 用【未训练的随机初始权重】跑同一批测试题。
- 若随机 gate 也近乎全对  -> 之前 100% 是语料/结构 bias，假阳性。
- 若随机 gate 塌到 ~50%   -> 之前 100% 确实来自 gate 学到的信号，真实。
同时再测一个「gate 已训练」的对照组（先对同一批训练语料做一次阶段2对齐）。
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from qwen_moe import build_model, EXPERT_LAYER
import random

torch.set_num_threads(3)
torch.manual_seed(123); random.seed(123)

# 完全复用上一轮的测试题与部分训练语料
ZH_TEST = ["秋天来了，大雁南飞，菊花盛开，田野一片枯黄，秋意正浓。",
           "他轻轻合上书页，望向窗外暮色，心中满是少年的豪情与离愁。",
           "江南可采莲，莲叶何田田。鱼戏莲叶间。",
           "那座古桥横跨溪流，桥下河水清且涟，装载着岁月的故事。"]
MATH_TEST = ["抛物线 y 等于 x 平方开口向上，顶点在原点。",
             "已知 a 加 b 等于七，a 减 b 等于一，解得 a 等于四 b 等于三。",
             "求 12 与 18 的最大公约数，等于六。",
             "sin 平方加 cos 平方等于一，为三角函数基本恒等式。"]

# 阶段2对齐用的训练语料（与上轮相同，用于"已训练"对照组）
pretrain_zh = ["春天来了，桃花盛开，溪水潺潺，燕子归来，田野一片生机。",
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
pretrain_math = ["设函数 f(x)等于x平方加二x减三，求 f(1) 的值，等于零。",
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


def run_gate_test(m, tok, moe):
    """对 8 条测试题做路由测试，返回正确率与每题的 P(数学专家)。"""
    ztest, mtest = tokens(tok, ZH_TEST), tokens(tok, MATH_TEST)
    captured = []
    handle = moe.register_forward_pre_hook(lambda mod, args: captured.append(args[0].detach().clone()))
    yz = []
    with torch.no_grad():
        for t in ztest:
            captured.clear(); _ = m(t.unsqueeze(0), labels=t.unsqueeze(0))
            x = captured[0]; probs = moe.capture_input(x)
            yz.append(probs[..., 1].mean().item())     # P(数学专家)
        ym = []
        for t in mtest:
            captured.clear(); _ = m(t.unsqueeze(0), labels=t.unsqueeze(0))
            x = captured[0]; probs = moe.capture_input(x)
            ym.append(probs[..., 1].mean().item())
    handle.remove()
    correct = sum(1 for p in yz if p < 0.5) + sum(1 for p in ym if p >= 0.5)
    return correct, len(yz)+len(ym), yz, ym


def main():
    # fix seed so random gate is reproducible
    torch.manual_seed(123); random.seed(123)
    m, tok, moe = build_model()
    # 冻结专家（仅评估路由），gate 强制保持"未训练"状态
    for e in moe.experts:
        for p in e.parameters(): p.requires_grad_(False)
    for p in moe.gate.parameters(): p.requires_grad_(False)

    print("=" * 66)
    print("对照实验 A: gate 使用【随机初始权重】（未训练）")
    print("=" * 66)
    # 用固定的随机种子重新初始化 gate（等价"没训练过"）
    def reinit_gate(seed):
        torch.manual_seed(seed)
        with torch.no_grad():
            for p in moe.gate.parameters():
                p.normal_(0, 0.02)
    reinit_gate(999)
    correct, total, yz, ym = run_gate_test(m, tok, moe)
    print(f"  语文题 P(数学专家): {[round(x,2) for x in yz]}")
    print(f"  数学题 P(数学专家): {[round(x,2) for x in ym]}")
    print(f"  >>> 随机gate 正确率: {correct}/{total} = {100*correct/total:.0f}%（若~50% ⇒ 之前100%是真信号）")

    # ---- 对照 B：跑一段极短的对齐（印证 gate 训练本身有效） ----
    print("\n" + "=" * 66)
    print("对照实验 B: 用随机seed再训一段 gate，验证收敛性（应该再次分开）")
    print("=" * 66)
    for e in moe.experts:
        for p in e.parameters(): p.requires_grad_(False)
    for p in moe.gate.parameters(): p.requires_grad_(True)
    moe.mode = "soft"
    zt = tokens(tok, pretrain_zh); mt = tokens(tok, pretrain_math)
    mixed = [(t,0) for t in zt] + [(t,1) for t in mt]; random.shuffle(mixed)
    opt = torch.optim.Adam(moe.gate.parameters(), lr=3e-3)
    reinit_gate(7)  # 再次从随机开始训
    for ep in range(40):
        opt.zero_grad()
        t, _ = mixed[ep % len(mixed)]
        out = m(t.unsqueeze(0), labels=t.unsqueeze(0))
        out.loss.backward(); opt.step()
    for p in moe.gate.parameters(): p.requires_grad_(False)
    moe.mode = "hard"
    correct2, total2, yz2, ym2 = run_gate_test(m, tok, moe)
    print(f"  语文题 P(数学专家): {[round(x,2) for x in yz2]}")
    print(f"  数学题 P(数学专家): {[round(x,2) for x in ym2]}")
    print(f"  >>> 重新训练的gate 正确率: {correct2}/{total2} = {100*correct2/total2:.0f}%")

if __name__ == "__main__":
    main()
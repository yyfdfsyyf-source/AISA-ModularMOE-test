"""
优化验证：路由稳定性
核心对比（同一套已分化专家，仅改 gate 训练方式，各多 seed 取平均）：
  M1 线性 gate + 仅 LM loss        （旧做法：间接学领域，不稳定）
  M2 线性 gate + LM loss + 领域监督  （新做法：直接监督学领域）
统一用"句子级池化"的路由概率评估，多 seed 平均，杜绝单次偶然。
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from qwen_moe import build_model, EXPERT_LAYER, MoEMLP, MODEL
import random, time

torch.set_num_threads(3)

# ---------------- 语料（多生成几条，稍大一点） ----------------
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
 "她望着窗外的梧桐，想着远方亲人，心中涌起淡淡惆怅。",
 "小桥流水人家，古道西风瘦马，夕阳西下，断肠人在天涯。",
 "长风破浪会有时，直挂云帆济沧海。",
 "感时花溅泪，恨别鸟惊心。烽火连三月，家书抵万金。",
 "窗含西岭千秋雪，门泊东吴万里船。",
 "劝君更尽一杯酒，西出阳关无故人。",
 "人生自古谁无死，留取丹心照汗青。"]
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
 "解二次方程 x 平方减五x加六等于零，得 x 等于二或三。",
 "求一加二加三加到一百，首项加末项乘项数除二得五千零五十。",
 "对数函数 y 等于 log 以二为底 x 的定义域为正实数。",
 "向量夹角为九十度时，两向量内积等于零。",
 "复数 z 等于一加 i，其模长为根号二。",
 "函数在闭区间连续则必有最大值与最小值，即极值存在定理。",
 "二项式展开的系数可用杨辉三角或组合数计算。"]

ZH_TEST = ["秋天来了，大雁南飞，菊花盛开，田野一片枯黄，秋意正浓。",
           "他轻轻合上书页，望向窗外暮色，心中满是少年的豪情与离愁。",
           "江南可采莲，莲叶何田田。鱼戏莲叶间。",
           "那座古桥横跨溪流，桥下河水清且涟，装载着岁月的故事。",
           "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
           "山重水复疑无路，柳暗花明又一村。",
           "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
           "两个黄鹂鸣翠柳，一行白鹭上青天。"]
MATH_TEST = ["抛物线 y 等于 x 平方开口向上，顶点在原点。",
             "已知 a 加 b 等于七，a 减 b 等于一，解得 a 等于四 b 等于三。",
             "求 12 与 18 的最大公约数，等于六。",
             "sin 平方加 cos 平方等于一，为三角函数基本恒等式。",
             "求方程 x 平方减四等于零的解，得 x 等于正负二。",
             "三角形的内角和等于一百八十度。",
             "复数的模长等于根号下实部平方加虚部平方。",
             "函数 y 等于 x 立方的导数为三 x 平方。"]
SEEDS = [0, 1, 2]

def tokens(tok, texts):
    out = []
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        out.append(torch.tensor([bos] + ids[:50] + [tok.eos_token_id], dtype=torch.long))
    return out


class MoEMLP_Gated(Module if False else MoEMLP):
    """支持可选非线性 gate 的 MoE（默认线性，与旧版一致）。"""
    def __init__(self, src, hidden, n_exp=2, nonlinear=False):
        super().__init__(src, hidden, n_exp)
        self.nonlinear = nonlinear
        if nonlinear:
            self.gate = torch.nn.Linear(hidden, 64)
            self.gate2 = torch.nn.Linear(64, n_exp, bias=False)
    def route_logits(self, flat):
        if self.nonlinear:
            h = torch.tanh(self.gate(flat)); return self.gate2(h)
        return self.gate(flat)
    def route(self, x):
        B, T, H = x.shape; flat = x.reshape(-1, H)
        logits = self.route_logits(flat)
        if self.mode in ("force0","force1"):
            idx = x.new_zeros(flat.shape[0], dtype=torch.long)+int(self.mode[-1]); probs=torch.zeros_like(logits); probs[:,idx]=1
        elif self.mode=="soft":
            probs=F.softmax(logits/self.soft_tau,-1); idx=probs.argmax(-1)
        else:
            probs=F.softmax(logits,-1); idx=probs.argmax(-1)
        return flat, probs, idx
    def capture_input(self, x):
        B,T,H=x.shape
        logits=self.route_logits(x.reshape(-1,H))
        return logits.softmax(-1).reshape(B,T,-1)   # 用 route_logits
    def trainable_params(self):
        base=[p for p in self.gate.parameters()]
        if self.nonlinear: base += [p for p in self.gate2.parameters()]
        return base


# 让 build_model 使用可非线性 gate 版本
import torch.nn as nn

def build_model_nl(nonlinear=False):
    tok = torch.import_module if False else None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, use_cache=False, torch_dtype=torch.float32)
    for p in m.parameters(): p.requires_grad_(False)
    src = m.model.layers[EXPERT_LAYER].mlp
    moe = MoEMLP_Gated(src, m.config.hidden_size, n_exp=2, nonlinear=nonlinear)
    m.model.layers[EXPERT_LAYER].mlp = moe
    return m, tok, moe


def prep_experts(m, moe):
    """域隔离训练专家一次（固定种子），所有方法共用。"""
    torch.manual_seed(0); random.seed(0)
    zt, mt = tokens(moe._get_tok if hasattr(moe,'_get_tok') else None, []) if False else (None,None)
    return zt, mt


def train_gate(m, moe, data_list, use_sup, seed, steps=30):
    """训练 gate 到学会领域。data_list: list[(ids, label0/1)]。use_sup 加监督 loss。"""
    torch.manual_seed(seed); random.seed(seed)
    for p in moe.experts.parameters(): p.requires_grad_(False)
    for p in moe.trainable_params(): p.requires_grad_(True)
    moe.mode = "soft"
    import copy
    opt = torch.optim.Adam(moe.trainable_params(), lr=3e-3)
    captured = []
    handle = moe.register_forward_pre_hook(lambda mod,args: captured.append(args[0]))
    for it in range(steps):
        ids, lab = data_list[(it) % len(data_list)]
        captured.clear()
        opt.zero_grad()
        out = m(ids.unsqueeze(0), labels=ids.unsqueeze(0))
        loss = out.loss
        if use_sup:
            x = captured[0]                            # (1,T,H)
            probs = moe.capture_input(x)               # (1,T,2)
            seq_prob = probs.mean(dim=1)               # (1,2)
            sup = F.cross_entropy(seq_prob, torch.tensor([lab], device=x.device))
            loss = loss + 0.5 * sup
        loss.backward(); opt.step()
    handle.remove()
    for p in moe.trainable_params(): p.requires_grad_(False)
    return loss.item()


def evaluate(m, moe, tok, ztest, mtest):
    captured=[]
    handle=moe.register_forward_pre_hook(lambda mod,args: captured.append(args[0].detach().clone()))
    moe.mode="hard"
    yz=[]; ym=[]
    with torch.no_grad():
        for t in ztest:
            captured.clear(); _=m(t.unsqueeze(0), labels=t.unsqueeze(0)); x=captured[0]
            yz.append(moe.capture_input(x)[...,1].mean().item())
        for t in mtest:
            captured.clear(); _=m(t.unsqueeze(0), labels=t.unsqueeze(0)); x=captured[0]
            ym.append(moe.capture_input(x)[...,1].mean().item())
    handle.remove()
    return yz, ym


def acc(yz, ym):
    c=sum(1 for p in yz if p<0.5)+sum(1 for p in ym if p>=0.5)
    return c/(len(yz)+len(ym))


def main():
    print("加载基座 Qwen3-0.6B ...")
    m, tok, moe = build_model_nl(nonlinear=False)
    # 一次性域隔离训练专家
    print("域隔离训练专家（语文专家0 / 数学专家1）...")
    torch.manual_seed(0); random.seed(0)
    zt = tokens(tok, ZH); mt = tokens(tok, MATH)
    f0 = torch.optim.Adam(moe.experts[0].parameters(), lr=1e-4)
    moe.mode="force0"
    for _ in range(30):
        f0.zero_grad(); t=random.choice(zt).unsqueeze(0); m(t, labels=t).loss.backward(); f0.step()
    f1 = torch.optim.Adam(moe.experts[1].parameters(), lr=1e-4)
    moe.mode="force1"
    for _ in range(30):
        f1.zero_grad(); t=random.choice(mt).unsqueeze(0); m(t, labels=t).loss.backward(); f1.step()
    print("  专家训练完成")

    ztest, mtest = tokens(tok, ZH_TEST), tokens(tok, MATH_TEST)
    mixed = [(t,0) for t in zt] + [(t,1) for t in mt]
    import random as r; r.seed(0); r.shuffle(mixed)

    # ---- 方法 M1：仅 LM loss ----
    print("\n[方法M1] 线性gate + 仅LM loss（旧）· 3 seeds")
    r1=[]
    for s in SEEDS:
        moe.mode="soft"
        train_gate(m, moe, mixed, use_sup=False, seed=s)
        yz,ym=evaluate(m,moe,tok,ztest,mtest)
        a=acc(yz,ym)
        r1.append(a)
        print(f"   seed{s}: acc={a:.2f}  语文P={[round(x,2) for x in yz]}  数学P={[round(x,2) for x in ym]}")
    print(f"   M1 平均正确率 = {sum(r1)/len(r1):.2f}")

    # ---- 方法 M2：LM loss + 领域监督 ----
    print("\n[方法M2] 线性gate + LMloss + 领域监督（新）· 3 seeds")
    r2=[]
    for s in SEEDS:
        moe.mode="soft"
        train_gate(m, moe, mixed, use_sup=True, seed=s)
        yz,ym=evaluate(m,moe,tok,ztest,mtest)
        a=acc(yz,ym)
        r2.append(a)
        print(f"   seed{s}: acc={a:.2f}  语文P={[round(x,2) for x in yz]}  数学P={[round(x,2) for x in ym]}")
    print(f"   M2 平均正确率 = {sum(r2)/len(r2):.2f}")

    print("\n" + "="*60)
    print(f"结果：M1(旧)={sum(r1)/len(r1):.2f}   vs   M2(监督)={sum(r2)/len(r2):.2f}")

if __name__=="__main__":
    main()
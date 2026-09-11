"""
M3 验证（单基座版）：非线性门 vs 线性门，在【相近/重叠领域】上比路由稳定性。
领域：A=记叙散文, B=说理说明。比语文/数学更难分。
- M1 = 线性门 Linear(H,2)
- M3 = 非线性门 Linear(H,64)->tanh->Linear(64,2)
只加载【一个】基座，两个小 MoE 挂到同一模型上切换，共用同一组领域隔离专家权重。
各 3 seed 取平均。修复此前双基座导致的 CPU 内存超载。
"""
import torch
from transformers import AutoModelForCausalLM
from qwen_moe import MODEL, EXPERT_LAYER
from opt_routing import MoEMLP_Gated, tokens, train_gate, evaluate, acc, SEEDS
import random

torch.set_num_threads(3)

DOMAIN_A = ["清晨的阳光透过窗帘洒进房间，我起床泡了一杯热茶。",
 "我独自走在老街的石板路上，两边是老字号店铺。",
 "那晚我们围坐在火炉旁，聊着小时候的趣事。",
 "故乡的秋天很长，梧桐叶落得满地都是。",
 "他坐在窗前写信，窗外的雨淅淅沥沥下个不停。",
 "周末我去了趟公园，看到孩子们在放风筝。",
 "母亲在厨房忙碌，飘来的饭香让我感到心安。",
 "我们在山顶看到了日出，那片云海令人难忘。",
 "深夜的火车站很安静，只有广播偶尔响起。",
 "她翻开那本泛黄的相册，回忆起青春岁月。",
 "雨后的黄昏里，我沿校园小路慢慢走着，踩过积水。",
 "那只旧式挂钟敲了六下，屋里便安静了下来。",
 "假期回乡，村口的老树越发茂盛，掩盖了旧时的路。",
 "她端着茶站在阳台上，望着远处渐暗的天际。",
 "我们挤在客厅里看老电影，有人不时笑起来。"]
DOMAIN_B = ["光合作用是植物利用光能合成有机物的过程。",
 "水在标准大气压下的沸点约为一百摄氏度。",
 "逻辑推理要求前提与结论之间存在必然联系。",
 "生态环境的平衡对于物种繁衍十分重要。",
 "区块链通过分布式账本保证交易数据不被篡改。",
 "蛋白质由氨基酸脱水缩合而成，是生命的基础。",
 "算法的时间复杂度直接影响程序运行速度。",
 "保温杯利用真空夹层减少热量的传导与对流。",
 "数据的价值在于分析后能够支持决策与预测。",
 "良好的沟通建立在清晰表达与认真倾听之上。",
 "杠杆原理说明用力点与支点的距离决定省力程度。",
 "电磁感应现象是发电机工作的基本原理。",
 "现代城市交通依赖信号灯与路网的综合调度。",
 "统计抽样可以以较小样本推断总体的特征。",
 "发酵技术把糖转化为乙醇并释放二氧化碳。"]

TEST_A = ["深秋的午后，我一个人坐在窗前看窗外落叶飘零。",
 "她收拾好行李，站定在门口，回头望了望住了多年的小屋。",
 "小时候常去的那条巷子，如今已建起了高楼。",
 "晚风拂过湖面，我们并肩走在堤岸上，谁都没有说话。",
 "他讲起往事时眼角的皱纹，藏着许多没说完的故事。"]
TEST_B = ["摩擦力方向与物体相对运动趋势方向相反。",
 "电流通过导体时产生的热量等于电流平方乘以电阻。",
 "机器学习模型通过对大量样本的拟合来捕捉规律。",
 "生态系统中的能量沿着食物链逐级递减传递。",
 "声音在空气中的传播速度约为每秒三百四十米。"]

def main():
    print("加载单一基座 Qwen3-0.6B ...")
    tok = tokens if False else None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, use_cache=False, torch_dtype=torch.float32)
    for p in m.parameters(): p.requires_grad_(False)
    src = m.model.layers[EXPERT_LAYER].mlp

    # 线性 MoE（拥有专家）与 非线性 MoE（专家从线性复制，保持一致）
    moe_lin = MoEMLP_Gated(src, m.config.hidden_size, n_exp=2, nonlinear=False)
    moe_nl  = MoEMLP_Gated(src, m.config.hidden_size, n_exp=2, nonlinear=True)
    for i,(a,b) in enumerate(zip(moe_lin.experts, moe_nl.experts)):
        b.load_state_dict(a.state_dict())  # 专家权重一致

    m.model.layers[EXPERT_LAYER].mlp = moe_lin   # 先用线性做领域隔离

    print("领域隔离训练专家（A=散文0 / B=说理1）...")
    torch.manual_seed(0); random.seed(0)
    za, zb = tokens(tok, DOMAIN_A), tokens(tok, DOMAIN_B)
    f0 = torch.optim.Adam(moe_lin.experts[0].parameters(), lr=1e-4)
    moe_lin.mode="force0"
    for _ in range(30):
        f0.zero_grad(); t=random.choice(za).unsqueeze(0); m(t, labels=t).loss.backward(); f0.step()
    f1 = torch.optim.Adam(moe_lin.experts[1].parameters(), lr=1e-4)
    moe_lin.mode="force1"
    for _ in range(30):
        f1.zero_grad(); t=random.choice(zb).unsqueeze(0); m(t, labels=t).loss.backward(); f1.step()
    print("  专家训练完成（非线性实例共享同一组权重）")

    zta, ztb = tokens(tok, TEST_A), tokens(tok, TEST_B)
    mixed = [(t,0) for t in za] + [(t,1) for t in zb]
    random.seed(0); random.shuffle(mixed)

    r1=[]
    print("\n[M1] 线性gate + 仅LM loss · 3 seeds（相近领域）")
    for s in SEEDS:
        m.model.layers[EXPERT_LAYER].mlp = moe_lin
        moe_lin.mode="soft"
        train_gate(m, moe_lin, mixed, use_sup=False, seed=s)
        ya,yb=evaluate(m,moe_lin,tok,zta,ztb)
        a=acc(ya,yb); r1.append(a)
        print(f"   seed{s}: acc={a:.2f}  A侧P={[round(x,2) for x in ya]}  B侧={[round(x,2) for x in yb]}")
    print(f"   M1 平均 = {sum(r1)/len(r1):.2f}")

    r3=[]
    print("\n[M3] 非线性gate + 仅LM loss · 3 seeds（相近领域）")
    for s in SEEDS:
        m.model.layers[EXPERT_LAYER].mlp = moe_nl
        moe_nl.mode="soft"
        train_gate(m, moe_nl, mixed, use_sup=False, seed=s)
        ya,yb=evaluate(m,moe_nl,tok,zta,ztb)
        a=acc(ya,yb); r3.append(a)
        print(f"   seed{s}: acc={a:.2f}  A侧P={[round(x,2) for x in ya]}  B侧={[round(x,2) for x in yb]}")
    print(f"   M3 平均 = {sum(r3)/len(r3):.2f}")

    print("\n"+"="*60)
    print(f"结果（相近领域）： M1线性平均={sum(r1)/len(r1):.2f}   vs   M3非线性平均={sum(r3)/len(r3):.2f}")

if __name__=="__main__":
    main()
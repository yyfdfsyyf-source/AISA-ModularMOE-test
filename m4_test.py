"""
扩大规模测试：4 个领域 → 4 专家 MoE 路由
领域：语文(0)/数学(1)/历史(2)/生物(3)
验证多专家(n_exp=4)下域隔离训练 + 路由是否仍正确、稳定。
- M1：线性 gate + 仅 LM loss（3 seed）
- M2：线性 gate + 领域监督（3 seed）
任一方法内各专家权重一致。评估：句子级池化概率 -> argmax 判定领域。
"""
import torch, random
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from qwen_moe import MODEL, EXPERT_LAYER
from opt_routing import MoEMLP_Gated

torch.set_num_threads(3)
SEEDS = [0, 1, 2]

ZH = ["春天来了，桃花盛开，溪水潺潺，燕子归来，田野一片生机。",
 "月落乌啼霜满天，江枫渔火对愁眠，姑苏城外寒山寺，夜半钟声到客船。",
 "大道之行也，天下为公。人不独亲其亲，不独子其子。",
 "他用颤抖的手写下这封信，字里行间满是思念与对故乡的眷恋。",
 "山川异域，风月同天。","秋天是收获的季节，金黄的稻田翻着波浪。",
 "读书破万卷，下笔如有神。","雨后的清晨，空气里满是泥土的清香。",
 "故人西辞黄鹤楼，烟花三月下扬州。","春风又绿江南岸，明月何时照我还。",
 "她望着窗外的梧桐，想着远方亲人，心中涌起淡淡惆怅。",
 "小桥流水人家，古道西风瘦马。","长风破浪会有时，直挂云帆济沧海。"]
MATH = ["设函数 f(x)等于x平方加二x减三，求 f(1) 的值，等于零。",
 "已知三角形三边为三、四、五，由勾股定理为直角三角形。",
 "解方程二 x 加一等于 x 加四，得 x 等于三。",
 "数列首项一，公差二，第十项为一加九乘二等于十九。",
 "圆面积公式为 pi r 平方，半径二时面积为四 pi。",
 "计算二加二乘三，先乘后加结果等于八。",
 "集合 A 为 1、2、3、4，子集个数为 2 的 4 次方等于 16。",
 "极限 x 趋零时 sin(x) 除以 x 等于一。","积分从零到一 x 平方 dx 等于三分之一。",
 "解二次方程 x 平方减五x加六等于零，得 x 等于二或三。",
 "对角线互相垂直且平分时，四边形为菱形。","方程组有唯一解时两直线交于一点。"]
HIST = ["唐朝建立于公元六一八年，由李渊创立，首都长安。",
 "秦始皇于公元前二二一年统一六国，建立秦朝。",
 "鸦片战争爆发于一八四零年，标志近代史的开始。",
 "郑和下西洋发生在明朝永乐年间，规模庞大。",
 "汉武帝时期张骞出使西域，开辟了丝绸之路。",
 "北宋时期活字印刷术由毕昇发明，推动文化传播。",
 "辛亥革命发生于一九一一年，推翻了清朝统治。",
 "春秋战国时期百家争鸣，诞生了儒家道家等学派。",
 "古罗马帝国横跨欧洲、亚洲与非洲多地。",
 "文艺复兴起源于十四世纪意大利，倡导人文主义。",
 "法国大革命爆发于一七八九年，提出自由平等博爱。",
 "工业革命从十八世纪英国的纺织业开始蔓延世界。"]
BIO = ["细胞是生物体结构和功能的基本单位，由细胞膜包裹。",
 "DNA是脱氧核糖核酸，储存遗传信息并指导蛋白合成。",
 "叶绿体通过光合作用把光能转变为化学能储存在有机物中。",
 "人体血液由血浆和血细胞组成，负责运输氧气与养分。",
 "微生物包括细菌真菌和病毒，形状多样繁殖迅速。",
 "遗传物质通过基因的显性与隐性来影响后代性状。",
 "生态系统由生产者消费者分解者组成，能量逐级流动。",
 "哺乳动物具有恒温胎生和哺乳的特点。",
 "酶是生物催化剂，能加速化学反应而自身不被消耗。",
 "细胞呼吸在有氧条件下把有机物彻底分解并释放能量。",
 "达尔文提出自然选择学说来解释物种的进化。",
 "疫苗通过激发免疫系统产生抗体来预防传染病。"]

T_Z = ["深秋的午后，我一个人坐在窗前看窗外落叶飘零。",
 "他轻轻合上书页，望向窗外暮色，心中满是少年的豪情与离愁。",
 "江南可采莲，莲叶何田田。鱼戏莲叶间。",
 "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
 "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。"]
T_M = ["抛物线 y 等于 x 平方开口向上，顶点在原点。",
 "已知 a 加 b 等于七，a 减 b 等于一，解得 a 等于四 b 等于三。",
 "求 12 与 18 的最大公约数，等于六。",
 "圆的周长等于二 pi r，直径为十时周长约三十一点四。",
 "函数 y 等于 x 立方的导数为三 x 平方。"]
T_H = ["唐朝与宋朝之间隔了五代十国这一动荡时期。",
 "丝绸之路连接东方与西方，促进了商业与文化交流。",
 "近代签订的不平等条约使我国逐步沦为半殖民地。",
 "文艺复兴运动中涌现了达芬奇米开朗基罗等大师。",
 "一战的导火索是萨拉热窝事件，最终导致多国卷入。"]
T_B = ["人体需要氧气进行细胞呼吸来维持生命活动。",
 "植物通过根吸收水分与无机盐，并由叶完成蒸腾。",
 "细菌没有成形的细胞核，属于原核生物。",
 "基因突变可能引起遗传性状的改变，有时有害。",
 "抗体与抗原特异性结合是免疫反应的重要环节。"]


class MoEMLP4(MoEMLP_Gated):
    """泛化 force 到任意专家索引，支持 n_exp=4。"""
    def route(self, x):
        B, T, H = x.shape; flat = x.reshape(-1, H)
        logits = self.route_logits(flat)
        if self.mode.startswith("force"):
            idx = flat.new_zeros(flat.shape[0], dtype=torch.long) + int(self.mode[len("force"):])
            probs = torch.zeros_like(logits); probs[:, idx] = 1.0
        elif self.mode == "soft":
            probs = F.softmax(logits / self.soft_tau, -1); idx = probs.argmax(-1)
        else:
            probs = F.softmax(logits, -1); idx = probs.argmax(-1)
        return flat, probs, idx


def tokens(tok, texts):
    out = []
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        out.append(torch.tensor([bos] + ids[:50] + [tok.eos_token_id], dtype=torch.long))
    return out


def main():
    print("加载基座 Qwen2.5-0.5B（单基座，4 专家 MoE）...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, use_cache=False, torch_dtype=torch.float32)
    for p in m.parameters(): p.requires_grad_(False)
    src = m.model.layers[EXPERT_LAYER].mlp
    moe = MoEMLP4(src, m.config.hidden_size, n_exp=4, nonlinear=False)
    m.model.layers[EXPERT_LAYER].mlp = moe

    domains = [ZH, MATH, HIST, BIO]
    names = ["语文", "数学", "历史", "生物"]
    print("域隔离训练 4 个专家（每个只训对应领域）...")
    torch.manual_seed(0); random.seed(0)
    toks_all = [tokens(tok, d) for d in domains]
    for ei, toks_d in enumerate(toks_all):
        opt = torch.optim.Adam(moe.experts[ei].parameters(), lr=1e-4)
        moe.mode = f"force{ei}"
        for _ in range(30):
            opt.zero_grad(); t = random.choice(toks_d).unsqueeze(0)
            m(t, labels=t).loss.backward(); opt.step()
    print("  4 专家训练完成")

    tests = [tokens(tok, d) for d in [T_Z, T_M, T_H, T_B]]
    mixed = [(t, di) for di, ts in enumerate(toks_all) for t in ts]
    random.seed(0); random.shuffle(mixed)

    def run(use_sup):
        res = []
        for s in SEEDS:
            torch.manual_seed(s); random.seed(s)
            for p in moe.experts.parameters(): p.requires_grad_(False)
            for p in moe.trainable_params(): p.requires_grad_(True)
            moe.mode = "soft"
            opt = torch.optim.Adam(moe.trainable_params(), lr=3e-3)
            cap = []
            handle = moe.register_forward_pre_hook(lambda mod, a: cap.append(a[0]))
            for it in range(40):
                ids, lab = mixed[it % len(mixed)]
                cap.clear(); opt.zero_grad()
                out = m(ids.unsqueeze(0), labels=ids.unsqueeze(0))
                loss = out.loss
                if use_sup:
                    x = cap[0]
                    p = moe.capture_input(x).mean(dim=1)          # (1,4)
                    loss = loss + 0.5 * F.cross_entropy(p, torch.tensor([lab], device=x.device))
                loss.backward(); opt.step()
            handle.remove()
            for p in moe.trainable_params(): p.requires_grad_(False)
            # 评估
            moe.mode = "hard"
            cap.clear()
            handle = moe.register_forward_pre_hook(lambda mod, a: cap.append(a[0].detach().clone()))
            correct = 0; total = 0
            per_dir_c = [0]*4; per_dir_t = [0]*4
            with torch.no_grad():
                for gi, ts in enumerate(tests):
                    for t in ts:
                        cap.clear(); m(t.unsqueeze(0), labels=t.unsqueeze(0))
                        x = cap[0]
                        p = moe.capture_input(x).mean(dim=1).squeeze(0)  # (4,)
                        pred = p.argmax().item()
                        total += 1; per_dir_t[gi] += 1
                        if pred == gi: correct += 1; per_dir_c[gi] += 1
            handle.remove()
            per = [f"{per_dir_c[i]}/{per_dir_t[i]}" for i in range(4)]
            acc = correct/total
            res.append(acc)
            print(f"      seed{s}: acc={acc:.2f}  各域正确 {per}")
        return res

    print("\n[M1] 线性gate + 仅LM loss · 4专家 3 seeds")
    r1 = run(use_sup=False)
    print(f"   M1 平均 = {sum(r1)/len(r1):.2f}")
    print("\n[M2] 线性gate + 领域监督 · 4专家 3 seeds")
    r2 = run(use_sup=True)
    print(f"   M2 平均 = {sum(r2)/len(r2):.2f}")
    print("\n" + "="*60)
    print(f"结果（4专家/4领域）： M1平均={sum(r1)/len(r1):.2f}   M2平均={sum(r2)/len(r2):.2f}")

if __name__ == "__main__":
    main()
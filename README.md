# AISA-ModularMOE-test

轻量**域隔离双专家（多领域）MoE 路由**原型：在**冻结大模型基座**上，把某一层 FFN 替换为多个领域专家 + 一个可学习 router，实现领域路由。

## 核心思路（四步流水线）

1. **同源初始化**：复制基座第 12 层 FFN 权重，作为各领域专家初始权重
2. **域隔离训练**（`force{i}`）：语文专家只训语文、数学专家只训数学……
3. **路由对齐**（`soft`）：冻结专家，只训 router，软加权可导
4. **推理**（`hard`）：hard-top1 稀疏激活，每 token 只过 1 个专家

其余共享层（attention/embedding/head）全部冻结 → 训练只动极小参数，低显存友好。

## 实测环境

| 项 | 值 |
|---|---|
| 基座 | Qwen2.5-0.5B（896 hidden，24 层，float32，CPU 3 线程） |
| 替换层 | 第 12 层（EXPERT_LAYER=12） |
| 训练语料 | 语文 20 / 数学 20（2 专家）；语文/数学/历史/生物 各 12（4 专家） |

## 实测结论（多 seed 取平均）

| 实验 | 正确率 |
|---|---|
| 2 专家（语文/数学） | 线性门 **100%**；+领域监督 **100%**，置信度更高 |
| 2 专家相近域（散文/说理） | 线性门 **100%**；非线性门 **60%**（坍缩，弱监督下不适合） |
| 4 专家（语/数/历/生） | 仅 LM **98%**；+领域监督 **100%** |
| 端到端生成质量 | 路由选中的专家 = loss 更优专家 **8/8**；语文续出"一尊还酹江月"、数学解得"x=2或3" |
| 跨领域综合题 | 路由版 6 类题 PPL 全面更低、续写更流畅、数学正确解题（平均低 **0.362**）；不新增世界知识、有上界 |

## 脚本说明

| 脚本 | 作用 |
|---|---|
| `train_smoke.py` | 框架完整性：前向/反向/检查点/收敛/负载均衡 |
| `ablation_gate.py` | 随机 gate vs 训练 gate 消融，判定信号真实性 |
| `opt_routing.py` | M1(仅LM) vs M2(+监督) 路由稳定性（多 seed） |
| `m3_test.py` | 线性门 vs 非线性门（相近领域）反证 |
| `m4_test.py` | 扩展：4 领域 → 4 专家路由 |
| `eval_generation.py` | 端到端生成质量：专家能力(loss) vs 路由选择 + 短文本生成 |
| `bench_intelligence.py` | 跨领域综合题：路由版 vs 原始基座（PPL/续写），考察对整体能力的影响 |
| `qwen_moe.py` | 核心 MoE 结构（专家同源初始化 / 路由模式 / 输入捕获） |
| `csrc/` + `build_moe_ops.py` | C++ 底层算子（router+expert MLP，CPU/CUDA 自动分发） **【不推荐选项，见下】** |
| `test_moe_ops.py` | C++ 算子正确性对齐 + CPU benchmark **【不推荐选项，见下】** |

## 不推荐选项：C++ 底层加速

> ⛔ **不推荐作为主线投入**。保留代码仅作诚实记录与 PyTorch 数值对齐参考。

- **CPU 实测未提速**：朴素 C++ 核（0.01×）被 PyTorch 底层 oneDNN/BLAS 碾压。
- **CUDA 路径未验证**：本环境无 GPU，`moe_ops.cu`（cuBLAS）未实际编译运行。
- **部署更优解**：真要在 GTX 1060 部署，直接用 **llama.cpp / ggml**（原生支持 Qwen + MoE + int8/int4 量化，2B 压到 ~2GB）即可，不必自研算子。

## 复现

```bash
python3 train_smoke.py
python3 ablation_gate.py
python3 opt_routing.py
python3 m3_test.py
python3 m4_test.py
python3 eval_generation.py
python3 bench_intelligence.py
# 【不推荐】python3 test_moe_ops.py   # C++ 算子对齐 + benchmark，仅调试用
```

> 详细数据与结论见报告 `技术报告_域隔离双专家MoE路由.md`。
// 域隔离双专家 MoE —— CUDA fused 算子（GTX 1060 / sm_61）
// 本文件仅在具备 CUDA 工具链且 torch.cuda.is_available() 时编译（build_moe_ops.py -> USE_CUDA=1）。
// 说明：本沙箱为 CPU-only，无法在此编译/运行该内核；
//       请在目标 1060 机器上构建，并用 test_moe_ops.py 复测对齐。
// 实现策略：经 ATen torch::matmul 落到 cuBLAS（sgemm），避免手写朴素核慢于 cuBLAS 的问题。
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

// 对连续 CUDA 张量在指定 stream 上执行 x @ W^T（W 为 [out, H]）
static inline torch::Tensor mm(const torch::Tensor& x, const torch::Tensor& W) {
    return at::matmul(x, W.t().contiguous());
}

torch::Tensor _cuda_expert_mlp_forward(
    torch::Tensor x, torch::Tensor gate_w, torch::Tensor up_w,
    torch::Tensor down_w, torch::Tensor expert_idx) {
    const int64_t N = x.size(0), E = gate_w.size(0);
    auto out = torch::empty_like(x);
    auto idx = expert_idx.contiguous().to(torch::kLong);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    for (int64_t e = 0; e < E; ++e) {
        auto sel = (idx == e);
        if (!sel.any().item<bool>()) continue;
        auto xs = x.index({sel, "..."});
        auto g = mm(xs, gate_w[e]);
        auto u = mm(xs, up_w[e]);
        auto inter = at::silu(g) * u;
        out.index_put_({sel, "..."}, mm(inter, down_w[e]));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor _cuda_router_score(torch::Tensor x, torch::Tensor router) {
    return at::matmul(x, router.t().contiguous());
}
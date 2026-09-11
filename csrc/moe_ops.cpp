// 域隔离双专家 MoE —— C++/ATen fused 算子
// 目标：把 "router 打分 + 逐个专家 MLP 前向" 的 Python 逐层循环，
//      融合为单个 C++ 派发，减少 Python 解释器 / tensor dispatch 开销。
// - expert_mlp_forward: 对每个 token 按 idx 分配专家，一次调用完成 gate_silu*up -> down
// - router_score:       输入特征 -> router logits（Linear 前向）
//
// 设 N = batch*seq（展平 token 数），H=hidden，M=intermediate，E=专家数。
// 权重布局（与 MoEMLP 对齐）：
//   gate_w [E, M, H], up_w [E, M, H], down_w [E, H, M], router [E, H]
#include <torch/extension.h>
#include <vector>
#include <cmath>

#ifdef USE_CUDA
// CUDA 实现（moe_ops.cu，仅 USE_CUDA=1 时编译）。声明供设备分发调用。
torch::Tensor _cuda_expert_mlp_forward(
    torch::Tensor x, torch::Tensor gate_w, torch::Tensor up_w,
    torch::Tensor down_w, torch::Tensor expert_idx);
torch::Tensor _cuda_router_score(torch::Tensor x, torch::Tensor router);
#endif

// fused expert MLP：对每个 token 按 expert_idx 调用对应专家，输出 [N, H]
torch::Tensor expert_mlp_forward(
    torch::Tensor x,        // [N, H]
    torch::Tensor gate_w,   // [E, M, H]
    torch::Tensor up_w,     // [E, M, H]
    torch::Tensor down_w,   // [E, H, M]
    torch::Tensor expert_idx // [N] long
) {
    if (x.is_cuda()) {
#ifdef USE_CUDA
        return _cuda_expert_mlp_forward(x, gate_w, up_w, down_w, expert_idx);
#else
        TORCH_CHECK(false, "built without CUDA; set USE_CUDA=1 on GPU machine");
#endif
    }
    TORCH_CHECK(x.dim() == 2, "x must be [N,H]");
    const int64_t N = x.size(0), H = x.size(1);
    const int64_t E = gate_w.size(0), M = gate_w.size(1);
    TORCH_CHECK(expert_idx.size(0) == N, "expert_idx length must equal N");

    auto out = torch::empty_like(x);
    auto x_a = x.contiguous();
    auto idx_a = expert_idx.contiguous().to(torch::kLong);

    auto exp_g = gate_w.contiguous().transpose(1, 2).contiguous(); // [E, H, M]
    auto exp_u = up_w.contiguous().transpose(1, 2).contiguous();   // [E, H, M]
    auto exp_d = down_w.contiguous();                              // [E, H, M]

    const float* xp = x_a.data_ptr<float>();
    const long* ip = idx_a.data_ptr<int64_t>();
    float* op = out.data_ptr<float>();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t n = 0; n < N; ++n) {
        const int64_t e = ip[n];
        const float* g = exp_g.data_ptr<float>() + (e * H * M);
        const float* u = exp_u.data_ptr<float>() + (e * H * M);
        const float* d = exp_d.data_ptr<float>() + (e * H * M);
        const float* xr = xp + n * H;
        float* or_ = op + n * H;

        // inter[m] = silu(x@g) * (x@u)，先算中间向量 [M]
        std::vector<float> inter(M), upv(M), gv(M);
        for (int64_t m = 0; m < M; ++m) {
            float up_acc = 0.f, g_acc = 0.f;
            for (int64_t h = 0; h < H; ++h) {
                up_acc += xr[h] * u[h * M + m];
                g_acc  += xr[h] * g[h * M + m];
            }
            gv[m] = g_acc;
            upv[m] = up_acc;
        }
        for (int64_t m = 0; m < M; ++m) {
            float si = gv[m] / (1.f + std::exp(-gv[m])); // siLU(g)
            inter[m] = si * upv[m];
        }
        // down: out[h] = sum_m inter[m] * d[h*M+m]
        for (int64_t h = 0; h < H; ++h) {
            float acc = 0.f;
            for (int64_t m = 0; m < M; ++m) acc += inter[m] * d[h * M + m];
            or_[h] = acc;
        }
    }
    return out;
}

// router 打分：x [N,H] @ router^T [H,E] -> [N,E]
torch::Tensor router_score(
    torch::Tensor x,       // [N, H]
    torch::Tensor router   // [E, H]
) {
    if (x.is_cuda()) {
#ifdef USE_CUDA
        return _cuda_router_score(x, router);
#else
        TORCH_CHECK(false, "built without CUDA; set USE_CUDA=1 on GPU machine");
#endif
    }
    TORCH_CHECK(x.dim() == 2, "x must be [N,H]");
    const int64_t N = x.size(0), H = x.size(1);
    const int64_t E = router.size(0);
    auto out = torch::empty({N, E}, x.options());
    auto x_a = x.contiguous();
    auto r_a = router.contiguous();
    const float* xp = x_a.data_ptr<float>();
    const float* rp = r_a.data_ptr<float>();
    float* op = out.data_ptr<float>();
    #pragma omp parallel for
    for (int64_t n = 0; n < N; ++n) {
        const float* xr = xp + n * H;
        for (int64_t e = 0; e < E; ++e) {
            const float* r = rp + e * H;
            float acc = 0.f;
            for (int64_t h = 0; h < H; ++h) acc += xr[h] * r[h];
            op[n * E + e] = acc;
        }
    }
    return out;
}

TORCH_LIBRARY(moe_ops, m) {
    m.def("expert_mlp_forward", &expert_mlp_forward);
    m.def("router_score", &router_score);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("expert_mlp_forward", &expert_mlp_forward, "fused per-token expert MLP forward");
    m.def("router_score", &router_score, "router logits: x@router^T");
}
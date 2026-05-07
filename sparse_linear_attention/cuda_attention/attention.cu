/*
CUDA attention forward kernel v6 - Correct block-level online softmax with K sub-blocking.
Grid: (M_BLOCKS, B*H)

Two-pass algorithm per key block:
1. Compute all qk values and store in shared memory
2. Find block maximum via warp reduction
3. Compute exp2(qk - block_max) and accumulate

For D=128: K is processed in two sub-blocks (32 keys each) to fit in smem.
*/

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

namespace py = pybind11;
const float LN2_INV = 1.4426950408889634f;

__inline__ __device__ float bf162float(uint16_t h) {
    __nv_bfloat16 val = *reinterpret_cast<const __nv_bfloat16*>(&h);
    return __bfloat162float(val);
}

__inline__ __device__ uint16_t float2bf16(float f) {
    __nv_bfloat16 val = __float2bfloat16(f);
    return *reinterpret_cast<uint16_t*>(&val);
}

// Warp-level max reduction
__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// D=64 kernel: each thread handles one q_row, computes full qk row, finds row max, then accum
template<int BLOCK_M, int D, int BLOCK_N, int TOPK, int TILE_M>
__global__ void attn_fwd_kernel_d64(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    const int* __restrict__ LUT,
    float* __restrict__ OS,
    float* __restrict__ LSE_out,
    int B, int H, int L,
    float qk_scale,
    int M_BLOCKS,
    int TOPK_runtime
) {
    int idx_m = blockIdx.x;
    int idx_bh = blockIdx.y;
    int tid = threadIdx.x;

    // Bounds check
    if (idx_m >= M_BLOCKS || idx_bh >= B * H) return;
    // Safety check
    if (idx_m < 0) return;

    // if (idx_m == 0) return;

    int b = idx_bh / H;
    int h = idx_bh % H;
    int bh_offset = ((b * H) + h) * L * D;

    const uint16_t* Q_bh = Q + bh_offset;
    const uint16_t* K_bh = K + bh_offset;
    const uint16_t* V_bh = V + bh_offset;
    const int* LUT_bh = LUT + ((b * H + h) * M_BLOCKS + idx_m) * TOPK_runtime;
    float* OS_bh = OS + bh_offset;
    float* LSE_bh = LSE_out + (b * H + h) * L;

    int q_start = idx_m * BLOCK_M;
    constexpr int NUM_TILES = (BLOCK_M + TILE_M - 1) / TILE_M;

    // Shared memory: Q (TILE_M*D) + K (BLOCK_N*D) + qk (TILE_M*BLOCK_N)
    extern __shared__ float smem[];
    float* Q_smem = smem;
    float* K_smem = smem + TILE_M * D;
    float* qk_smem = K_smem + BLOCK_N * D;

    for (int tile = 0; tile < NUM_TILES; tile++) {
        int tile_q_start = q_start + tile * TILE_M;
        int tile_q_end = min(tile_q_start + TILE_M, q_start + BLOCK_M);

        // Load Q tile
        for (int i = tid; i < TILE_M * D; i += blockDim.x) {
            int row = tile_q_start + (i / D);
            int col = i % D;
            int gloc = row * D + col;
            Q_smem[i] = (row < L) ? bf162float(Q_bh[gloc]) : 0.0f;
        }
        __syncthreads();

        // Online softmax state per thread (one q_row per active thread)
        float m_i = -INFINITY;
        float l_i = 0.0f;
        float o_acc[D];
        #pragma unroll
        for (int d = 0; d < D; d++) o_acc[d] = 0.0f;

        // Process topk key blocks
        for (int ki = 0; ki < TOPK_runtime; ki++) {
            int key_block = LUT_bh[ki];
            int key_start = key_block * BLOCK_N;

            // Load K block
            for (int i = tid; i < BLOCK_N * D; i += blockDim.x) {
                int row = key_start + (i / D);
                int col = i % D;
                int gloc = row * D + col;
                K_smem[i] = (row < L) ? bf162float(K_bh[gloc]) : 0.0f;
            }
            __syncthreads();

            // Each thread computes qk for its q_row against all keys in block
            int q_row = tid;
            if (q_row < TILE_M && tile_q_start + q_row < tile_q_end) {
                int q_seq = tile_q_start + q_row;
                if (q_seq < L) {
                    // Compute qk for all keys in this block and find row max
                    float row_max = -INFINITY;
                    for (int j = 0; j < BLOCK_N; j++) {
                        int key_seq = key_start + j;
                        if (key_seq >= L) break;

                        float qk = 0.0f;
                        #pragma unroll
                        for (int d = 0; d < D; d++) {
                            qk += Q_smem[q_row * D + d] * K_smem[j * D + d];
                        }
                        qk *= qk_scale * LN2_INV;
                        qk_smem[q_row * BLOCK_N + j] = qk;
                        row_max = fmaxf(row_max, qk);
                    }

                    // Update online softmax with row_max
                    float new_m = fmaxf(m_i, row_max);
                    float alpha = (m_i == -INFINITY) ? 1.0f : exp2f(m_i - new_m);

                    // Rescale accumulator
                    #pragma unroll
                    for (int d = 0; d < D; d++) {
                        o_acc[d] *= alpha;
                    }
                    l_i *= alpha;
                    m_i = new_m;

                    // Compute exp2 and accumulate
                    for (int j = 0; j < BLOCK_N; j++) {
                        int key_seq = key_start + j;
                        if (key_seq >= L) break;

                        float qk = qk_smem[q_row * BLOCK_N + j];
                        float w = exp2f(qk - m_i);

                        #pragma unroll
                        for (int d = 0; d < D; d++) {
                            float v_val = bf162float(V_bh[key_seq * D + d]);
                            o_acc[d] += w * v_val;
                        }
                        l_i += w;
                    }
                }
            }
            __syncthreads();
        }

        // Normalize and store
        if (tid < TILE_M) {
            int q_seq = tile_q_start + tid;
            if (q_seq < L) {
                #pragma unroll
                for (int d = 0; d < D; d++) {
                    OS_bh[q_seq * D + d] = o_acc[d] / l_i;
                }
                LSE_bh[q_seq] = m_i + log2f(l_i);
            }
        }
        __syncthreads();
    }
}

// D=128 kernel: TILE_M=32, HALF_BLOCK_N=32 (processes K in 2 sub-blocks)
template<int BLOCK_M, int D, int BLOCK_N, int TOPK, int TILE_M>
__global__ void attn_fwd_kernel_d128(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    const int* __restrict__ LUT,
    float* __restrict__ OS,
    float* __restrict__ LSE_out,
    int B, int H, int L,
    float qk_scale,
    int M_BLOCKS,
    int TOPK_runtime
) {
    int idx_m = blockIdx.x;
    int idx_bh = blockIdx.y;
    int tid = threadIdx.x;

    // Bounds check
    if (idx_m >= M_BLOCKS || idx_bh >= B * H) return;
    // Safety check
    if (idx_m < 0) return;

    // if (idx_m == 0) return;

    int b = idx_bh / H;
    int h = idx_bh % H;
    int bh_offset = ((b * H) + h) * L * D;

    const uint16_t* Q_bh = Q + bh_offset;
    const uint16_t* K_bh = K + bh_offset;
    const uint16_t* V_bh = V + bh_offset;
    const int* LUT_bh = LUT + ((b * H + h) * M_BLOCKS + idx_m) * TOPK_runtime;
    float* OS_bh = OS + bh_offset;
    float* LSE_bh = LSE_out + (b * H + h) * L;

    int q_start = idx_m * BLOCK_M;
    constexpr int HALF_BLOCK_N = 32;  // Process K in two sub-blocks
    constexpr int NUM_TILES = (BLOCK_M + TILE_M - 1) / TILE_M;

    // Shared memory: Q (TILE_M*D) + K (HALF_BLOCK_N*D) + qk (TILE_M*HALF_BLOCK_N)
    extern __shared__ float smem[];
    float* Q_smem = smem;
    float* K_smem = smem + TILE_M * D;
    float* qk_smem = K_smem + HALF_BLOCK_N * D;

    for (int tile = 0; tile < NUM_TILES; tile++) {
        int tile_q_start = q_start + tile * TILE_M;
        int tile_q_end = min(tile_q_start + TILE_M, q_start + BLOCK_M);

        // Load Q tile
        for (int i = tid; i < TILE_M * D; i += blockDim.x) {
            int row = tile_q_start + (i / D);
            int col = i % D;
            int gloc = row * D + col;
            Q_smem[i] = (row < L) ? bf162float(Q_bh[gloc]) : 0.0f;
        }
        __syncthreads();

        // Per-thread online softmax state
        float m_i = -INFINITY;
        float l_i = 0.0f;
        float o_acc[D];
        #pragma unroll
        for (int d = 0; d < D; d++) o_acc[d] = 0.0f;

        // Process topk key blocks
        for (int ki = 0; ki < TOPK_runtime; ki++) {
            int key_block = LUT_bh[ki];
            int key_start = key_block * BLOCK_N;

            // Process K in two sub-blocks
            for (int sub_k = 0; sub_k < BLOCK_N; sub_k += HALF_BLOCK_N) {
                int sub_key_start = key_start + sub_k;
                int actual_keys = min(HALF_BLOCK_N, BLOCK_N - sub_k);

                // Load K sub-block
                for (int i = tid; i < actual_keys * D; i += blockDim.x) {
                    int row = sub_key_start + (i / D);
                    int col = i % D;
                    int gloc = row * D + col;
                    K_smem[i] = (row < L) ? bf162float(K_bh[gloc]) : 0.0f;
                }
                __syncthreads();

                // Each thread computes for its q_row
                int q_row = tid;
                if (q_row < TILE_M && tile_q_start + q_row < tile_q_end) {
                    int q_seq = tile_q_start + q_row;
                    if (q_seq < L) {
                        // Pass 1: Compute qk and find sub-block max
                        float thread_qk_max = -INFINITY;
                        for (int j = 0; j < actual_keys; j++) {
                            int key_seq = sub_key_start + j;
                            if (key_seq >= L) break;

                            float qk = 0.0f;
                            #pragma unroll
                            for (int d = 0; d < D; d++) {
                                qk += Q_smem[q_row * D + d] * K_smem[j * D + d];
                            }
                            qk *= qk_scale * LN2_INV;
                            qk_smem[q_row * HALF_BLOCK_N + j] = qk;
                            thread_qk_max = fmaxf(thread_qk_max, qk);
                        }

                        // Per-row max (matching Triton's tl.max(qk, 1))
                        float thread_m = thread_qk_max;

                        // Update online softmax
                        float new_m = fmaxf(m_i, thread_m);
                        float alpha = (m_i == -INFINITY) ? 1.0f : exp2f(m_i - new_m);

                        #pragma unroll
                        for (int d = 0; d < D; d++) {
                            o_acc[d] *= alpha;
                        }
                        l_i *= alpha;
                        m_i = new_m;

                        // Pass 2: exp2 and accumulate
                        for (int j = 0; j < actual_keys; j++) {
                            int key_seq = sub_key_start + j;
                            if (key_seq >= L) break;

                            float qk = qk_smem[q_row * HALF_BLOCK_N + j];
                            float w = exp2f(qk - m_i);

                            #pragma unroll
                            for (int d = 0; d < D; d++) {
                                float v_val = bf162float(V_bh[key_seq * D + d]);
                                o_acc[d] += w * v_val;
                            }
                            l_i += w;
                        }
                    }
                }
                __syncthreads();
            }
        }

        // Normalize and store
        if (tid < TILE_M) {
            int q_seq = tile_q_start + tid;
            if (q_seq < L) {
                #pragma unroll
                for (int d = 0; d < D; d++) {
                    OS_bh[q_seq * D + d] = o_acc[d] / l_i;
                }
                LSE_bh[q_seq] = m_i + log2f(l_i);
            }
        }
        __syncthreads();
    }
}

// ===== Launcher =====

// D=64: TILE_M=64 fits in smem (16+16+16=48KB)
template<int BLOCK_M, int D, int BLOCK_N, int TOPK, int TILE_M>
cudaError_t launch_attn_d64(
    const uint16_t* Q, const uint16_t* K, const uint16_t* V,
    const int* LUT,
    float* OS, float* LSE_out,
    int B, int H, int L,
    float qk_scale,
    int M_BLOCKS, int TOPK_runtime,
    cudaStream_t stream
) {
    // smem: Q (TILE_M*D) + K (BLOCK_N*D) + qk (TILE_M*BLOCK_N)
    int smem = (TILE_M * D + BLOCK_N * D + TILE_M * BLOCK_N) * sizeof(float);
    dim3 grid(M_BLOCKS, B * H);
    dim3 block(128);

    attn_fwd_kernel_d64<BLOCK_M, D, BLOCK_N, TOPK, TILE_M><<<grid, block, smem, stream>>>(
        Q, K, V, LUT, OS, LSE_out, B, H, L, qk_scale, M_BLOCKS, TOPK_runtime
    );
    return cudaGetLastError();
}

// D=128: TILE_M=32, HALF_BLOCK_N=32 (K processed in 2 sub-blocks, smem=36KB)
template<int BLOCK_M, int D, int BLOCK_N, int TOPK, int TILE_M>
cudaError_t launch_attn_d128(
    const uint16_t* Q, const uint16_t* K, const uint16_t* V,
    const int* LUT,
    float* OS, float* LSE_out,
    int B, int H, int L,
    float qk_scale,
    int M_BLOCKS, int TOPK_runtime,
    cudaStream_t stream
) {
    constexpr int HALF_BLOCK_N = 32;
    // smem: Q (TILE_M*D) + K (HALF_BLOCK_N*D) + qk (TILE_M*HALF_BLOCK_N)
    int smem = (TILE_M * D + HALF_BLOCK_N * D + TILE_M * HALF_BLOCK_N) * sizeof(float);
    dim3 grid(M_BLOCKS, B * H);
    dim3 block(128);

    attn_fwd_kernel_d128<BLOCK_M, D, BLOCK_N, TOPK, TILE_M><<<grid, block, smem, stream>>>(
        Q, K, V, LUT, OS, LSE_out, B, H, L, qk_scale, M_BLOCKS, TOPK_runtime
    );
    return cudaGetLastError();
}

cudaError_t dispatch_attn_fwd(
    int BLOCK_M, int D, int BLOCK_N,
    const uint16_t* Q, const uint16_t* K, const uint16_t* V,
    const int* LUT,
    float* OS, float* LSE_out,
    int B, int H, int L,
    float qk_scale,
    int M_BLOCKS, int TOPK_runtime,
    cudaStream_t stream
) {
    if (D == 64) {
        // TILE_M=64: smem = 64*64*4 + 64*64*4 + 64*64*4 = 16+16+16=48KB
        if (BLOCK_M == 64) {
            return launch_attn_d64<64, 64, 64, 64, 64>(Q, K, V, LUT, OS, LSE_out, B, H, L, qk_scale, M_BLOCKS, TOPK_runtime, stream);
        } else {
            return launch_attn_d64<128, 64, 64, 64, 64>(Q, K, V, LUT, OS, LSE_out, B, H, L, qk_scale, M_BLOCKS, TOPK_runtime, stream);
        }
    } else {
        // D=128: TILE_M=32, HALF_BLOCK_N=32
        // smem = 32*128*4 + 32*128*4 + 32*32*4 = 16+16+4=36KB
        if (BLOCK_M == 64) {
            return launch_attn_d128<64, 128, 64, 64, 32>(Q, K, V, LUT, OS, LSE_out, B, H, L, qk_scale, M_BLOCKS, TOPK_runtime, stream);
        } else {
            return launch_attn_d128<128, 128, 64, 64, 32>(Q, K, V, LUT, OS, LSE_out, B, H, L, qk_scale, M_BLOCKS, TOPK_runtime, stream);
        }
    }
}

// ===== Python callable =====

py::object cuda_attention_forward(
    py::object Q_obj, py::object K_obj, py::object V_obj,
    py::object LUT_obj,
    int BLOCK_M, int BLOCK_N, float qk_scale,
    int M_BLOCKS
) {
    auto Q = Q_obj.cast<torch::Tensor>();
    auto K = K_obj.cast<torch::Tensor>();
    auto V = V_obj.cast<torch::Tensor>();
    auto LUT = LUT_obj.cast<torch::Tensor>();

    int B = Q.size(0), H = Q.size(1), L = Q.size(2), D = Q.size(3);
    int TOPK = LUT.size(3);

    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bf16");

    auto OS = torch::empty({B, H, L, D}, torch::dtype(torch::kFloat32).device(Q.device()));
    auto LSE_out = torch::empty({B, H, L}, torch::dtype(torch::kFloat32).device(Q.device()));

    cudaStream_t stream = 0;
    if (TOPK == 0) {
        OS.zero_();
        LSE_out.zero_();
    } else {
        cudaError_t err = dispatch_attn_fwd(
            BLOCK_M, D, BLOCK_N,
            (uint16_t*)Q.data_ptr(), (uint16_t*)K.data_ptr(), (uint16_t*)V.data_ptr(),
            (int*)LUT.data_ptr(),
            (float*)OS.data_ptr(), (float*)LSE_out.data_ptr(),
            B, H, L, qk_scale, M_BLOCKS, TOPK, stream
        );
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    }
    return py::make_tuple(OS, LSE_out);
}

PYBIND11_MODULE(sparse_linear_attention_cuda_ext, m) {
    m.def("cuda_attention_forward", &cuda_attention_forward,
          "CUDA sparse attention forward");
}

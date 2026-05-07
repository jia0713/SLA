/*
CUDA attention forward kernel v9 - dual TensorCore (Q@K + P@V) for d64.
Grid: (M_BLOCKS, B*H)

- Uses wmma for both Q@K^T and P@V (TensorCore) for d64
- Q_smem, KV_smem stored as bf16; qk_smem as float32
- KV_smem reused: K then V, weights_smem separate for bf16 weights
- V load (tid>=64) overlaps with row scan + weight conv (tid<64)
- D=64: TILE_M=64, 4 warps, full BLOCK_N=64, smem=40KB (3 blocks/SM)
- D=128: TILE_M=64, 4 warps, HALF_BLOCK_N=32, scalar P@V, smem=32KB
*/

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

namespace py = pybind11;
using namespace nvcuda;

__inline__ __device__ float bf162float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

// D=64 kernel: TILE_M=64, BLOCK_N=64, D=64
// Dual TensorCore: WMMA for both Q@K^T and P@V.
// smem (40KB): Q(8) + KV(8, reused K→V) + qk/output(16) + weights(8)
// V load (tid>=64) overlaps with row scan + weight conv (tid<64).
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

    if (idx_m >= M_BLOCKS || idx_bh >= B * H) return;

    int b = idx_bh / H;
    int h = idx_bh % H;
    int bh_offset = ((b * H) + h) * L * D;

    const __nv_bfloat16* Q_bf16 = (const __nv_bfloat16*)Q + bh_offset;
    const __nv_bfloat16* K_bf16 = (const __nv_bfloat16*)K + bh_offset;
    const __nv_bfloat16* V_bf16 = (const __nv_bfloat16*)V + bh_offset;
    const int* LUT_bh = LUT + ((b * H + h) * M_BLOCKS + idx_m) * TOPK_runtime;
    float* OS_bh = OS + bh_offset;
    float* LSE_bh = LSE_out + (b * H + h) * L;

    int q_start = idx_m * BLOCK_M;
    constexpr int NUM_TILES = (BLOCK_M + TILE_M - 1) / TILE_M;

    // smem (40KB): Q(8) + KV(8, reused K→V) + qk/output(16) + weights(8)
    extern __shared__ char smem_raw[];
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* KV_smem = Q_smem + TILE_M * D;
    float* qk_smem = (float*)(KV_smem + BLOCK_N * D);
    __nv_bfloat16* weights_smem = (__nv_bfloat16*)(qk_smem + TILE_M * BLOCK_N);
    float* output_smem = qk_smem;

    int warp_id = tid / 32;
    constexpr int M_TILES = TILE_M / 16;
    constexpr int N_TILES_QK = BLOCK_N / 16;
    constexpr int K_STEPS_QK = D / 16;
    constexpr int N_TILES_PV = D / 16;
    constexpr int K_STEPS_PV = BLOCK_N / 16;
    constexpr int KV_SIZE = BLOCK_N * D;

    for (int tile = 0; tile < NUM_TILES; tile++) {
        int tile_q_start = q_start + tile * TILE_M;
        int tile_q_end = min(tile_q_start + TILE_M, q_start + BLOCK_M);

        for (int i = tid; i < TILE_M * D; i += blockDim.x) {
            int row = tile_q_start + (i / D);
            int col = i % D;
            Q_smem[i] = (row < L) ? __ldg(Q_bf16 + row * D + col) : __float2bfloat16(0.0f);
        }
        __syncthreads();

        float m_i = -INFINITY;
        float l_i = 0.0f;
        float o_acc[D];
        #pragma unroll
        for (int d = 0; d < D; d++) o_acc[d] = 0.0f;

        for (int ki = 0; ki < TOPK_runtime; ki++) {
            int key_block = LUT_bh[ki];
            int key_start = key_block * BLOCK_N;

            // Pass 1: Load K as bf16 into KV_smem
            for (int i = tid; i < KV_SIZE; i += blockDim.x) {
                int row = key_start + (i / D);
                int col = i % D;
                KV_smem[i] = (row < L) ? __ldg(K_bf16 + row * D + col) : __float2bfloat16(0.0f);
            }
            __syncthreads();

            // Pass 2: mma Q @ K^T (TensorCore)
            int m_tile = warp_id;
            if (m_tile < M_TILES) {
                for (int n_tile = 0; n_tile < N_TILES_QK; n_tile++) {
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                    wmma::fill_fragment(c_frag, 0.0f);
                    for (int k_step = 0; k_step < K_STEPS_QK; k_step++) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                        wmma::load_matrix_sync(a_frag, Q_smem + m_tile * 16 * D + k_step * 16, D);
                        wmma::load_matrix_sync(b_frag, KV_smem + n_tile * 16 * D + k_step * 16, D);
                        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }
                    wmma::store_matrix_sync(qk_smem + m_tile * 16 * BLOCK_N + n_tile * 16,
                                            c_frag, BLOCK_N, wmma::mem_row_major);
                }
            }
            __syncthreads();

            // Pass 3: row scan + weight conv (tid<64) + V load (tid>=64)
            float row_sum = 0.0f;
            if (tid < TILE_M) {
                int q_seq = tile_q_start + tid;
                if (q_seq < L && tile_q_start + tid < tile_q_end) {
                    float row_max = -INFINITY;
                    for (int j = 0; j < BLOCK_N; j++) {
                        int key_seq = key_start + j;
                        if (key_seq >= L) break;
                        float qk = qk_smem[tid * BLOCK_N + j] * qk_scale;
                        row_max = fmaxf(row_max, qk);
                    }
                    float new_m = fmaxf(m_i, row_max);
                    float alpha = (m_i == -INFINITY) ? 1.0f : exp2f(m_i - new_m);
                    #pragma unroll
                    for (int d = 0; d < D; d++) o_acc[d] *= alpha;
                    l_i *= alpha;
                    m_i = new_m;

                    for (int j = 0; j < BLOCK_N; j++) {
                        int key_seq = key_start + j;
                        if (key_seq >= L) break;
                        float qk = qk_smem[tid * BLOCK_N + j] * qk_scale;
                        float w = exp2f(qk - m_i);
                        weights_smem[tid * BLOCK_N + j] = __float2bfloat16(w);
                        row_sum += w;
                    }
                }
            } else {
                // tid 64..127: load V into KV_smem (overwrite K)
                int t = tid - TILE_M;
                for (int i = t; i < KV_SIZE; i += (blockDim.x - TILE_M)) {
                    int row = key_start + (i / D);
                    int col = i % D;
                    KV_smem[i] = (row < L) ? __ldg(V_bf16 + row * D + col) : __float2bfloat16(0.0f);
                }
            }
            __syncthreads();

            // Pass 4: mma weights @ V (TensorCore)
            if (m_tile < M_TILES) {
                for (int n_tile = 0; n_tile < N_TILES_PV; n_tile++) {
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                    wmma::fill_fragment(c_frag, 0.0f);
                    for (int k_step = 0; k_step < K_STEPS_PV; k_step++) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
                        wmma::load_matrix_sync(a_frag, weights_smem + m_tile * 16 * BLOCK_N + k_step * 16, BLOCK_N);
                        wmma::load_matrix_sync(b_frag, KV_smem + k_step * 16 * D + n_tile * 16, D);
                        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }
                    wmma::store_matrix_sync(output_smem + m_tile * 16 * D + n_tile * 16,
                                            c_frag, D, wmma::mem_row_major);
                }
            }
            __syncthreads();

            // Pass 5: accumulate output into o_acc, update l_i
            if (tid < TILE_M && tile_q_start + tid < tile_q_end) {
                int q_seq = tile_q_start + tid;
                if (q_seq < L) {
                    #pragma unroll
                    for (int d = 0; d < D; d++) {
                        o_acc[d] += output_smem[tid * D + d];
                    }
                    l_i += row_sum;
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

// D=128 kernel: TILE_M=64, HALF_BLOCK_N=32, D=128
// WMMA for Q@K, scalar P@V. K_smem reused for V.
// Half-block split keeps smem at 32KB for 4 blocks/SM occupancy.
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

    if (idx_m >= M_BLOCKS || idx_bh >= B * H) return;

    int b = idx_bh / H;
    int h = idx_bh % H;
    int bh_offset = ((b * H) + h) * L * D;

    const __nv_bfloat16* Q_bf16 = (const __nv_bfloat16*)Q + bh_offset;
    const __nv_bfloat16* K_bf16 = (const __nv_bfloat16*)K + bh_offset;
    const __nv_bfloat16* V_bf16 = (const __nv_bfloat16*)V + bh_offset;
    const int* LUT_bh = LUT + ((b * H + h) * M_BLOCKS + idx_m) * TOPK_runtime;
    float* OS_bh = OS + bh_offset;
    float* LSE_bh = LSE_out + (b * H + h) * L;

    int q_start = idx_m * BLOCK_M;
    constexpr int HALF_BLOCK_N = 32;
    constexpr int NUM_TILES = (BLOCK_M + TILE_M - 1) / TILE_M;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* K_smem = Q_smem + TILE_M * D;
    float* qk_smem = (float*)(K_smem + HALF_BLOCK_N * D);

    int warp_id = tid / 32;
    constexpr int M_TILES = TILE_M / 16;
    constexpr int N_TILES = HALF_BLOCK_N / 16;
    constexpr int K_STEPS = D / 16;

    for (int tile = 0; tile < NUM_TILES; tile++) {
        int tile_q_start = q_start + tile * TILE_M;
        int tile_q_end = min(tile_q_start + TILE_M, q_start + BLOCK_M);

        for (int i = tid; i < TILE_M * D; i += blockDim.x) {
            int row = tile_q_start + (i / D);
            int col = i % D;
            int gloc = row * D + col;
            Q_smem[i] = (row < L) ? __ldg(Q_bf16 + gloc) : __float2bfloat16(0.0f);
        }
        __syncthreads();

        float m_i = -INFINITY;
        float l_i = 0.0f;
        float o_acc[D];
        #pragma unroll
        for (int d = 0; d < D; d++) o_acc[d] = 0.0f;

        for (int ki = 0; ki < TOPK_runtime; ki++) {
            int key_block = LUT_bh[ki];
            int key_start = key_block * BLOCK_N;

            for (int sub_k = 0; sub_k < BLOCK_N; sub_k += HALF_BLOCK_N) {
                int sub_key_start = key_start + sub_k;
                int actual_keys = min(HALF_BLOCK_N, BLOCK_N - sub_k);

                // Load K sub-block as bf16
                for (int i = tid; i < actual_keys * D; i += blockDim.x) {
                    int row = sub_key_start + (i / D);
                    int col = i % D;
                    int gloc = row * D + col;
                    K_smem[i] = (row < L) ? __ldg(K_bf16 + gloc) : __float2bfloat16(0.0f);
                }
                __syncthreads();

                // mma Q @ K^T (TensorCore)
                int m_tile = warp_id;
                if (m_tile < M_TILES) {
                    for (int n_tile = 0; n_tile < N_TILES; n_tile++) {
                        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                        wmma::fill_fragment(c_frag, 0.0f);
                        for (int k_step = 0; k_step < K_STEPS; k_step++) {
                            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                            wmma::load_matrix_sync(a_frag, Q_smem + m_tile * 16 * D + k_step * 16, D);
                            wmma::load_matrix_sync(b_frag, K_smem + n_tile * 16 * D + k_step * 16, D);
                            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                        }
                        wmma::store_matrix_sync(qk_smem + m_tile * 16 * HALF_BLOCK_N + n_tile * 16,
                                                c_frag, HALF_BLOCK_N, wmma::mem_row_major);
                    }
                }
                __syncthreads();

                // Row scan + rescale
                if (tid < TILE_M && tile_q_start + tid < tile_q_end) {
                    int q_seq = tile_q_start + tid;
                    if (q_seq < L) {
                        float thread_qk_max = -INFINITY;
                        for (int j = 0; j < actual_keys; j++) {
                            int key_seq = sub_key_start + j;
                            if (key_seq >= L) break;
                            float qk = qk_smem[tid * HALF_BLOCK_N + j] * qk_scale;
                            qk_smem[tid * HALF_BLOCK_N + j] = qk;
                            thread_qk_max = fmaxf(thread_qk_max, qk);
                        }
                        float new_m = fmaxf(m_i, thread_qk_max);
                        float alpha = (m_i == -INFINITY) ? 1.0f : exp2f(m_i - new_m);
                        #pragma unroll
                        for (int d = 0; d < D; d++) o_acc[d] *= alpha;
                        l_i *= alpha;
                        m_i = new_m;
                    }
                }
                __syncthreads();

                // Reload K_smem with V sub-block
                for (int i = tid; i < actual_keys * D; i += blockDim.x) {
                    int row = sub_key_start + (i / D);
                    int col = i % D;
                    int gloc = row * D + col;
                    K_smem[i] = (row < L) ? __ldg(V_bf16 + gloc) : __float2bfloat16(0.0f);
                }
                __syncthreads();

                // Scalar P@V accumulate
                if (tid < TILE_M && tile_q_start + tid < tile_q_end) {
                    int q_seq = tile_q_start + tid;
                    if (q_seq < L) {
                        for (int j = 0; j < actual_keys; j++) {
                            int key_seq = sub_key_start + j;
                            if (key_seq >= L) break;
                            float w = exp2f(qk_smem[tid * HALF_BLOCK_N + j] - m_i);
                            #pragma unroll
                            for (int d = 0; d < D; d++) {
                                o_acc[d] += w * bf162float(K_smem[j * D + d]);
                            }
                            l_i += w;
                        }
                    }
                }
                __syncthreads();
            }
        }

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

// ===== Launchers =====

template<int BLOCK_M, int D, int BLOCK_N, int TOPK, int TILE_M>
cudaError_t launch_attn_d64(
    const uint16_t* Q, const uint16_t* K, const uint16_t* V,
    const int* LUT, float* OS, float* LSE_out,
    int B, int H, int L, float qk_scale,
    int M_BLOCKS, int TOPK_runtime, cudaStream_t stream
) {
    // smem (40KB): Q(8) + KV(8) + qk/output(16) + weights(8)
    int smem = TILE_M * D * sizeof(__nv_bfloat16)
             + BLOCK_N * D * sizeof(__nv_bfloat16)
             + TILE_M * BLOCK_N * sizeof(float)
             + TILE_M * BLOCK_N * sizeof(__nv_bfloat16);
    dim3 grid(M_BLOCKS, B * H);
    dim3 block(128);
    cudaFuncSetAttribute(attn_fwd_kernel_d64<BLOCK_M, D, BLOCK_N, TOPK, TILE_M>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attn_fwd_kernel_d64<BLOCK_M, D, BLOCK_N, TOPK, TILE_M><<<grid, block, smem, stream>>>(
        Q, K, V, LUT, OS, LSE_out, B, H, L, qk_scale, M_BLOCKS, TOPK_runtime);
    return cudaGetLastError();
}

template<int BLOCK_M, int D, int BLOCK_N, int TOPK, int TILE_M>
cudaError_t launch_attn_d128(
    const uint16_t* Q, const uint16_t* K, const uint16_t* V,
    const int* LUT, float* OS, float* LSE_out,
    int B, int H, int L, float qk_scale,
    int M_BLOCKS, int TOPK_runtime, cudaStream_t stream
) {
    constexpr int HALF_BLOCK_N = 32;
    int smem = TILE_M * D * sizeof(__nv_bfloat16)
             + HALF_BLOCK_N * D * sizeof(__nv_bfloat16)
             + TILE_M * HALF_BLOCK_N * sizeof(float);
    dim3 grid(M_BLOCKS, B * H);
    dim3 block(128);
    attn_fwd_kernel_d128<BLOCK_M, D, BLOCK_N, TOPK, TILE_M><<<grid, block, smem, stream>>>(
        Q, K, V, LUT, OS, LSE_out, B, H, L, qk_scale, M_BLOCKS, TOPK_runtime);
    return cudaGetLastError();
}

cudaError_t dispatch_attn_fwd(
    int BLOCK_M, int D, int BLOCK_N,
    const uint16_t* Q, const uint16_t* K, const uint16_t* V,
    const int* LUT, float* OS, float* LSE_out,
    int B, int H, int L, float qk_scale,
    int M_BLOCKS, int TOPK_runtime, cudaStream_t stream
) {
    if (D == 64) {
        if (BLOCK_M == 64)
            return launch_attn_d64<64, 64, 64, 64, 64>(Q, K, V, LUT, OS, LSE_out, B, H, L, qk_scale, M_BLOCKS, TOPK_runtime, stream);
        else
            return launch_attn_d64<128, 64, 64, 64, 64>(Q, K, V, LUT, OS, LSE_out, B, H, L, qk_scale, M_BLOCKS, TOPK_runtime, stream);
    } else {
        if (BLOCK_M == 64)
            return launch_attn_d128<64, 128, 64, 64, 64>(Q, K, V, LUT, OS, LSE_out, B, H, L, qk_scale, M_BLOCKS, TOPK_runtime, stream);
        else
            return launch_attn_d128<128, 128, 64, 64, 64>(Q, K, V, LUT, OS, LSE_out, B, H, L, qk_scale, M_BLOCKS, TOPK_runtime, stream);
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

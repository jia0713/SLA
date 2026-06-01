#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <type_traits>

#if defined(USE_MACA) && defined(SLA_ENABLE_MACA_CUTE)
#define SLA_HAS_MACA_CUTE 1
#else
#define SLA_HAS_MACA_CUTE 0
#endif

#if SLA_HAS_MACA_CUTE

#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>

#include <mctlass/mctlass.h>
#include <mctlass/numeric_types.h>

#include "kernel_traits.h"
#include "softmax.h"
#include "utils.h"

namespace {

using namespace cute;

constexpr float kLog2E = 1.4426950408889634f;
using SparseD64Traits =
    Flash_fwd_kernel_traits<64, 64, 64, 4, false, false, mctlass::half_t, 64>;
using SparseD128Traits =
    Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, mctlass::half_t, 128>;

template <int TOPK>
__global__ void sparse_attn_fwd_maca_cute_d64_bm64_kernel(
    const mctlass::half_t* __restrict__ q,
    const mctlass::half_t* __restrict__ k,
    const mctlass::half_t* __restrict__ v,
    const int64_t* __restrict__ lut,
    mctlass::half_t* __restrict__ out,
    int64_t L,
    int64_t M_BLOCKS,
    int64_t topk,
    float scale,
    float scale_log2) {
  using Kernel_traits = SparseD64Traits;
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;

  static_assert(Kernel_traits::kNThreads == 256);
  static_assert(Kernel_traits::kHeadDim == 64);
  static_assert(Kernel_traits::kBlockM == 64);
  static_assert(Kernel_traits::kBlockN == 64);

  extern __shared__ char smem_[];
  uint32_t tQrQ[int(Kernel_traits::kRegSize)];
  uint32_t tKrK[int(Kernel_traits::kRegSize / 2)];
  uint32_t tVrV[int(Kernel_traits::kRegSize / 2)];

  const int tidx = threadIdx.x;
  const int64_t m_block = blockIdx.x;
  const int64_t bh = blockIdx.y;
  const int64_t qkv_base = bh * L * 64;
  const int64_t row_offset_q = qkv_base + m_block * 64 * 64;
  const int64_t lut_base = (bh * M_BLOCKS + m_block) * topk;

  Tensor gQ = make_tensor(
      make_gmem_ptr(const_cast<Element*>(q) + row_offset_q),
      Shape<Int<64>, Int<64>>{},
      make_stride(Int<64>{}, _1{}));
  Tensor sQ = make_tensor(
      make_smem_ptr(reinterpret_cast<Element*>(smem_)),
      typename Kernel_traits::SmemLayoutQ{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

  Tensor cQ = make_identity_tensor(Shape<Int<64>, Int<64>>{});
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
  flash::copy_global_to_reg</*Is_even_MN=*/true, /*Is_even_K=*/true>(
      tQgQ, tQrQ, tQcQ, /*d=*/64, /*max_MN=*/64);
  flash::copy_reg_to_share(tQrQ, tQsQ);

  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  flash::sync_threads();
  Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
  cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);

  // Keep Q and K in separate shared-memory tiles. The FlashAttention example
  // can reuse Q/K smem after caching Q in registers, but the standalone SLA
  // kernel is more robust with the explicit Q-smem path.
  Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
  Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
  Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
  Tensor sVtNoSwizzle =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
  Tensor tSrK = thr_mma.partition_fragment_B(sK);
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<64>, Int<64>>{});
  clear(acc_o);
  flash::Softmax<size<1>(acc_o)> softmax;

  constexpr int ldg_Num = (64 * 64 / Kernel_traits::kNThreads) / 4;
  constexpr int lds_Tuple = 1;
  constexpr bool Is_perm_4x4 = true;
  const uint32_t laneId = __lane_id();
  const int cpy_offset = ((laneId & 0xf) << 2) - (laneId & 0xf);
  const uint32_t perm_mask[2] = {0x05040100, 0x07060302};

  int tVgV_offset[ldg_Num];
  uint32_t* tVsV_ptr[ldg_Num];
  int tVcV[ldg_Num];
#pragma unroll
  for (int i = 0; i < ldg_Num; ++i) {
    const int gv_row =
        (((laneId >> 4) & 0x1) << 2) + ((laneId >> 5) << 5) + (i & 0x3);
    const int gv_col = (laneId & 0xf) << 2;
    const int old_gv_row = laneId >> 3;
    const int old_gv_col = (laneId & 0x7) << 3;
    tVgV_offset[i] = (gv_row - old_gv_row) * 64 + (gv_col - old_gv_col);

    const int old_sv_row = old_gv_row;
    const int old_sv_col = old_gv_col;
    tVsV_ptr[i] = reinterpret_cast<uint32_t*>(
        tVsV(_, _0{}, _0{}).data().ptr_ + ((gv_row - old_sv_row) << 6) +
        (gv_col - old_sv_col));
    tVcV[i] = gv_row + ((tidx >> 6) << 3);
  }

  const int block_count = TOPK > 0 ? TOPK : static_cast<int>(topk);
  if (block_count <= 0) {
    return;
  }

  Tensor cKV = make_identity_tensor(Shape<Int<64>, Int<64>>{});
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

  for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    const int64_t n_block = lut[lut_base + block_idx];
    const int64_t row_offset_kv = qkv_base + n_block * 64 * 64;

    Tensor gK = make_tensor(
        make_gmem_ptr(const_cast<Element*>(k) + row_offset_kv),
        Shape<Int<64>, Int<64>>{},
        make_stride(Int<64>{}, _1{}));
    Tensor gV = make_tensor(
        make_gmem_ptr(const_cast<Element*>(v) + row_offset_kv),
        Shape<Int<64>, Int<64>>{},
        make_stride(Int<64>{}, _1{}));
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    flash::copy_global_to_reg</*Is_even_MN=*/true, /*Is_even_K=*/true>(
        tKgK, tKrK, tKVcKV, /*d=*/64, /*max_MN=*/64);
    flash::copy_global_to_reg_V</*Is_even_MN=*/true,
                                /*Is_even_K=*/true,
                                /*ldg_type=*/0,
                                ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, /*d=*/64, /*max_MN=*/64);

    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<64>, Int<64>>{});
    clear(acc_s);
    flash::copy_reg_to_share(tKrK, tKsK);
    flash::copy_reg_to_share_V</*sts_type=*/0, ldg_Num, Is_perm_4x4>(
        tVrV, tVsV_ptr, perm_mask);
    flash::sync_threads();

    flash::gemm</*A_in_regs=*/false>(
        acc_s,
        tSrQ,
        tSrK,
        tSsQ,
        tSsK,
        tiled_mma,
        smem_tiled_copy_Q,
        smem_tiled_copy_K,
        smem_thr_copy_Q,
        smem_thr_copy_K);

    if (block_idx == 0) {
      softmax.template softmax_rescale_o</*Is_first=*/true,
                                          /*Check_inf=*/false,
                                          /*Syncthreads=*/false,
                                          /*AddVec=*/true>(
          acc_s, acc_o, scale_log2);
    } else {
      softmax.template softmax_rescale_o</*Is_first=*/false,
                                          /*Check_inf=*/false,
                                          /*Syncthreads=*/false,
                                          /*AddVec=*/true>(
          acc_s, acc_o, scale_log2);
    }

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
    flash::gemm_rs<Is_perm_4x4, lds_Tuple>(
        acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset);
    flash::sync_threads();
  }

  Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false>(
      acc_o, scale, /*rp_dropout=*/1.0f);
  (void)lse;

  CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)

  // The MACA accumulator layout is not row-major. Reuse FlashAttention's
  // hdim64 epilogue: byte-permute register fragments into the O shared layout,
  // then issue coalesced b128 global stores.
  Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

  flash::barrier();
  const int sm_col = ((((laneId & 0x7) ^ (laneId >> 5)) << 1) +
                      ((laneId >> 4) & 0x1))
                     << 2;
  auto s_ptr = taccOsO.data().ptr_ - sm_col;
  auto ptr = reinterpret_cast<uint32_t*>(rO.data().ptr_);

  auto a = __builtin_mxc_byte_perm(ptr[2], ptr[0], perm_mask[0]);
  ptr[2] = __builtin_mxc_byte_perm(ptr[2], ptr[0], perm_mask[1]);
  ptr[0] = a;
  a = __builtin_mxc_byte_perm(ptr[3], ptr[1], perm_mask[0]);
  auto b = __builtin_mxc_byte_perm(ptr[3], ptr[1], perm_mask[1]);
  ptr[1] = __builtin_mxc_byte_perm(ptr[6], ptr[4], perm_mask[0]);
  ptr[3] = __builtin_mxc_byte_perm(ptr[6], ptr[4], perm_mask[1]);
  ptr[4] = a;
  ptr[6] = b;
  a = __builtin_mxc_byte_perm(ptr[7], ptr[5], perm_mask[0]);
  ptr[7] = __builtin_mxc_byte_perm(ptr[7], ptr[5], perm_mask[1]);
  ptr[5] = a;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    auto reg_ptr = reinterpret_cast<uint64_t*>(rO.data().ptr_ + (i << 2));
    const int col =
        ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (i >> 1))) << 1) +
         (i & 0x1))
        << 2;
    auto sm_ptr = reinterpret_cast<uint64_t*>(s_ptr + col);
    sm_ptr[0] = reg_ptr[0];
  }

  const int64_t row_offset_o = bh * L * 64 + m_block * 64 * 64;
  Tensor gO = make_tensor(
      make_gmem_ptr(out + row_offset_o),
      Shape<Int<64>, Int<64>>{},
      make_stride(Int<64>{}, _1{}));
  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  flash::sync_threads();
  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
  Tensor cO = make_identity_tensor(Shape<Int<64>, Int<64>>{});
  Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
  flash::copy_reg_to_global</*Is_even_MN=*/true, /*Is_even_K=*/true>(
      tOrO, tOgO, tOcO, /*d=*/64, /*max_MN=*/64);
}

void launch_sparse_attn_fwd_maca_cute_d64(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& lut,
    torch::Tensor& out,
    int64_t topk) {
  const auto B = q.size(0);
  const auto H = q.size(1);
  const auto L = q.size(2);
  const auto M_BLOCKS = (L + 63) / 64;
  const float scale = 1.0f / std::sqrt(64.0f);
  const float scale_log2 = scale * kLog2E;

  dim3 grid(M_BLOCKS, B * H);
  dim3 block(256);
  constexpr int kSmemBytes = SparseD64Traits::kSmemSize;

#define SLA_LAUNCH_CUTE(TOPK_VALUE)                                            \
  sparse_attn_fwd_maca_cute_d64_bm64_kernel<TOPK_VALUE>                        \
      <<<grid, block, kSmemBytes, at::cuda::getCurrentCUDAStream()>>>(          \
          reinterpret_cast<const mctlass::half_t*>(q.data_ptr<at::Half>()),     \
          reinterpret_cast<const mctlass::half_t*>(k.data_ptr<at::Half>()),     \
          reinterpret_cast<const mctlass::half_t*>(v.data_ptr<at::Half>()),     \
          lut.data_ptr<int64_t>(),                                              \
          reinterpret_cast<mctlass::half_t*>(out.data_ptr<at::Half>()),         \
          L,                                                                    \
          M_BLOCKS,                                                             \
          topk,                                                                 \
          scale,                                                                \
          scale_log2)

  if (topk == 1) {
    SLA_LAUNCH_CUTE(1);
  } else if (topk == 2) {
    SLA_LAUNCH_CUTE(2);
  } else if (topk == 4) {
    SLA_LAUNCH_CUTE(4);
  } else if (topk == 8) {
    SLA_LAUNCH_CUTE(8);
  } else if (topk == 16) {
    SLA_LAUNCH_CUTE(16);
  } else if (topk == 32) {
    SLA_LAUNCH_CUTE(32);
  } else {
    SLA_LAUNCH_CUTE(-1);
  }

#undef SLA_LAUNCH_CUTE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int TOPK>
__global__ void sparse_attn_fwd_maca_cute_d128_bm64_kernel(
    const mctlass::half_t* __restrict__ q,
    const mctlass::half_t* __restrict__ k,
    const mctlass::half_t* __restrict__ v,
    const int64_t* __restrict__ lut,
    mctlass::half_t* __restrict__ out,
    int64_t L,
    int64_t M_BLOCKS,
    int64_t topk,
    float scale,
    float scale_log2) {
  using Kernel_traits = SparseD128Traits;
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;

  static_assert(Kernel_traits::kNThreads == 256);
  static_assert(Kernel_traits::kHeadDim == 128);
  static_assert(Kernel_traits::kBlockM == 64);
  static_assert(Kernel_traits::kBlockN == 64);

  extern __shared__ char smem_[];
  uint32_t tQrQ[int(Kernel_traits::kRegSize)];
  uint32_t tKrK[int(Kernel_traits::kRegSize / 2)];
  uint32_t tVrV[int(Kernel_traits::kRegSize / 2)];

  const int tidx = threadIdx.x;
  const uint32_t laneId = __lane_id();
  const int64_t m_block = blockIdx.x;
  const int64_t bh = blockIdx.y;
  const int64_t qkv_base = bh * L * 128;
  const int64_t row_offset_q = qkv_base + m_block * 64 * 128;
  const int64_t lut_base = (bh * M_BLOCKS + m_block) * topk;

  Tensor gQ = make_tensor(
      make_gmem_ptr(const_cast<Element*>(q) + row_offset_q),
      Shape<Int<64>, Int<128>>{},
      make_stride(Int<128>{}, _1{}));
  Tensor sQ = make_tensor(
      make_smem_ptr(reinterpret_cast<Element*>(smem_)),
      typename Kernel_traits::SmemLayoutQ{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

  Tensor cQ = make_identity_tensor(Shape<Int<64>, Int<128>>{});
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
  flash::copy_global_to_reg</*Is_even_MN=*/true, /*Is_even_K=*/true>(
      tQgQ, tQrQ, tQcQ, /*d=*/128, /*max_MN=*/64);
  flash::copy_reg_to_share(tQrQ, tQsQ);

  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  // hdim128 uses two 64-column halves in the MMA fragment. This is the same
  // register materialization pattern used by the FlashAttention hdim128 kernel.
  {
    const int col =
        (((laneId & 0x7) ^ (laneId >> 5)) << 3) + (((laneId >> 4) & 0x1) << 2);
    flash::sync_threads();
    auto tSsQ_ptr = reinterpret_cast<uint32_t*>(tSsQ.data().ptr_ - col);
    auto tSrQ_ptr = reinterpret_cast<uint32_t*>(tSrQ.data());
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int q_col =
          (((laneId & 0x7) ^ ((laneId >> 5) + (i << 1))) << 2) +
          (((laneId >> 4) & 0x1) << 1);
      auto reg0 = reinterpret_cast<uint64_t*>(tSrQ_ptr + (i << 1));
      auto smem0 = reinterpret_cast<uint64_t*>(tSsQ_ptr + q_col);
      auto reg1 = reinterpret_cast<uint64_t*>(tSrQ_ptr + (i << 1) + 8);
      auto smem1 = reinterpret_cast<uint64_t*>(tSsQ_ptr + q_col + 64 * 32);
      reg0[0] = smem0[0];
      reg1[0] = smem1[0];
    }
  }

  Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
  Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
  Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
  Tensor sVtNoSwizzle =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
  Tensor tSrK = thr_mma.partition_fragment_B(sK);
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<64>, Int<128>>{});
  clear(acc_o);
  flash::Softmax<size<1>(acc_o)> softmax;

  constexpr int ldg_Num = (64 * 128 / Kernel_traits::kNThreads) / 8;
  constexpr int lds_Tuple = 2;
  constexpr bool Is_perm_4x4 = true;
  const int cpy_offset = ((laneId & 0xf) << 2) - (laneId & 0xf);
  const uint32_t tOsVt_stride = get<1>(get<1>(tOsVt(_, _, _0{}).layout().stride()));
  const uint32_t tOrVt_stride = get<1>(get<1>(tOrVt(_, _, _0{}).layout().stride()));
  const uint32_t perm_mask[2] = {0x05040100, 0x07060302};

  int tVgV_offset[ldg_Num];
  uint32_t* tVsV_ptr[ldg_Num];
  int tVcV[ldg_Num];
#pragma unroll
  for (int i = 0; i < ldg_Num; ++i) {
    const int gv_row =
        (((laneId >> 4) & 0x1) << 2) + ((laneId >> 5) << 5) + (i & 0x3);
    const int gv_col = (laneId & 0xf) << 3;
    const int old_gv_row = laneId >> 3;
    const int old_gv_col = (laneId & 0x7) << 3;
    tVgV_offset[i] = (gv_row - old_gv_row) * 128 + (gv_col - old_gv_col);

    const int sv_row = gv_row + ((old_gv_row & 0x1) * 64);
    tVsV_ptr[i] = reinterpret_cast<uint32_t*>(
        tVsV(_, _0{}, _0{}).data().ptr_ + ((sv_row - old_gv_row) << 6));
    tVcV[i] = gv_row + ((tidx >> 6) << 3);
  }

  const int block_count = TOPK > 0 ? TOPK : static_cast<int>(topk);
  if (block_count <= 0) {
    return;
  }

  Tensor cKV = make_identity_tensor(Shape<Int<64>, Int<128>>{});
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

  for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    const int64_t n_block = lut[lut_base + block_idx];
    const int64_t row_offset_kv = qkv_base + n_block * 64 * 128;

    Tensor gK = make_tensor(
        make_gmem_ptr(const_cast<Element*>(k) + row_offset_kv),
        Shape<Int<64>, Int<128>>{},
        make_stride(Int<128>{}, _1{}));
    Tensor gV = make_tensor(
        make_gmem_ptr(const_cast<Element*>(v) + row_offset_kv),
        Shape<Int<64>, Int<128>>{},
        make_stride(Int<128>{}, _1{}));
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    flash::copy_global_to_reg</*Is_even_MN=*/true, /*Is_even_K=*/true>(
        tKgK, tKrK, tKVcKV, /*d=*/128, /*max_MN=*/64);
    flash::copy_global_to_reg_V</*Is_even_MN=*/true,
                                /*Is_even_K=*/true,
                                /*ldg_type=*/true,
                                ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, /*d=*/128, /*max_MN=*/64);

    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<64>, Int<64>>{});
    clear(acc_s);
    flash::copy_reg_to_share(tKrK, tKsK);
    flash::copy_reg_to_share_V</*sts_type=*/true, ldg_Num>(
        tVrV, tVsV_ptr, perm_mask);
    flash::sync_threads();

    flash::gemm</*A_in_regs=*/true>(
        acc_s,
        tSrQ,
        tSrK,
        tSsQ,
        tSsK,
        tiled_mma,
        smem_tiled_copy_Q,
        smem_tiled_copy_K,
        smem_thr_copy_Q,
        smem_thr_copy_K);

    if (block_idx == 0) {
      softmax.template softmax_rescale_o</*Is_first=*/true,
                                          /*Check_inf=*/false,
                                          /*Syncthreads=*/true,
                                          /*AddVec=*/true>(
          acc_s, acc_o, scale_log2);
    } else {
      softmax.template softmax_rescale_o</*Is_first=*/false,
                                          /*Check_inf=*/false,
                                          /*Syncthreads=*/true,
                                          /*AddVec=*/true>(
          acc_s, acc_o, scale_log2);
    }

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
    flash::gemm_rs<Is_perm_4x4, lds_Tuple>(
        acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
    flash::sync_threads();
  }

  Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false>(
      acc_o, scale, /*rp_dropout=*/1.0f);
  (void)lse;

  CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)

  Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

  flash::barrier();
  int sm_col = ((((laneId & 0x7) ^ (laneId >> 5)) << 1) + ((laneId >> 4) & 0x1)) << 2;
  auto s_ptr = taccOsO.data().ptr_ - sm_col;
  auto ptr0 = reinterpret_cast<uint32_t*>(rO.data().ptr_);
  auto ptr1 = reinterpret_cast<uint32_t*>(rO.data().ptr_ + 16);
  auto ptr = reinterpret_cast<uint64_t*>(rO.data().ptr_);

  auto temp_a = __builtin_mxc_byte_perm(ptr0[2], ptr0[0], perm_mask[0]);
  ptr0[2] = __builtin_mxc_byte_perm(ptr0[2], ptr0[0], perm_mask[1]);
  ptr0[0] = temp_a;
  temp_a = __builtin_mxc_byte_perm(ptr0[3], ptr0[1], perm_mask[0]);
  auto temp_b = __builtin_mxc_byte_perm(ptr0[3], ptr0[1], perm_mask[1]);
  ptr0[1] = __builtin_mxc_byte_perm(ptr0[6], ptr0[4], perm_mask[0]);
  ptr0[3] = __builtin_mxc_byte_perm(ptr0[6], ptr0[4], perm_mask[1]);

  auto temp_c = __builtin_mxc_byte_perm(ptr1[2], ptr1[0], perm_mask[0]);
  ptr1[2] = __builtin_mxc_byte_perm(ptr1[2], ptr1[0], perm_mask[1]);
  ptr1[0] = temp_c;
  temp_c = __builtin_mxc_byte_perm(ptr1[3], ptr1[1], perm_mask[0]);
  auto temp_d = __builtin_mxc_byte_perm(ptr1[3], ptr1[1], perm_mask[1]);
  ptr1[1] = __builtin_mxc_byte_perm(ptr1[6], ptr1[4], perm_mask[0]);
  ptr1[3] = __builtin_mxc_byte_perm(ptr1[6], ptr1[4], perm_mask[1]);

  sm_col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (0 >> 1))) << 1) + (0 & 0x1)) << 2;
  auto sm_ptr = reinterpret_cast<uint64_t*>(s_ptr + sm_col);
  sm_ptr[0] = ptr[0];
  sm_ptr[1024] = ptr[4];
  sm_col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (1 >> 1))) << 1) + (1 & 0x1)) << 2;
  sm_ptr = reinterpret_cast<uint64_t*>(s_ptr + sm_col);
  sm_ptr[0] = ptr[1];
  sm_ptr[1024] = ptr[5];

  ptr0[4] = temp_a;
  ptr0[6] = temp_b;
  temp_a = __builtin_mxc_byte_perm(ptr0[7], ptr0[5], perm_mask[0]);
  ptr0[7] = __builtin_mxc_byte_perm(ptr0[7], ptr0[5], perm_mask[1]);
  ptr0[5] = temp_a;

  ptr1[4] = temp_c;
  ptr1[6] = temp_d;
  temp_c = __builtin_mxc_byte_perm(ptr1[7], ptr1[5], perm_mask[0]);
  ptr1[7] = __builtin_mxc_byte_perm(ptr1[7], ptr1[5], perm_mask[1]);
  ptr1[5] = temp_c;

  sm_col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (2 >> 1))) << 1) + (2 & 0x1)) << 2;
  sm_ptr = reinterpret_cast<uint64_t*>(s_ptr + sm_col);
  sm_ptr[0] = ptr[2];
  sm_ptr[1024] = ptr[6];
  sm_col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (3 >> 1))) << 1) + (3 & 0x1)) << 2;
  sm_ptr = reinterpret_cast<uint64_t*>(s_ptr + sm_col);
  sm_ptr[0] = ptr[3];
  sm_ptr[1024] = ptr[7];

  const int64_t row_offset_o = bh * L * 128 + m_block * 64 * 128;
  Tensor gO = make_tensor(
      make_gmem_ptr(out + row_offset_o),
      Shape<Int<64>, Int<128>>{},
      make_stride(Int<128>{}, _1{}));
  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  flash::sync_threads();
  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
  Tensor cO = make_identity_tensor(Shape<Int<64>, Int<128>>{});
  Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
  flash::copy_reg_to_global</*Is_even_MN=*/true, /*Is_even_K=*/true>(
      tOrO, tOgO, tOcO, /*d=*/128, /*max_MN=*/64);
}

void launch_sparse_attn_fwd_maca_cute_d128(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& lut,
    torch::Tensor& out,
    int64_t topk) {
  const auto B = q.size(0);
  const auto H = q.size(1);
  const auto L = q.size(2);
  const auto M_BLOCKS = (L + 63) / 64;
  const float scale = 1.0f / std::sqrt(128.0f);
  const float scale_log2 = scale * kLog2E;

  dim3 grid(M_BLOCKS, B * H);
  dim3 block(256);
  constexpr int kSmemBytes = SparseD128Traits::kSmemSize;

#define SLA_LAUNCH_CUTE_D128(TOPK_VALUE)                                      \
  sparse_attn_fwd_maca_cute_d128_bm64_kernel<TOPK_VALUE>                      \
      <<<grid, block, kSmemBytes, at::cuda::getCurrentCUDAStream()>>>(         \
          reinterpret_cast<const mctlass::half_t*>(q.data_ptr<at::Half>()),    \
          reinterpret_cast<const mctlass::half_t*>(k.data_ptr<at::Half>()),    \
          reinterpret_cast<const mctlass::half_t*>(v.data_ptr<at::Half>()),    \
          lut.data_ptr<int64_t>(),                                             \
          reinterpret_cast<mctlass::half_t*>(out.data_ptr<at::Half>()),        \
          L,                                                                   \
          M_BLOCKS,                                                            \
          topk,                                                                \
          scale,                                                               \
          scale_log2)

  if (topk == 1) {
    SLA_LAUNCH_CUTE_D128(1);
  } else if (topk == 2) {
    SLA_LAUNCH_CUTE_D128(2);
  } else if (topk == 4) {
    SLA_LAUNCH_CUTE_D128(4);
  } else if (topk == 8) {
    SLA_LAUNCH_CUTE_D128(8);
  } else if (topk == 16) {
    SLA_LAUNCH_CUTE_D128(16);
  } else if (topk == 32) {
    SLA_LAUNCH_CUTE_D128(32);
  } else {
    SLA_LAUNCH_CUTE_D128(-1);
  }

#undef SLA_LAUNCH_CUTE_D128
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace
#endif

torch::Tensor sparse_attn_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor lut,
    int64_t topk,
    int64_t block_m,
    int64_t block_n) {
  const c10::cuda::CUDAGuard device_guard(q.device());
  (void)block_m;
  (void)block_n;

#if SLA_HAS_MACA_CUTE
  auto out = torch::empty_like(q);
  if (q.size(3) == 64) {
    launch_sparse_attn_fwd_maca_cute_d64(q, k, v, lut, out, topk);
  } else {
    launch_sparse_attn_fwd_maca_cute_d128(q, k, v, lut, out, topk);
  }
  return out;
#else
  TORCH_CHECK(false, "MACA CUTE sparse attention was not enabled at build time");
#endif
}

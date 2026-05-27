#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mc_common.h>
#include <mc_runtime.h>

using v4f16 = __NATIVE_VECTOR__(4, _Float16);
using v4f32 = __NATIVE_VECTOR__(4, float);

__global__ void qk_tile_kernel(const _Float16* Q, const _Float16* K, float* Scores) {
  const int tid = threadIdx.x;
  if (tid >= 64) {
    return;
  }

  v4f16 q;
  v4f16 kt;
  v4f32 scores = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q_row = tid % 16;
    const int q_dim = (tid / 16) * 4 + i;
    const int k_col = tid % 16;
    const int k_dim = (tid / 16) * 4 + i;
    q[i] = Q[q_row * 16 + q_dim];
    kt[i] = K[k_col * 16 + k_dim];
  }

  v4f32 out = __builtin_mxc_mma_16x16x16f16(q, kt, scores);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = (tid / 16) * 4 + i;
    const int col = tid % 16;
    Scores[row * 16 + col] = out[i];
  }
}

__global__ void pv_tile_kernel(const _Float16* P, const _Float16* V, float* Out) {
  const int tid = threadIdx.x;
  if (tid >= 64) {
    return;
  }

  v4f16 p;
  v4f16 v;
  v4f32 acc = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int p_row = tid % 16;
    const int p_col = (tid / 16) * 4 + i;
    const int v_row = (tid / 16) * 4 + i;
    const int v_col = tid % 16;
    p[i] = P[p_row * 16 + p_col];
    v[i] = V[v_row * 16 + v_col];
  }

  v4f32 out = __builtin_mxc_mma_16x16x16f16(p, v, acc);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = (tid / 16) * 4 + i;
    const int col = tid % 16;
    Out[row * 16 + col] = out[i];
  }
}

static void check_mc(mcError_t err, const char* what) {
  if (err != mcSuccess) {
    fprintf(stderr, "%s failed: %s\n", what, mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

static void matmul_ref(const float* A, const float* B, float* C) {
  for (int m = 0; m < 16; ++m) {
    for (int n = 0; n < 16; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < 16; ++k) {
        acc += A[m * 16 + k] * B[k * 16 + n];
      }
      C[m * 16 + n] = acc;
    }
  }
}

static void qk_ref(const float* Q, const float* K, float* Scores) {
  for (int m = 0; m < 16; ++m) {
    for (int n = 0; n < 16; ++n) {
      float acc = 0.0f;
      for (int d = 0; d < 16; ++d) {
        acc += Q[m * 16 + d] * K[n * 16 + d];
      }
      Scores[m * 16 + n] = acc;
    }
  }
}

static void fill_pattern(float* X, int seed, float scale) {
  for (int i = 0; i < 256; ++i) {
    X[i] = static_cast<float>(((i * seed + 7) % 31) - 15) * scale;
  }
}

static void compare_or_die(const char* name, const float* actual, const float* expected) {
  float max_abs = 0.0f;
  int max_idx = 0;
  for (int i = 0; i < 256; ++i) {
    const float err = fabsf(actual[i] - expected[i]);
    if (err > max_abs) {
      max_abs = err;
      max_idx = i;
    }
  }
  if (max_abs > 2.0e-2f) {
    fprintf(stderr,
            "%s mismatch at row=%d col=%d got=%f expected=%f max_abs=%f\n",
            name,
            max_idx / 16,
            max_idx % 16,
            actual[max_idx],
            expected[max_idx],
            max_abs);
    exit(EXIT_FAILURE);
  }
  printf("%s PASS max_abs=%f\n", name, max_abs);
}

static void run_qk() {
  float h_Q_f32[256];
  float h_K_f32[256];
  float h_ref[256];
  float h_scores[256];
  _Float16 h_Q[256];
  _Float16 h_K[256];

  fill_pattern(h_Q_f32, 3, 0.0625f);
  fill_pattern(h_K_f32, 5, 0.03125f);
  qk_ref(h_Q_f32, h_K_f32, h_ref);

  for (int i = 0; i < 256; ++i) {
    h_Q[i] = static_cast<_Float16>(h_Q_f32[i]);
    h_K[i] = static_cast<_Float16>(h_K_f32[i]);
    h_scores[i] = 0.0f;
  }

  _Float16* d_Q = nullptr;
  _Float16* d_K = nullptr;
  float* d_scores = nullptr;
  check_mc(mcMalloc(reinterpret_cast<void**>(&d_Q), sizeof(h_Q)), "mcMalloc Q");
  check_mc(mcMalloc(reinterpret_cast<void**>(&d_K), sizeof(h_K)), "mcMalloc K");
  check_mc(mcMalloc(reinterpret_cast<void**>(&d_scores), sizeof(h_scores)), "mcMalloc Scores");
  check_mc(mcMemcpy(d_Q, h_Q, sizeof(h_Q), mcMemcpyHostToDevice), "copy Q");
  check_mc(mcMemcpy(d_K, h_K, sizeof(h_K), mcMemcpyHostToDevice), "copy K");
  check_mc(mcMemset(d_scores, 0, sizeof(h_scores)), "clear Scores");

  qk_tile_kernel<<<1, 64, 0, 0>>>(d_Q, d_K, d_scores);
  check_mc(mcGetLastError(), "launch qk");
  check_mc(mcDeviceSynchronize(), "sync qk");
  check_mc(mcMemcpy(h_scores, d_scores, sizeof(h_scores), mcMemcpyDeviceToHost), "copy Scores");

  compare_or_die("QK tile", h_scores, h_ref);
  mcFree(d_Q);
  mcFree(d_K);
  mcFree(d_scores);
}

static void run_pv() {
  float h_P_f32[256];
  float h_V_f32[256];
  float h_ref[256];
  float h_out[256];
  _Float16 h_P[256];
  _Float16 h_V[256];

  fill_pattern(h_P_f32, 7, 0.03125f);
  fill_pattern(h_V_f32, 11, 0.0625f);
  matmul_ref(h_P_f32, h_V_f32, h_ref);

  for (int i = 0; i < 256; ++i) {
    h_P[i] = static_cast<_Float16>(h_P_f32[i]);
    h_V[i] = static_cast<_Float16>(h_V_f32[i]);
    h_out[i] = 0.0f;
  }

  _Float16* d_P = nullptr;
  _Float16* d_V = nullptr;
  float* d_out = nullptr;
  check_mc(mcMalloc(reinterpret_cast<void**>(&d_P), sizeof(h_P)), "mcMalloc P");
  check_mc(mcMalloc(reinterpret_cast<void**>(&d_V), sizeof(h_V)), "mcMalloc V");
  check_mc(mcMalloc(reinterpret_cast<void**>(&d_out), sizeof(h_out)), "mcMalloc Out");
  check_mc(mcMemcpy(d_P, h_P, sizeof(h_P), mcMemcpyHostToDevice), "copy P");
  check_mc(mcMemcpy(d_V, h_V, sizeof(h_V), mcMemcpyHostToDevice), "copy V");
  check_mc(mcMemset(d_out, 0, sizeof(h_out)), "clear Out");

  pv_tile_kernel<<<1, 64, 0, 0>>>(d_P, d_V, d_out);
  check_mc(mcGetLastError(), "launch pv");
  check_mc(mcDeviceSynchronize(), "sync pv");
  check_mc(mcMemcpy(h_out, d_out, sizeof(h_out), mcMemcpyDeviceToHost), "copy Out");

  compare_or_die("PV tile", h_out, h_ref);
  mcFree(d_P);
  mcFree(d_V);
  mcFree(d_out);
}

int main() {
  run_qk();
  run_pv();
  printf("MACA attention tile probe PASS\n");
  return EXIT_SUCCESS;
}

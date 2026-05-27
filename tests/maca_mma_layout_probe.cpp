#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mc_common.h>
#include <mc_runtime.h>

using v4f16 = __NATIVE_VECTOR__(4, _Float16);
using v4f32 = __NATIVE_VECTOR__(4, float);

__global__ void mma_16x16x16_kernel(const _Float16* A, const _Float16* B, float* C) {
  const int tid = threadIdx.x;
  if (tid >= 64) {
    return;
  }

  v4f16 a;
  v4f16 b;
  v4f32 c = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int a_row = tid % 16;
    const int a_col = (tid / 16) * 4 + i;
    const int b_row = (tid / 16) * 4 + i;
    const int b_col = tid % 16;
    a[i] = A[a_row * 16 + a_col];
    b[i] = B[b_row * 16 + b_col];
  }

  v4f32 d = __builtin_mxc_mma_16x16x16f16(a, b, c);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c_row = (tid / 16) * 4 + i;
    const int c_col = tid % 16;
    C[c_row * 16 + c_col] = d[i];
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

static void fill_case(int test_case, float* A, float* B) {
  for (int i = 0; i < 256; ++i) {
    A[i] = 0.0f;
    B[i] = 0.0f;
  }

  if (test_case == 0) {
    for (int i = 0; i < 256; ++i) {
      A[i] = 1.0f;
      B[i] = 1.0f;
    }
    return;
  }

  if (test_case == 1) {
    for (int i = 0; i < 16; ++i) {
      A[i * 16 + i] = 1.0f;
    }
    for (int i = 0; i < 256; ++i) {
      B[i] = static_cast<float>((i % 17) - 8) * 0.125f;
    }
    return;
  }

  if (test_case == 2) {
    for (int i = 0; i < 256; ++i) {
      A[i] = static_cast<float>((i % 19) - 9) * 0.125f;
    }
    for (int i = 0; i < 16; ++i) {
      B[i * 16 + i] = 1.0f;
    }
    return;
  }

  for (int i = 0; i < 256; ++i) {
    A[i] = static_cast<float>((i % 23) - 11) * 0.0625f;
    B[i] = static_cast<float>((i % 29) - 14) * 0.03125f;
  }
}

static void run_case(int test_case) {
  float h_A_f32[256];
  float h_B_f32[256];
  float h_ref[256];
  float h_C[256];
  _Float16 h_A[256];
  _Float16 h_B[256];

  fill_case(test_case, h_A_f32, h_B_f32);
  matmul_ref(h_A_f32, h_B_f32, h_ref);

  for (int i = 0; i < 256; ++i) {
    h_A[i] = static_cast<_Float16>(h_A_f32[i]);
    h_B[i] = static_cast<_Float16>(h_B_f32[i]);
    h_C[i] = 0.0f;
  }

  _Float16* d_A = nullptr;
  _Float16* d_B = nullptr;
  float* d_C = nullptr;
  check_mc(mcMalloc(reinterpret_cast<void**>(&d_A), sizeof(h_A)), "mcMalloc A");
  check_mc(mcMalloc(reinterpret_cast<void**>(&d_B), sizeof(h_B)), "mcMalloc B");
  check_mc(mcMalloc(reinterpret_cast<void**>(&d_C), sizeof(h_C)), "mcMalloc C");
  check_mc(mcMemcpy(d_A, h_A, sizeof(h_A), mcMemcpyHostToDevice), "copy A");
  check_mc(mcMemcpy(d_B, h_B, sizeof(h_B), mcMemcpyHostToDevice), "copy B");
  check_mc(mcMemset(d_C, 0, sizeof(h_C)), "clear C");

  mma_16x16x16_kernel<<<1, 64, 0, 0>>>(d_A, d_B, d_C);
  check_mc(mcGetLastError(), "launch");
  check_mc(mcDeviceSynchronize(), "sync");
  check_mc(mcMemcpy(h_C, d_C, sizeof(h_C), mcMemcpyDeviceToHost), "copy C");

  float max_abs = 0.0f;
  int max_idx = 0;
  for (int i = 0; i < 256; ++i) {
    const float err = fabsf(h_C[i] - h_ref[i]);
    if (err > max_abs) {
      max_abs = err;
      max_idx = i;
    }
  }

  mcFree(d_A);
  mcFree(d_B);
  mcFree(d_C);

  if (max_abs > 2.0e-2f) {
    fprintf(stderr,
            "case %d mismatch at row=%d col=%d got=%f expected=%f max_abs=%f\n",
            test_case,
            max_idx / 16,
            max_idx % 16,
            h_C[max_idx],
            h_ref[max_idx],
            max_abs);
    exit(EXIT_FAILURE);
  }

  printf("case %d PASS max_abs=%f\n", test_case, max_abs);
}

int main() {
  for (int test_case = 0; test_case < 4; ++test_case) {
    run_case(test_case);
  }
  printf("MACA MMA layout probe PASS\n");
  return EXIT_SUCCESS;
}

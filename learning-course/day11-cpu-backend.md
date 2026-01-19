# Day 11: CPU 后端与 SIMD 优化

## 课程目标

掌握 CPU 后端的优化技术：
- SIMD（Single Instruction Multiple Data）原理
- AVX2/AVX-512 优化
- ARM NEON 优化
- 矩阵乘法优化内核
- 量化计算加速

## 1. SIMD 基础

### 1.1 什么是 SIMD？

**标量计算**（传统方式）：
```cpp
// 一次处理一个元素
for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
}
// 4 次加法需要 4 个 CPU 周期
```

**SIMD 计算**：
```cpp
// 一次处理多个元素
__m256 va = _mm256_loadu_ps(&a[i]);     // 加载 8 个 float
__m256 vb = _mm256_loadu_ps(&b[i]);     // 加载 8 个 float
__m256 vc = _mm256_add_ps(va, vb);      // 同时计算 8 个加法
_mm256_storeu_ps(&c[i], vc);            // 存储 8 个 float

// 8 次加法只需 1 个 CPU 周期 → 8x 加速
```

### 1.2 主要 SIMD 指令集

| 指令集 | 寄存器宽度 | float 数量 | 平台 |
|--------|-----------|-----------|------|
| **SSE** | 128-bit | 4 | x86 (老) |
| **AVX** | 256-bit | 8 | x86 (现代) |
| **AVX2** | 256-bit | 8 + 整数 | x86 (推荐) |
| **AVX-512** | 512-bit | 16 | x86 (服务器) |
| **NEON** | 128-bit | 4 | ARM (手机/Mac) |
| **SVE** | 可变 | 可变 | ARM (服务器) |

## 2. AVX2 向量加法示例

### 2.1 标量版本

```c
// 位置：ggml/src/ggml-cpu/vec-dot.c
void ggml_vec_add_f32(int n, float * z, const float * x, const float * y) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}
// 性能: ~0.25 GFLOPS
```

### 2.2 AVX2 版本

```c
// 位置：ggml/src/ggml-cpu/vec-dot.c:256
void ggml_vec_add_f32_avx2(int n, float * z, const float * x, const float * y) {
    int i = 0;

    // 处理 32 个元素为一组（AVX2 可以同时处理 8 个 float）
    for (; i + 32 <= n; i += 32) {
        // 加载 8 个 float × 4 = 32 个元素
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);

        __m256 y0 = _mm256_loadu_ps(y + i);
        __m256 y1 = _mm256_loadu_ps(y + i + 8);
        __m256 y2 = _mm256_loadu_ps(y + i + 16);
        __m256 y3 = _mm256_loadu_ps(y + i + 24);

        // 向量加法
        __m256 z0 = _mm256_add_ps(x0, y0);
        __m256 z1 = _mm256_add_ps(x1, y1);
        __m256 z2 = _mm256_add_ps(x2, y2);
        __m256 z3 = _mm256_add_ps(x3, y3);

        // 存储结果
        _mm256_storeu_ps(z + i, z0);
        _mm256_storeu_ps(z + i + 8, z1);
        _mm256_storeu_ps(z + i + 16, z2);
        _mm256_storeu_ps(z + i + 24, z3);
    }

    // 处理剩余元素
    for (; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}
// 性能: ~2.0 GFLOPS → 8x 加速
```

## 3. 矩阵乘法优化

### 3.1 朴素实现

```c
void matmul_naive(int M, int N, int K,
                  const float * A,  // [M, K]
                  const float * B,  // [K, N]
                  float * C) {      // [M, N]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
// 性能: ~5 GFLOPS
// 问题: 缓存未命中、无 SIMD
```

### 3.2 分块 + AVX2 优化

```c
// 位置：ggml/src/ggml-cpu/ggml-cpu.c:2847
void ggml_compute_forward_mul_mat_f32(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];  // A
    const struct ggml_tensor * src1 = dst->src[1];  // B

    const int M = src0->ne[1];
    const int N = src1->ne[1];
    const int K = src0->ne[0];

    // 分块大小（针对 L1 缓存优化）
    const int MC = 128;   // M 方向
    const int NC = 4096;  // N 方向
    const int KC = 512;   // K 方向

    for (int ic = 0; ic < M; ic += MC) {
        const int mc = MIN(MC, M - ic);

        for (int jc = 0; jc < N; jc += NC) {
            const int nc = MIN(NC, N - jc);

            for (int pc = 0; pc < K; pc += KC) {
                const int kc = MIN(KC, K - pc);

                // 微内核：AVX2 优化的小矩阵乘法
                matmul_kernel_avx2(
                    mc, nc, kc,
                    src0->data + ic * K + pc,
                    src1->data + pc * N + jc,
                    dst->data + ic * N + jc);
            }
        }
    }
}

// 微内核实现
static void matmul_kernel_avx2(
    int M, int N, int K,
    const float * A,
    const float * B,
    float * C) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 8) {  // 每次处理 8 列
            __m256 sum = _mm256_setzero_ps();

            for (int k = 0; k < K; k++) {
                __m256 a = _mm256_set1_ps(A[i * K + k]);     // 广播 A 元素
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);   // 加载 B 的 8 个元素
                sum = _mm256_fmadd_ps(a, b, sum);            // FMA: sum += a * b
            }

            // 累加到 C
            __m256 c = _mm256_loadu_ps(&C[i * N + j]);
            c = _mm256_add_ps(c, sum);
            _mm256_storeu_ps(&C[i * N + j], c);
        }
    }
}
// 性能: ~50 GFLOPS → 10x 加速
```

**优化技巧**：
1. **分块**：利用 L1/L2 缓存
2. **FMA**：融合乘加指令
3. **循环展开**：减少分支
4. **向量化**：AVX2 同时处理 8 个元素

## 4. 量化矩阵乘法

### 4.1 Q4_0 格式

```c
// Q4_0: 32 个 float 压缩到 18 字节
typedef struct {
    ggml_fp16_t d;        // 缩放因子 (2 字节)
    uint8_t qs[16];       // 量化值 (16 字节, 每个 4-bit)
} block_q4_0;

// 32 个 float (128 字节) → 18 字节
// 压缩比: 7.1x
```

### 4.2 Q4_0 点积（AVX2）

```c
// 位置：ggml/src/ggml-quants.c:1234
void ggml_vec_dot_q4_0_q8_0_avx2(int n, float * s, const void * vx, const void * vy) {
    const block_q4_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

    __m256 acc = _mm256_setzero_ps();

    // 每次处理一个块 (32 个元素)
    const int nb = n / 32;
    for (int i = 0; i < nb; i++) {
        // 1. 反量化 x (4-bit → float)
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[i].qs);

        // 分离低 4 位和高 4 位
        __m256i qx_lo = _mm256_and_si256(qx, _mm256_set1_epi8(0x0F));
        __m256i qx_hi = _mm256_and_si256(_mm256_srli_epi16(qx, 4), _mm256_set1_epi8(0x0F));

        // 转换为 int8
        qx_lo = _mm256_sub_epi8(qx_lo, _mm256_set1_epi8(8));
        qx_hi = _mm256_sub_epi8(qx_hi, _mm256_set1_epi8(8));

        // 2. 加载 y (int8)
        __m256i qy_lo = _mm256_loadu_si256((const __m256i *)&y[i].qs[0]);
        __m256i qy_hi = _mm256_loadu_si256((const __m256i *)&y[i].qs[16]);

        // 3. int8 乘法
        __m256i p_lo = _mm256_maddubs_epi16(qx_lo, qy_lo);  // 16 个 int16
        __m256i p_hi = _mm256_maddubs_epi16(qx_hi, qy_hi);

        // 4. 累加
        __m256i p = _mm256_add_epi16(p_lo, p_hi);
        __m256i sum = _mm256_madd_epi16(p, _mm256_set1_epi16(1));  // 8 个 int32

        // 5. 转换为 float 并缩放
        __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));
        __m256 pf = _mm256_cvtepi32_ps(sum);
        acc = _mm256_fmadd_ps(d, pf, acc);
    }

    // 水平求和
    *s = hsum_float_8(acc);
}
// 性能: ~80 GFLOPS (Q4_0 vs F32)
// 内存带宽: 7x 减少
```

## 5. ARM NEON 优化

### 5.1 NEON 向量加法

```c
// 位置：ggml/src/ggml-cpu/vec-dot.c:789
void ggml_vec_add_f32_neon(int n, float * z, const float * x, const float * y) {
    int i = 0;

    // 处理 16 个元素为一组
    for (; i + 16 <= n; i += 16) {
        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);

        float32x4_t y0 = vld1q_f32(y + i);
        float32x4_t y1 = vld1q_f32(y + i + 4);
        float32x4_t y2 = vld1q_f32(y + i + 8);
        float32x4_t y3 = vld1q_f32(y + i + 12);

        float32x4_t z0 = vaddq_f32(x0, y0);
        float32x4_t z1 = vaddq_f32(x1, y1);
        float32x4_t z2 = vaddq_f32(x2, y2);
        float32x4_t z3 = vaddq_f32(x3, y3);

        vst1q_f32(z + i, z0);
        vst1q_f32(z + i + 4, z1);
        vst1q_f32(z + i + 8, z2);
        vst1q_f32(z + i + 12, z3);
    }

    // 处理剩余元素
    for (; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}
```

### 5.2 NEON vs AVX2 对比

| 特性 | ARM NEON | x86 AVX2 |
|------|----------|----------|
| **寄存器宽度** | 128-bit | 256-bit |
| **float 数量** | 4 | 8 |
| **int8 乘法** | vmlal_s8 | _mm256_maddubs_epi16 |
| **FMA** | vfmaq_f32 (ARMv8.2+) | _mm256_fmadd_ps |
| **性能** | ~20 GFLOPS | ~40 GFLOPS |
| **平台** | Mac M1/M2, 手机 | Intel/AMD CPU |

## 6. 性能调优技巧

### 6.1 编译器优化标志

```bash
# GCC/Clang
-O3                      # 最高优化级别
-march=native            # 针对当前 CPU 优化
-mavx2                   # 启用 AVX2
-mfma                    # 启用 FMA 指令
-ffast-math              # 快速数学（牺牲一些精度）

# 完整示例
gcc -O3 -march=native -mavx2 -mfma -ffast-math \
    -o ggml ggml.c -lm -lpthread
```

### 6.2 性能分析

```cpp
#include <chrono>

void benchmark_operation() {
    auto start = std::chrono::high_resolution_clock::now();

    // 执行操作
    ggml_vec_dot_f32(n, result, x, y);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 计算 GFLOPS
    double ops = 2.0 * n;  // n 次乘法 + n 次加法
    double gflops = ops / (duration.count() * 1e3);

    printf("Performance: %.2f GFLOPS\n", gflops);
}
```

### 6.3 内存对齐

```c
// 确保数据对齐到 32 字节（AVX2）
#define ALIGNED_32 __attribute__((aligned(32)))

// 分配对齐内存
float * data = (float *)aligned_alloc(32, n * sizeof(float));

// 使用对齐加载（更快）
__m256 v = _mm256_load_ps(data);  // 而非 _mm256_loadu_ps
```

## 7. 实战：优化自己的代码

### 7.1 向量化 ReLU

```c
// 标量版本
void relu_scalar(int n, float * x) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}

// AVX2 版本
void relu_avx2(int n, float * x) {
    __m256 zero = _mm256_setzero_ps();

    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_max_ps(v, zero);  // max(v, 0)
        _mm256_storeu_ps(x + i, v);
    }

    // 剩余元素
    for (int i = n & ~7; i < n; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}
```

### 7.2 向量化 Softmax

```c
void softmax_avx2(int n, float * x) {
    // 1. 找最大值
    __m256 max_vec = _mm256_set1_ps(-INFINITY);
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        max_vec = _mm256_max_ps(max_vec, v);
    }
    float max_val = hmax_f32_8(max_vec);

    // 2. 计算 exp 并累加
    __m256 max_bc = _mm256_set1_ps(max_val);
    __m256 sum_vec = _mm256_setzero_ps();

    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_sub_ps(v, max_bc);           // x - max
        v = _mm256_exp_ps(v);                   // exp(x - max)
        sum_vec = _mm256_add_ps(sum_vec, v);
        _mm256_storeu_ps(x + i, v);
    }
    float sum = hsum_f32_8(sum_vec);

    // 3. 归一化
    __m256 sum_bc = _mm256_set1_ps(sum);
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_div_ps(v, sum_bc);
        _mm256_storeu_ps(x + i, v);
    }
}
```

## 8. 总结

今天我们学习了 CPU 后端优化：

✅ **SIMD 基础**：AVX2/NEON 向量化
✅ **矩阵乘法**：分块 + 微内核优化
✅ **量化加速**：Q4_0 SIMD 实现
✅ **ARM 优化**：NEON 指令集
✅ **性能调优**：编译器优化、对齐、Benchmark

### 关键要点

1. **SIMD 是 CPU 性能的关键**：8-16x 加速
2. **分块优化缓存**：利用 L1/L2 缓存
3. **FMA 指令**：融合乘加提升性能
4. **量化 + SIMD**：内存带宽和计算双重加速

## 下一步

明天我们将学习 **Day 12: GPU 后端实现**：
- CUDA 编程基础
- cuBLAS 集成
- 自定义 CUDA 内核
- Metal/Vulkan 对比

---

**练习**：
1. 实现一个 AVX2 优化的向量点积
2. 对比标量和 SIMD 版本的性能差异
3. 使用 perf 分析你的代码瓶颈

📚 [Day 12: GPU 后端实现](day12-gpu-backend.md)

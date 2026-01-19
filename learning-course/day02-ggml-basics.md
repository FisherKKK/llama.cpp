# Day 2: GGML张量库基础

> 🎯 **学习目标**：深入理解GGML张量的数据结构、内存布局、以及基本操作原理。

## 1. GGML概述

### 1.1 什么是GGML？

GGML (Georgi Gerganov Machine Learning) 是专为机器学习推理设计的C语言张量库：

```
GGML的核心特点
┌─────────────────────────────────────────┐
│ 1. 轻量级                                │
│    • 纯C实现，零外部依赖                 │
│    • 核心代码 ~18000行                   │
│                                         │
│ 2. 高效                                  │
│    • SIMD优化（AVX, NEON）              │
│    • 多后端支持（CPU/GPU）               │
│    • 内存高效（量化支持）                │
│                                         │
│ 3. 推理优化                              │
│    • 只需前向传播                        │
│    • 计算图模式                          │
│    • 内存复用                            │
│                                         │
│ 4. 灵活                                  │
│    • 动态图和静态图                      │
│    • 支持4维张量                         │
│    • 多种数据类型                        │
└─────────────────────────────────────────┘
```

### 1.2 核心文件

```
ggml/
├── include/
│   ├── ggml.h           # 主API (102KB, ~3000行)
│   ├── ggml-backend.h   # 后端抽象
│   └── ggml-alloc.h     # 内存分配器
│
└── src/
    ├── ggml.c           # 核心实现 (242KB, ~18000行)
    ├── ggml-backend.cpp # 后端管理
    ├── ggml-alloc.c     # 分配器实现
    └── ggml-quants.c    # 量化函数 (217KB)
```

## 2. 张量数据结构

### 2.1 ggml_tensor结构体

```c
// 位置: ggml/include/ggml.h
struct ggml_tensor {
    enum ggml_type type;       // 数据类型

    GGML_DEPRECATED(enum ggml_backend_type backend, "use buffer");
    enum ggml_backend_type backend;

    struct ggml_backend_buffer * buffer;

    int64_t ne[GGML_MAX_DIMS];  // number of elements (每个维度的元素数)
    size_t  nb[GGML_MAX_DIMS];  // stride in bytes (每个维度的字节步长)

    // 操作相关
    enum ggml_op op;            // 操作类型

    // op params - 操作参数
    // allocated as int32_t for alignment
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

    int32_t flags;

    struct ggml_tensor * src[GGML_MAX_SRC];  // 源张量（用于构建计算图）

    // 性能统计
    int     perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;

    struct ggml_tensor * view_src;  // 视图源（如果是视图）
    size_t               view_offs; // 视图偏移

    void * data;                    // 数据指针

    char name[GGML_MAX_NAME];       // 名称（用于调试）

    void * extra;                   // 额外数据

    // char padding[8];
};
```

### 2.2 关键字段解析

#### ne - Number of Elements (元素数量)

```c
// 例子：一个 3x4x5 的三维张量
int64_t ne[4] = {
    3,  // ne[0]: 第0维（最内层）有3个元素
    4,  // ne[1]: 第1维有4个元素
    5,  // ne[2]: 第2维有5个元素
    1   // ne[3]: 第3维（未使用）
};

// 维度的逻辑：
// ne[0] -> 列（columns）
// ne[1] -> 行（rows）
// ne[2] -> 深度（depth）
// ne[3] -> 批次（batch）
```

可视化：
```
     ne[2]=5 (最外层，5个矩阵)
      ↓
    ┌─────┐
    │ 4x3 │  ← ne[1]=4 行, ne[0]=3 列
    │ 4x3 │
    │ 4x3 │
    │ 4x3 │
    │ 4x3 │
    └─────┘
```

#### nb - Stride in Bytes (字节步长)

```c
// nb定义了如何在内存中导航
size_t nb[4] = {
    sizeof(float),           // nb[0]: 移动1个元素的字节数
    nb[0] * ne[0],          // nb[1]: 移动1行的字节数
    nb[1] * ne[1],          // nb[2]: 移动1个矩阵的字节数
    nb[2] * ne[2]           // nb[3]: 移动1个批次的字节数
};
```

**为什么需要nb？**
- 支持非连续内存布局（转置、切片）
- 实现零拷贝视图
- 允许不同的内存对齐

#### 内存布局示例

```c
// 2x3矩阵的内存布局
// [[1, 2, 3],
//  [4, 5, 6]]

// 连续布局（行主序）：
float data[] = {1, 2, 3, 4, 5, 6};
ne[0] = 3, ne[1] = 2
nb[0] = 4 (sizeof(float))
nb[1] = 12 (3 * sizeof(float))

// 访问 data[i][j]:
float* ptr = (float*)((char*)data + i * nb[1] + j * nb[0]);

// 例如访问 data[1][2] (值为6):
ptr = (float*)((char*)data + 1*12 + 2*4) = data + 5  -> 6 ✓
```

#### 转置示例

```c
// 原始矩阵 A (2x3):
// [[1, 2, 3],
//  [4, 5, 6]]
A.ne[0] = 3, A.ne[1] = 2
A.nb[0] = 4, A.nb[1] = 12

// 转置后 A^T (3x2)，但不复制数据：
// [[1, 4],
//  [2, 5],
//  [3, 6]]
At.ne[0] = 2, At.ne[1] = 3      // 交换维度
At.nb[0] = 12, At.nb[1] = 4     // 交换步长
At.data = A.data                // 共享数据

// 访问 At[1][0] (应该是2):
ptr = (float*)((char*)At.data + 1*4 + 0*12) = data + 1 -> 2 ✓
```

### 2.3 数据类型

```c
enum ggml_type {
    GGML_TYPE_F32  = 0,   // float32 (4 bytes)
    GGML_TYPE_F16  = 1,   // float16 (2 bytes)
    GGML_TYPE_Q4_0 = 2,   // 4-bit quantization (0.5625 bytes/element)
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,   // 8-bit quantization (1.125 bytes/element)
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    // ... 更多量化类型
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_I64,
    GGML_TYPE_F64,
    GGML_TYPE_COUNT,
};
```

**量化的重要性**：
```
7B模型参数量化对比：
┌──────────┬────────────┬──────────────┬──────────┐
│ 类型     │ 大小/参数  │ 总大小       │ 质量     │
├──────────┼────────────┼──────────────┼──────────┤
│ F32      │ 4 bytes    │ ~28 GB       │ 基准     │
│ F16      │ 2 bytes    │ ~14 GB       │ 99.9%    │
│ Q8_0     │ 1.125 bytes│ ~8 GB        │ 99%      │
│ Q6_K     │ 0.875 bytes│ ~6 GB        │ 98%      │
│ Q5_K     │ 0.6875bytes│ ~5 GB        │ 96%      │
│ Q4_K     │ 0.5625bytes│ ~4 GB        │ 94%      │
│ Q3_K     │ 0.4375bytes│ ~3 GB        │ 90%      │
│ Q2_K     │ 0.3125bytes│ ~2 GB        │ 85%      │
└──────────┴────────────┴──────────────┴──────────┘

影响：
• 更小的内存占用 -> 可以运行更大的模型
• 更快的加载速度 -> 减少I/O时间
• 更快的推理速度 -> 减少内存带宽需求
```

## 3. 上下文与内存管理

### 3.1 ggml_context

```c
// GGML的内存管理单元
struct ggml_context {
    size_t mem_size;        // 内存池大小
    void * mem_buffer;      // 内存池指针
    bool   mem_buffer_owned;// 是否拥有内存
    bool   no_alloc;        // 是否禁止分配（用于只构建图）

    int    n_objects;       // 对象数量

    struct ggml_object * objects_begin;  // 对象链表
    struct ggml_object * objects_end;

    struct ggml_scratch scratch;         // 临时内存
    struct ggml_scratch scratch_save;
};
```

### 3.2 内存分配流程

```c
// 1. 创建上下文
struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,  // 16MB内存池
    .mem_buffer = NULL,          // 让GGML分配
    .no_alloc   = false,
};
struct ggml_context * ctx = ggml_init(params);

// 内存布局：
┌────────────────────────────────────────┐
│  GGML Context (16MB)                   │
├────────────────────────────────────────┤
│  ╔════════════════════════════════════╗│
│  ║  Object 1: ggml_tensor + metadata ║│
│  ╚════════════════════════════════════╝│
│  ╔════════════════════════════════════╗│
│  ║  Data 1: tensor data               ║│
│  ╚════════════════════════════════════╝│
│  ╔════════════════════════════════════╗│
│  ║  Object 2: ggml_tensor + metadata ║│
│  ╚════════════════════════════════════╝│
│  ╔════════════════════════════════════╗│
│  ║  Data 2: tensor data               ║│
│  ╚════════════════════════════════════╝│
│  ...                                   │
│  ╔════════════════════════════════════╗│
│  ║  Free space                        ║│
│  ╚════════════════════════════════════╝│
└────────────────────────────────────────┘
```

### 3.3 张量创建

```c
// 创建1D张量
struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
// 结果：shape = [10], data = uninitialized

// 创建2D张量
struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4);
// 结果：shape = [3, 4], 即 4行3列

// 创建3D张量
struct ggml_tensor * c = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 3, 4);
// 结果：shape = [2, 3, 4]

// 创建4D张量
struct ggml_tensor * d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 3, 4, 5);
// 结果：shape = [2, 3, 4, 5]
```

## 4. 基本张量操作

### 4.1 元素访问

```c
// 设置/获取单个元素
float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
void  ggml_set_f32_1d(struct ggml_tensor * tensor, int i, float value);

// 二维张量
float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, ...);

// 手动访问（理解stride）
float get_element_2d(struct ggml_tensor * t, int i, int j) {
    return *(float*)((char*)t->data + i * t->nb[1] + j * t->nb[0]);
}
```

### 4.2 常用操作

```c
// 位置: ggml/include/ggml.h

// 1. 矩阵乘法（最重要！）
struct ggml_tensor * ggml_mul_mat(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,    // [K, M]
    struct ggml_tensor  * b);   // [K, N]
// 结果: [M, N]
// 注意：这是 C^T = A*B^T, 即 C = B*A^T

// 2. 逐元素操作
struct ggml_tensor * ggml_add(ctx, a, b);      // 加法
struct ggml_tensor * ggml_mul(ctx, a, b);      // 乘法
struct ggml_tensor * ggml_div(ctx, a, b);      // 除法
struct ggml_tensor * ggml_sub(ctx, a, b);      // 减法

// 3. 激活函数
struct ggml_tensor * ggml_relu(ctx, a);        // ReLU
struct ggml_tensor * ggml_gelu(ctx, a);        // GELU
struct ggml_tensor * ggml_silu(ctx, a);        // SiLU (Swish)
struct ggml_tensor * ggml_tanh(ctx, a);        // Tanh

// 4. Softmax
struct ggml_tensor * ggml_soft_max(ctx, a);
struct ggml_tensor * ggml_soft_max_ext(ctx, a, mask, scale, max_bias);

// 5. 归一化
struct ggml_tensor * ggml_norm(ctx, a, eps);              // Layer Norm
struct ggml_tensor * ggml_rms_norm(ctx, a, eps);          // RMS Norm

// 6. 形状操作
struct ggml_tensor * ggml_reshape_2d(ctx, a, ne0, ne1);  // 重塑
struct ggml_tensor * ggml_view_2d(ctx, a, ne0, ne1, nb1, offset);  // 视图
struct ggml_tensor * ggml_permute(ctx, a, axis0, axis1, axis2, axis3);  // 置换
struct ggml_tensor * ggml_transpose(ctx, a);             // 转置

// 7. 位置编码
struct ggml_tensor * ggml_rope(ctx, a, n_past, n_dims, mode, n_ctx);
struct ggml_tensor * ggml_rope_ext(ctx, a, b, n_dims, ...);  // 扩展RoPE

// 8. 注意力相关
struct ggml_tensor * ggml_flash_attn_ext(ctx, q, k, v, mask, scale, ...);
```

### 4.3 矩阵乘法详解

这是最重要的操作，需要特别理解：

```c
// GGML的矩阵乘法约定（与常见约定不同！）
// C = ggml_mul_mat(A, B) 实际计算：C^T = A * B^T
// 即：C = B * A^T

示例：
A: [2, 3]  (3行2列)
    [[1, 2],
     [3, 4],
     [5, 6]]

B: [2, 4]  (4行2列)
    [[7, 8],
     [9, 10],
     [11, 12],
     [13, 14]]

C = ggml_mul_mat(A, B)
C: [3, 4]  (4行3列)

实际计算过程：
1. A保持不变: [2, 3]
2. B转置: [4, 2] -> [2, 4]
3. 矩阵乘法: [2, 3] × [2, 4] = [3, 4]

结果：
C[i, j] = sum_k(A[k, i] * B[k, j])

为什么这样设计？
- 符合内存布局（列主序）
- 优化缓存访问
- 减少数据移动
```

## 5. 计算图基础

### 5.1 什么是计算图？

```
计算图是张量操作的有向无环图（DAG）：

输入张量 (a, b)
     │      │
     └──┬───┘
        │
     ╔══▼══╗
     ║ Add ║ <- 操作节点
     ╚══╤══╝
        │
     ╔══▼══╗
     ║  c  ║ <- 中间结果
     ╚══╤══╝
        │
     ╔══▼══╗
     ║ Mul ║
     ╚══╤══╝
        │
     ╔══▼══╗
     ║  d  ║ <- 最终结果
     ╚═════╝

优点：
1. 延迟执行：构建图时不计算，执行时才计算
2. 优化机会：可以融合操作、重排序
3. 并行化：可以并行执行独立的操作
4. 内存复用：可以提前规划内存使用
```

### 5.2 构建计算图

```c
// 创建上下文
struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
    .no_alloc   = false,
};
struct ggml_context * ctx = ggml_init(params);

// 定义输入
struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);

// 设置为参数（需要梯度）
ggml_set_param(ctx, a);

// 构建计算图（只是定义操作，不计算）
struct ggml_tensor * c = ggml_add(ctx, a, b);        // c = a + b
struct ggml_tensor * d = ggml_mul(ctx, c, c);        // d = c * c
struct ggml_tensor * e = ggml_sum(ctx, d);           // e = sum(d)

// 创建计算图对象
struct ggml_cgraph * graph = ggml_new_graph(ctx);
ggml_build_forward_expand(graph, e);  // 从e开始反向构建图

// 设置输入数据
for (int i = 0; i < 10; i++) {
    ggml_set_f32_1d(a, i, i);
    ggml_set_f32_1d(b, i, i * 2);
}

// 执行计算图
ggml_graph_compute_with_ctx(ctx, graph, /*n_threads=*/4);

// 获取结果
float result = ggml_get_f32_1d(e, 0);
printf("Result: %f\n", result);

// 清理
ggml_free(ctx);
```

### 5.3 计算图结构

```c
struct ggml_cgraph {
    int size;           // 图的容量
    int n_nodes;        // 节点数量
    int n_leafs;        // 叶子节点数量（输入）

    struct ggml_tensor ** nodes;  // 操作节点列表
    struct ggml_tensor ** grads;  // 梯度（用于反向传播，推理时不用）
    struct ggml_tensor ** leafs;  // 叶子节点列表

    struct ggml_hash_set visited_hash_set;  // 访问标记

    enum ggml_cgraph_eval_order order;      // 执行顺序

    // 性能统计
    int     perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;
};
```

## 6. 实践练习

### 练习1：创建和操作张量

```c
#include "ggml.h"
#include <stdio.h>

int main() {
    // 初始化
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };
    struct ggml_context * ctx = ggml_init(params);

    // 练习1：创建2x3矩阵
    struct ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);

    // 练习2：填充数据
    // [[1, 2, 3],
    //  [4, 5, 6]]
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            ggml_set_f32_nd(m, j, i, 0, 0, i * 3 + j + 1);
        }
    }

    // 练习3：打印矩阵
    printf("Matrix M (2x3):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%.1f ", ggml_get_f32_nd(m, j, i, 0, 0));
        }
        printf("\n");
    }

    // 练习4：理解内存布局
    printf("\nMemory layout:\n");
    printf("ne[0]=%lld, ne[1]=%lld\n", m->ne[0], m->ne[1]);
    printf("nb[0]=%zu, nb[1]=%zu\n", m->nb[0], m->nb[1]);

    // 练习5：手动访问
    float* data = (float*)m->data;
    printf("\nRaw data: ");
    for (int i = 0; i < 6; i++) {
        printf("%.1f ", data[i]);
    }
    printf("\n");

    ggml_free(ctx);
    return 0;
}
```

### 练习2：矩阵乘法

```c
#include "ggml.h"
#include <stdio.h>

int main() {
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };
    struct ggml_context * ctx = ggml_init(params);

    // A: 2x3 矩阵
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    // [[1, 2, 3],
    //  [4, 5, 6]]
    float a_data[] = {1, 2, 3, 4, 5, 6};
    memcpy(a->data, a_data, sizeof(a_data));

    // B: 2x2 矩阵
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    // [[7, 8],
    //  [9, 10]]
    float b_data[] = {7, 9, 8, 10};  // 注意：列主序
    memcpy(b->data, b_data, sizeof(b_data));

    // C = ggml_mul_mat(A, B)
    struct ggml_tensor * c = ggml_mul_mat(ctx, a, b);

    // 构建并执行计算图
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // 打印结果
    printf("Result C (shape: %lld x %lld):\n", c->ne[0], c->ne[1]);
    for (int i = 0; i < c->ne[1]; i++) {
        for (int j = 0; j < c->ne[0]; j++) {
            printf("%.1f ", ggml_get_f32_nd(c, j, i, 0, 0));
        }
        printf("\n");
    }

    ggml_free(ctx);
    return 0;
}
```

### 练习3：构建简单神经网络

```c
#include "ggml.h"
#include <stdio.h>
#include <math.h>

// 简单的两层神经网络
// input (4) -> hidden (8) -> output (2)

int main() {
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
    };
    struct ggml_context * ctx = ggml_init(params);

    // 输入
    struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);

    // 第一层权重和偏置
    struct ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
    struct ggml_tensor * b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);

    // 第二层权重和偏置
    struct ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    struct ggml_tensor * b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);

    // 前向传播
    // hidden = relu(input @ w1 + b1)
    struct ggml_tensor * h1 = ggml_mul_mat(ctx, w1, input);
    struct ggml_tensor * h2 = ggml_add(ctx, h1, b1);
    struct ggml_tensor * hidden = ggml_relu(ctx, h2);

    // output = hidden @ w2 + b2
    struct ggml_tensor * o1 = ggml_mul_mat(ctx, w2, hidden);
    struct ggml_tensor * output = ggml_add(ctx, o1, b2);

    // 初始化权重（简单随机）
    // 实际应该用Xavier或He初始化
    for (int i = 0; i < ggml_nelements(w1); i++) {
        ((float*)w1->data)[i] = (float)rand() / RAND_MAX * 0.1f;
    }
    for (int i = 0; i < ggml_nelements(w2); i++) {
        ((float*)w2->data)[i] = (float)rand() / RAND_MAX * 0.1f;
    }
    memset(b1->data, 0, ggml_nbytes(b1));
    memset(b2->data, 0, ggml_nbytes(b2));

    // 设置输入
    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(input->data, input_data, sizeof(input_data));

    // 构建并执行
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // 输出结果
    printf("Output:\n");
    for (int i = 0; i < 2; i++) {
        printf("  %.4f\n", ggml_get_f32_1d(output, i));
    }

    ggml_free(ctx);
    return 0;
}
```

## 7. 常见问题

**Q1: 为什么使用列主序（ne[0]是列）？**
- GGML继承了Fortran/BLAS的约定
- 有利于矩阵乘法的缓存局部性
- 与PyTorch的行主序不同，需要注意转换

**Q2: stride (nb) 什么时候不连续？**
```c
// 情况1：转置
原始: ne=[3,2], nb=[4,12]  (连续)
转置: ne=[2,3], nb=[12,4]  (不连续!)

// 情况2：切片
原始: ne=[10,10], nb=[4,40]
切片: ne=[5,5], nb=[4,40]  (nb[1]指向原始步长)

// 情况3：padding
原始数据: [1,2,3,4,5,6,7,8]
padding后: [1,2,3,0,4,5,6,0]  (每3个元素后插入padding)
```

**Q3: 为什么需要计算图而不是立即执行？**
- **优化机会**: 可以融合操作 (add+mul -> fma)
- **内存规划**: 提前知道所有张量大小，优化内存分配
- **并行化**: 识别可并行的操作
- **调试**: 可以可视化和检查整个计算流程

**Q4: no_alloc参数什么时候用？**
```c
// 场景1：只想知道内存需求
params.no_alloc = true;
ctx = ggml_init(params);
// ... 构建图 ...
size_t required = ggml_used_mem(ctx);  // 获取所需内存

// 场景2：使用外部内存管理
params.no_alloc = true;
// 手动分配张量的data指针
```

## 8. 性能技巧

### 8.1 内存对齐

```c
// GGML自动对齐到32字节（AVX2需要）
#define GGML_MEM_ALIGN 32

// 自定义张量时也要注意对齐
void* aligned_malloc(size_t size) {
    void* ptr;
    posix_memalign(&ptr, GGML_MEM_ALIGN, size);
    return ptr;
}
```

### 8.2 复用上下文

```c
// ❌ 不好：每次都创建新上下文
for (int i = 0; i < 100; i++) {
    ctx = ggml_init(params);
    // ... do work ...
    ggml_free(ctx);
}

// ✅ 好：复用上下文
ctx = ggml_init(params);
for (int i = 0; i < 100; i++) {
    // ... do work ...
    ggml_reset(ctx);  // 重置而不是释放
}
ggml_free(ctx);
```

### 8.3 使用视图而非复制

```c
// ❌ 复制数据（慢）
struct ggml_tensor * sub = ggml_new_tensor_2d(ctx, type, w, h);
memcpy(sub->data, src->data + offset, ggml_nbytes(sub));

// ✅ 使用视图（快）
struct ggml_tensor * sub = ggml_view_2d(ctx, src, w, h,
                                        src->nb[1], offset);
```

## 9. 下一步

明天我们将学习：

**Day 3 预告：张量操作与计算图**
- 深入理解各种张量操作
- 计算图的构建与优化
- 自动微分机制
- 内存分配器详解

## 作业

1. ✏️ 实现一个函数打印任意张量的维度和stride信息
2. 🔍 用gdb观察张量的内存布局
3. 📖 阅读 `ggml/src/ggml.c` 中的 `ggml_mul_mat` 实现
4. 💻 实现练习3，并理解每一步的计算

---

**继续学习**: [Day 3: 张量操作与计算图](day03-tensor-operations.md) →

# Day 3: 张量操作与计算图

## 课程目标

今天我们将深入学习：
- GGML 的核心张量操作
- 计算图（Computation Graph）的概念与实现
- 如何构建和执行计算图
- 常见张量操作的实现细节

## 1. 核心张量操作

### 1.1 张量创建操作

GGML 提供了丰富的张量创建函数，位于 `ggml/src/ggml.c`：

```c
// 创建1D张量
struct ggml_tensor * ggml_new_tensor_1d(
    struct ggml_context * ctx,
    enum ggml_type type,
    int64_t ne0);  // 元素数量

// 创建2D张量（最常用：矩阵）
struct ggml_tensor * ggml_new_tensor_2d(
    struct ggml_context * ctx,
    enum ggml_type type,
    int64_t ne0,   // 列数（第一维）
    int64_t ne1);  // 行数（第二维）

// 创建3D张量
struct ggml_tensor * ggml_new_tensor_3d(
    struct ggml_context * ctx,
    enum ggml_type type,
    int64_t ne0, int64_t ne1, int64_t ne2);

// 创建4D张量
struct ggml_tensor * ggml_new_tensor_4d(
    struct ggml_context * ctx,
    enum ggml_type type,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
```

**重要概念**：GGML 使用行主序（row-major）存储：
- `ne[0]` = 列数（最快变化维度）
- `ne[1]` = 行数
- `ne[2]` = 深度
- `ne[3]` = 批次

### 1.2 基础算术操作

#### 加法操作

```c
// 位置：ggml/src/ggml.c:4523
struct ggml_tensor * ggml_add(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * b) {
    // 创建结果张量
    struct ggml_tensor * result = ggml_dup_tensor(ctx, a);
    result->op = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// 对应的前向计算函数
// 位置：ggml/src/ggml.c:10234
static void ggml_compute_forward_add_f32(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // 获取维度
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    // ... 其他维度

    // 执行逐元素加法
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                // 使用SIMD优化的向量加法
                ggml_vec_add_f32(ne00,
                    (float *) dst_ptr,
                    (float *) src0_ptr,
                    (float *) src1_ptr);
            }
        }
    }
}
```

#### 乘法操作

```c
// 逐元素乘法
struct ggml_tensor * ggml_mul(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * b);

// 标量乘法
struct ggml_tensor * ggml_scale(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    float s);
```

### 1.3 矩阵乘法 - 最关键的操作

矩阵乘法是 LLM 推理中最耗时的操作，占总计算量的 80%+。

```c
// 位置：ggml/src/ggml.c:5247
// C = ggml_mul_mat(ctx, A, B) 计算: C^T = AB^T ⟺ C = BA^T
struct ggml_tensor * ggml_mul_mat(
    struct ggml_context * ctx,
    struct ggml_tensor * a,    // [K, M] 权重矩阵
    struct ggml_tensor * b) {  // [K, N] 输入矩阵

    // 断言维度兼容
    GGML_ASSERT(a->ne[0] == b->ne[0]);  // K 维度必须匹配

    // 创建输出张量 [M, N]
    struct ggml_tensor * result = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32,
        a->ne[1],  // M (A的列数)
        b->ne[1]); // N (B的列数)

    result->op = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}
```

**矩阵乘法的优化**：
- CPU：使用 AVX2/NEON SIMD 指令
- GPU：使用 cuBLAS（CUDA）或 Metal Performance Shaders
- 量化：特殊的量化矩阵乘法内核

#### 量化矩阵乘法示例（CUDA）

```cuda
// 位置：ggml/src/ggml-cuda/mmq.cuh
// 针对 Q4_K 量化的优化矩阵乘法
template<int qk, int qr, dequantize_kernel_t dequantize_kernel, int ncols>
static __global__ void dequantize_mul_mat_vec_q4_k(
    const void * __restrict__ vx,
    const float * __restrict__ y,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x) {

    // 每个 warp 处理一行
    const int row = blockIdx.x;

    // 加载量化数据到共享内存
    __shared__ float tmp[WARP_SIZE];

    // 反量化并执行点积
    float sum = 0.0f;
    for (int i = threadIdx.x; i < ncols_x; i += WARP_SIZE) {
        const float xi = dequantize_q4_k(vx, i);  // 反量化
        sum += xi * y[i];  // 点积累加
    }

    // Warp 内归约
    sum = warp_reduce_sum(sum);

    if (threadIdx.x == 0) {
        dst[row] = sum;
    }
}
```

### 1.4 RoPE 位置编码

RoPE（Rotary Position Embedding）是现代 LLM 的核心技术。

```c
// 位置：ggml/src/ggml.c:5819
struct ggml_tensor * ggml_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * a,     // 输入张量 [n_embd, n_tokens, ...]
    struct ggml_tensor * b,     // 位置索引 [n_tokens]
    int n_dims,                 // RoPE 维度
    int mode,                   // RoPE 模式（NORM/NEOX/...）
    int n_ctx) {                // 上下文长度

    struct ggml_tensor * result = ggml_dup_tensor(ctx, a);

    // 设置操作参数
    int32_t params[11] = {
        /*n_dims=*/ n_dims,
        /*mode=*/   mode,
        /*n_ctx=*/  n_ctx,
        // ... 其他参数
    };
    ggml_set_op_params(result, params, sizeof(params));

    result->op = GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}
```

**RoPE 前向计算**（位置：`ggml/src/ggml.c:12456`）：

```c
static void ggml_compute_forward_rope_f32(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];  // 输入
    const struct ggml_tensor * src1 = dst->src[1];  // 位置

    // 获取 RoPE 参数
    const int n_dims     = ((int32_t *) dst->op_params)[0];
    const int mode       = ((int32_t *) dst->op_params)[1];
    const float freq_base = ((float *) dst->op_params)[4];

    // 对每个 token 应用旋转
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int64_t p = positions[i2];  // 当前位置

            for (int64_t i1 = 0; i1 < ne1; i1++) {
                // 对每对维度应用旋转
                for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                    // 计算旋转角度
                    const float theta = powf(freq_base, -(float)i0 / n_dims);
                    const float cos_theta = cosf(p * theta);
                    const float sin_theta = sinf(p * theta);

                    // 旋转操作
                    const float x0 = src[i0];
                    const float x1 = src[i0 + 1];

                    dst[i0]     = x0 * cos_theta - x1 * sin_theta;
                    dst[i0 + 1] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    }
}
```

### 1.5 其他重要操作

#### 归一化操作

```c
// RMS Norm（LLaMA使用）
struct ggml_tensor * ggml_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    float eps);

// Layer Norm（GPT使用）
struct ggml_tensor * ggml_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    float eps);
```

#### 激活函数

```c
// SiLU / Swish (LLaMA FFN)
struct ggml_tensor * ggml_silu(
    struct ggml_context * ctx,
    struct ggml_tensor * a);

// GELU (GPT FFN)
struct ggml_tensor * ggml_gelu(
    struct ggml_context * ctx,
    struct ggml_tensor * a);

// ReLU
struct ggml_tensor * ggml_relu(
    struct ggml_context * ctx,
    struct ggml_tensor * a);
```

#### 注意力相关操作

```c
// Softmax
struct ggml_tensor * ggml_soft_max(
    struct ggml_context * ctx,
    struct ggml_tensor * a);

// Flash Attention（优化的注意力）
struct ggml_tensor * ggml_flash_attn_ext(
    struct ggml_context * ctx,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    struct ggml_tensor * mask,
    float scale);
```

## 2. 计算图（Computation Graph）

### 2.1 计算图概念

计算图是一个有向无环图（DAG），表示张量操作的依赖关系：
- **节点**：张量（变量）
- **边**：操作（函数）

```
示例：f(x) = σ(Wx + b)

      [x]  [W]  [b]
       │    │    │
       └─→[mul]←─┘
           │
         [add]
           │
        [sigmoid]
           │
          [y]
```

### 2.2 计算图结构

```c
// 位置：ggml/include/ggml.h:625
struct ggml_cgraph {
    int size;          // 最大节点数
    int n_nodes;       // 当前节点数
    int n_leafs;       // 叶子节点数

    struct ggml_tensor ** nodes;   // 所有操作节点
    struct ggml_tensor ** grads;   // 梯度（反向传播用）
    struct ggml_tensor ** leafs;   // 输入节点

    struct ggml_hash_set visited_hash_set;  // 已访问节点

    // 执行顺序（拓扑排序后）
    enum ggml_cgraph_eval_order order;
};
```

### 2.3 构建计算图

```c
// 创建计算图
struct ggml_cgraph * ggml_new_graph(struct ggml_context * ctx) {
    return ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}

// 添加节点到图
void ggml_build_forward_expand(
    struct ggml_cgraph * cgraph,
    struct ggml_tensor * tensor) {

    // 递归添加所有依赖节点
    if (tensor->op != GGML_OP_NONE) {
        // 先添加源节点
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i]) {
                ggml_build_forward_expand(cgraph, tensor->src[i]);
            }
        }
    }

    // 添加当前节点
    if (!ggml_hash_contains(&cgraph->visited_hash_set, tensor)) {
        ggml_hash_insert(&cgraph->visited_hash_set, tensor);
        cgraph->nodes[cgraph->n_nodes++] = tensor;
    }
}
```

### 2.4 执行计算图

```c
// 位置：ggml/src/ggml-backend.cpp:823
enum ggml_status ggml_backend_graph_compute(
    ggml_backend_t backend,
    struct ggml_cgraph * cgraph) {

    // 为每个节点分配后端资源
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        // 分配输出缓冲区
        if (!node->data) {
            ggml_backend_tensor_alloc(backend, node);
        }
    }

    // 按顺序执行每个操作
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        // 根据操作类型调用对应的计算函数
        switch (node->op) {
            case GGML_OP_ADD:
                ggml_compute_forward_add(params, node);
                break;
            case GGML_OP_MUL_MAT:
                ggml_compute_forward_mul_mat(params, node);
                break;
            case GGML_OP_ROPE:
                ggml_compute_forward_rope(params, node);
                break;
            // ... 200+ 种操作
        }
    }

    return GGML_STATUS_SUCCESS;
}
```

## 3. 实战示例：构建简单的 FFN

让我们构建一个前馈神经网络层：`y = GELU(xW1) W2`

```c
#include "ggml.h"
#include <stdio.h>

int main() {
    // 1. 初始化上下文
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,  // 128MB
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // 2. 创建输入张量 [batch_size=1, d_model=512]
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1);
    ggml_set_name(x, "input");

    // 3. 创建权重矩阵
    // W1: [d_model=512, d_ff=2048]
    struct ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 2048);
    ggml_set_name(w1, "ffn.w1");

    // W2: [d_ff=2048, d_model=512]
    struct ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 512);
    ggml_set_name(w2, "ffn.w2");

    // 4. 构建计算图
    // hidden = x @ W1
    struct ggml_tensor * hidden = ggml_mul_mat(ctx, w1, x);
    ggml_set_name(hidden, "hidden");

    // activated = GELU(hidden)
    struct ggml_tensor * activated = ggml_gelu(ctx, hidden);
    ggml_set_name(activated, "activated");

    // output = activated @ W2
    struct ggml_tensor * output = ggml_mul_mat(ctx, w2, activated);
    ggml_set_name(output, "output");

    // 5. 创建计算图
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    // 6. 初始化输入数据（示例）
    float * x_data = (float *) x->data;
    for (int i = 0; i < 512; i++) {
        x_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // 7. 执行计算图
    int n_threads = 4;
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    // 8. 获取结果
    float * output_data = (float *) output->data;
    printf("Output[0] = %f\n", output_data[0]);

    // 9. 清理
    ggml_free(ctx);

    return 0;
}
```

编译运行：
```bash
gcc -O3 -o ffn_example ffn_example.c \
    -I./ggml/include \
    -L./build/ggml/src -lggml \
    -lm -lpthread

./ffn_example
```

## 4. 计算图可视化

GGML 提供了计算图导出功能：

```c
// 导出为 DOT 格式（Graphviz）
ggml_graph_dump_dot(gf, NULL, "ffn_graph.dot");
```

生成可视化：
```bash
dot -Tpng ffn_graph.dot -o ffn_graph.png
```

## 5. 性能优化技巧

### 5.1 操作融合

将多个操作融合为一个，减少内存访问：

```c
// 不融合：3次内存读写
y = ggml_add(ctx, x, bias);    // 读x, 写y
y = ggml_mul(ctx, y, scale);   // 读y, 写y
y = ggml_relu(ctx, y);         // 读y, 写y

// 融合：1次内存读写
y = ggml_add_mul_relu(ctx, x, bias, scale);  // 一次完成
```

### 5.2 In-place 操作

某些操作可以原地修改，节省内存：

```c
// 原地 ReLU
struct ggml_tensor * ggml_relu_inplace(
    struct ggml_context * ctx,
    struct ggml_tensor * a);

// 原地 Scale
struct ggml_tensor * ggml_scale_inplace(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    float s);
```

### 5.3 使用合适的数据类型

```c
// 对于权重，使用量化类型
struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 512, 2048);

// 对于激活值，使用 FP16
struct ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 1);
```

## 6. 调试技巧

### 6.1 打印张量信息

```c
void print_tensor_info(struct ggml_tensor * t) {
    printf("Tensor: %s\n", t->name);
    printf("  Type: %s\n", ggml_type_name(t->type));
    printf("  Shape: [%ld, %ld, %ld, %ld]\n",
           t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    printf("  Stride: [%ld, %ld, %ld, %ld]\n",
           t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
    printf("  Op: %s\n", ggml_op_name(t->op));
}
```

### 6.2 检查计算图结构

```c
void print_graph_info(struct ggml_cgraph * gf) {
    printf("Graph nodes: %d\n", gf->n_nodes);

    for (int i = 0; i < gf->n_nodes; i++) {
        struct ggml_tensor * node = gf->nodes[i];
        printf("  [%d] %s: %s -> [%ld, %ld]\n",
               i, node->name,
               ggml_op_name(node->op),
               node->ne[0], node->ne[1]);
    }
}
```

## 7. 实践练习

### 练习 1：实现 LayerNorm
编写代码实现 LayerNorm 层：
```
y = (x - mean) / sqrt(variance + eps) * gamma + beta
```

### 练习 2：构建注意力层
构建单头自注意力：
```
Q = xWq, K = xWk, V = xWv
scores = QK^T / sqrt(d_k)
attn = softmax(scores)
output = attn @ V
```

### 练习 3：分析操作性能
使用 `ggml_graph_compute` 的性能统计功能，分析每个操作的耗时。

## 8. 总结

今天我们学习了：

✅ **张量操作**：创建、算术、矩阵乘法、RoPE 等核心操作
✅ **计算图**：理解计算图的概念、构建和执行流程
✅ **实战示例**：构建简单的 FFN 层
✅ **优化技巧**：操作融合、in-place 操作、数据类型选择
✅ **调试方法**：打印信息、可视化计算图

### 关键要点

1. **矩阵乘法是性能瓶颈**：占据 80%+ 计算时间
2. **计算图实现延迟执行**：定义时不计算，执行时统一计算
3. **操作实现分为两部分**：图构建函数 + 前向计算函数
4. **SIMD 优化至关重要**：AVX2/NEON 可提升 4-8 倍性能

## 下一步

明天我们将学习 **Day 4: GGML 内存管理机制**，深入理解：
- 内存分配策略
- mmap 文件映射
- 后端缓冲区管理
- 内存优化技巧

---

**思考题**：
1. 为什么 `ggml_mul_mat(A, B)` 计算的是 `BA^T` 而不是 `AB`？
2. RoPE 相比传统位置编码有什么优势？
3. 如何优化大批量的矩阵乘法？

📚 继续学习：[Day 4: GGML 内存管理机制](day04-ggml-memory.md)

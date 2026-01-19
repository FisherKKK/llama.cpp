# Day 10: 注意力机制优化

## 课程目标

深入理解注意力机制的优化技术：
- Flash Attention 原理与实现
- 多查询注意力（MQA）
- Sliding Window Attention（SWA）
- 稀疏注意力技术
- 长上下文优化

## 1. Flash Attention

### 1.1 标准注意力的问题

```
标准注意力计算流程：

1. Q @ K^T  →  [n_tokens, kv_size] (写入 HBM)
2. softmax   →  [n_tokens, kv_size] (读写 HBM)
3. @ V       →  [n_tokens, n_embd]  (读写 HBM)

问题：
• 中间结果 (QK^T) 需要大量内存
• 多次 HBM 访问（GPU 的瓶颈）
• 时间复杂度: O(N²)
• 空间复杂度: O(N²)

对于 n_tokens=2048, 需要:
    2048² × 4 bytes = 16MB 每层
    16MB × 32 层 = 512MB 仅用于注意力矩阵！
```

### 1.2 Flash Attention 原理

**核心思想**：分块计算 + 在线 Softmax

```cpp
// 伪代码
FlashAttention(Q, K, V, block_size):
    // 分块
    Q_blocks = split(Q, block_size)
    K_blocks = split(K, block_size)
    V_blocks = split(V, block_size)

    O = zeros_like(Q)  // 输出
    l = zeros(Q.rows)  // 累积归一化因子
    m = -inf(Q.rows)   // 累积最大值

    // 外循环：Q 块
    for Q_block in Q_blocks:
        // 内循环：K, V 块
        for K_block, V_block in zip(K_blocks, V_blocks):
            // 1. 计算当前块的注意力分数
            S_block = Q_block @ K_block.T

            // 2. 在线更新最大值和归一化因子
            m_new = max(m, max(S_block, axis=1))
            l_new = exp(m - m_new) * l + exp(S_block - m_new).sum(axis=1)

            // 3. 更新输出
            O = O * (l / l_new) + (S_block @ V_block) * (exp(S_block - m_new) / l_new)

            m = m_new
            l = l_new

    return O
```

**优势**：
- ✅ 内存使用：O(N) 而非 O(N²)
- ✅ HBM 访问减少：~5x 加速
- ✅ 支持更长上下文
- ✅ 数值稳定

### 1.3 CUDA 实现（简化版）

```cuda
// 位置：ggml/src/ggml-cuda/fattn.cuh
template<int D, int ncols>
__global__ void flash_attn_kernel(
    const float * Q,    // [n_head, n_tokens, d_head]
    const float * K,    // [n_head, kv_size, d_head]
    const float * V,    // [n_head, kv_size, d_head]
    float * O,          // [n_head, n_tokens, d_head]
    const float scale,
    const int kv_size) {

    const int tid = threadIdx.x;
    const int head_idx = blockIdx.y;
    const int token_idx = blockIdx.x;

    // 共享内存：减少 HBM 访问
    __shared__ float Q_shared[D];
    __shared__ float K_shared[D];
    __shared__ float V_shared[D];
    __shared__ float scores_shared[32];  // 块内分数

    // 加载 Q 到共享内存
    if (tid < D) {
        Q_shared[tid] = Q[head_idx * n_tokens * D + token_idx * D + tid];
    }
    __syncthreads();

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output[D] = {0};

    // 分块处理 K, V
    const int n_blocks = (kv_size + 31) / 32;
    for (int block = 0; block < n_blocks; block++) {
        int kv_idx = block * 32 + tid;

        // 加载 K 块
        if (kv_idx < kv_size && tid < D) {
            K_shared[tid] = K[head_idx * kv_size * D + kv_idx * D + tid];
        }
        __syncthreads();

        // 计算 Q @ K^T
        if (kv_idx < kv_size) {
            float score = 0.0f;
            for (int i = 0; i < D; i++) {
                score += Q_shared[i] * K_shared[i];
            }
            score *= scale;
            scores_shared[tid] = score;

            // 在线更新最大值
            max_score = fmaxf(max_score, score);
        }
        __syncthreads();

        // 计算 exp 并累积
        if (kv_idx < kv_size) {
            float exp_score = expf(scores_shared[tid] - max_score);
            sum_exp += exp_score;

            // 加载 V 并累积到输出
            if (tid < D) {
                V_shared[tid] = V[head_idx * kv_size * D + kv_idx * D + tid];
            }
            __syncthreads();

            for (int i = 0; i < D; i++) {
                output[i] += exp_score * V_shared[i];
            }
        }
        __syncthreads();
    }

    // 归一化并写回
    if (tid < D) {
        O[head_idx * n_tokens * D + token_idx * D + tid] = output[tid] / sum_exp;
    }
}
```

### 1.4 使用 Flash Attention

```cpp
// 启用 Flash Attention
llama_context_params ctx_params = llama_context_default_params();
ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

llama_context * ctx = llama_new_context_with_model(model, ctx_params);

// 自动使用 Flash Attention（如果 GPU 支持）
// 性能提升：2-4x（取决于序列长度）
```

## 2. 多查询注意力（MQA）

### 2.1 MQA vs GQA vs MHA

```
MHA (Multi-Head Attention):
    Q: [n_head=32, n_tokens, d_head]
    K: [n_head=32, kv_size, d_head]
    V: [n_head=32, kv_size, d_head]
    KV 缓存: 32 × 2 × kv_size × d_head

GQA (Grouped-Query Attention):
    Q: [n_head=32, n_tokens, d_head]
    K: [n_head_kv=4, kv_size, d_head]  ← 共享
    V: [n_head_kv=4, kv_size, d_head]  ← 共享
    KV 缓存: 4 × 2 × kv_size × d_head (8x 减少)

MQA (Multi-Query Attention):
    Q: [n_head=32, n_tokens, d_head]
    K: [n_head_kv=1, kv_size, d_head]  ← 全局共享
    V: [n_head_kv=1, kv_size, d_head]  ← 全局共享
    KV 缓存: 1 × 2 × kv_size × d_head (32x 减少)
```

### 2.2 MQA 实现

```cpp
// 位置：src/llama-graph.cpp:589
struct ggml_tensor * llm_build_mqa(
    struct ggml_context * ctx,
    struct ggml_tensor * Q,    // [n_head, n_tokens, d_head]
    struct ggml_tensor * K,    // [1, kv_size, d_head]
    struct ggml_tensor * V) {  // [1, kv_size, d_head]

    const int n_head = Q->ne[1];

    // 广播 K, V 到所有头
    K = ggml_repeat(ctx, K, ggml_new_tensor_3d(ctx, K->type, d_head, n_head, kv_size));
    V = ggml_repeat(ctx, V, ggml_new_tensor_3d(ctx, V->type, d_head, n_head, kv_size));

    // 标准注意力计算
    struct ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
    kq = ggml_scale(ctx, kq, 1.0f / sqrtf(d_head));
    kq = ggml_soft_max(ctx, kq);

    struct ggml_tensor * output = ggml_mul_mat(ctx, V, kq);

    return output;
}
```

**Trade-off**：
- MQA：最省内存，但质量略降
- GQA：平衡内存和质量（推荐）
- MHA：最高质量，但内存最多

## 3. Sliding Window Attention

### 3.1 原理

```
标准因果注意力：
    Token 0: 可以看到 [0]
    Token 1: 可以看到 [0, 1]
    Token 2: 可以看到 [0, 1, 2]
    ...
    Token N: 可以看到 [0, 1, 2, ..., N]

    ❌ 问题：KV 缓存随 N 线性增长

SWA (window_size=1024)：
    Token 1000: 可以看到 [0, 1, ..., 1000]        (全部)
    Token 1500: 可以看到 [476, 477, ..., 1500]    (最近 1024)
    Token 2000: 可以看到 [976, 977, ..., 2000]    (最近 1024)

    ✅ 好处：KV 缓存固定为 window_size
```

### 3.2 实现

```cpp
// 位置：src/llama-kv-cache.cpp:456
void llama_kv_cache_update_swa(
    struct llama_kv_cache & cache,
    int n_tokens_new,
    int window_size) {

    if (cache.head + n_tokens_new <= window_size) {
        // 还在窗口内，正常添加
        cache.head += n_tokens_new;
        return;
    }

    // 超出窗口，需要滑动
    const int n_discard = cache.head + n_tokens_new - window_size;

    // 移动 KV 缓存
    for (int il = 0; il < cache.n_layer; il++) {
        ggml_tensor * k = cache.k_l[il];
        ggml_tensor * v = cache.v_l[il];

        // 丢弃最旧的 n_discard 个 token
        // K: [d_head, kv_size] → shift left by n_discard
        memmove(k->data,
                (char*)k->data + n_discard * k->nb[1],
                (window_size - n_discard) * k->nb[1]);

        // 类似地移动 V
        memmove(v->data,
                (char*)v->data + n_discard * v->nb[1],
                (window_size - n_discard) * v->nb[1]);
    }

    cache.head = window_size - n_discard;
}
```

**使用场景**：
- Mistral 7B：window_size=4096
- 长文档生成
- 对话系统

## 4. 长上下文优化

### 4.1 YaRN（Yet another RoPE extensioN）

**问题**：RoPE 外推性能差
```
训练时: max_pos = 2048
推理时: pos = 4096  ← 超出训练范围
→ 模型性能大幅下降
```

**YaRN 解决方案**：
```cpp
// 位置：src/llama-graph.cpp:234
float yarn_get_scaling_factor(float pos, float base_freq, float scale_factor) {
    // 根据位置动态调整频率
    if (pos <= ctx_train) {
        return 1.0f;  // 训练范围内，不缩放
    } else {
        // 外推范围，应用缩放
        float alpha = scale_factor * (pos / ctx_train);
        return 1.0f / powf(alpha, 2.0f / dim);
    }
}
```

### 4.2 上下文压缩

```cpp
// 保留重要 token，丢弃不重要的
std::vector<int> compress_context(
    const std::vector<llama_token> & tokens,
    const std::vector<float> & importance_scores,
    int target_len) {

    // 按重要性排序
    std::vector<int> indices(tokens.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&](int a, int b) {
            return importance_scores[a] > importance_scores[b];
        });

    // 保留前 target_len 个
    indices.resize(target_len);
    std::sort(indices.begin(), indices.end());

    std::vector<int> compressed;
    for (int idx : indices) {
        compressed.push_back(tokens[idx]);
    }

    return compressed;
}
```

## 5. 性能对比

| 技术 | 内存节省 | 速度提升 | 质量影响 | 适用场景 |
|------|---------|---------|---------|---------|
| **Flash Attention** | 50% | 2-4x | 无 | 所有 |
| **GQA** | 75% | 1.5x | 轻微 | 推理 |
| **MQA** | 97% | 2x | 中等 | 速度优先 |
| **SWA** | 固定 | 1.2x | 取决于窗口 | 长文本 |
| **YaRN** | 无 | 无 | 改善外推 | 长上下文 |

## 6. 总结

今天我们学习了注意力机制的优化：

✅ **Flash Attention**：减少内存和计算
✅ **MQA/GQA**：共享 KV 头
✅ **SWA**：固定窗口大小
✅ **长上下文**：YaRN、压缩技术

### 关键要点

1. **Flash Attention 是必备**：2-4x 加速
2. **GQA 最平衡**：质量和速度兼顾
3. **SWA 适合长文本**：固定内存
4. **组合使用**：叠加优化效果

## 下一步

明天我们将学习 **Day 11: CPU 后端与 SIMD 优化**：
- AVX2/AVX-512 优化
- ARM NEON 优化
- 矩阵乘法内核
- 量化加速

---

📚 [Day 11: CPU 后端与 SIMD 优化](day11-cpu-backend.md)

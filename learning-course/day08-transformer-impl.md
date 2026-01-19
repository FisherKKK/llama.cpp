# Day 8: Transformer 架构实现

## 课程目标

深入理解 Transformer 架构在 llama.cpp 中的实现：
- LLaMA 架构详解
- Self-Attention 实现细节
- FFN (Feed-Forward Network) 实现
- 不同模型架构的变体
- 架构特定的优化

## 1. LLaMA 架构概览

### 1.1 模型结构

```
LLaMA 模型结构：

Input: token IDs [n_tokens]
    ↓
┌─────────────────────────────────────┐
│ Token Embedding [n_vocab, n_embd]  │
└──────────────┬──────────────────────┘
               ↓
    ┌──────────────────┐
    │  Transformer     │ × n_layer (通常 32 层)
    │  Block           │
    └────────┬─────────┘
             │
    ╔════════▼═══════╗
    ║ 1. RMS Norm    ║
    ╚════════╤═══════╝
             ↓
    ╔════════▼════════════════════╗
    ║ 2. Self-Attention           ║
    ║   • Q, K, V 投影             ║
    ║   • RoPE 位置编码           ║
    ║   • 缩放点积注意力           ║
    ║   • KV 缓存                 ║
    ║   • 输出投影                ║
    ╚════════╤════════════════════╝
             ↓
    ╔════════▼═══════╗
    ║ 3. Residual    ║ +
    ╚════════╤═══════╝
             ↓
    ╔════════▼═══════╗
    ║ 4. RMS Norm    ║
    ╚════════╤═══════╝
             ↓
    ╔════════▼════════════════════╗
    ║ 5. FFN (SwiGLU)             ║
    ║   • Gate 投影                ║
    ║   • Up 投影                  ║
    ║   • SiLU 激活                ║
    ║   • Element-wise 乘法        ║
    ║   • Down 投影                ║
    ╚════════╤════════════════════╝
             ↓
    ╔════════▼═══════╗
    ║ 6. Residual    ║ +
    ╚════════╤═══════╝
             │
    └────────┴─────────┘
             ↓
┌─────────────────────────────────────┐
│ Final RMS Norm                      │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Output Projection [n_embd, n_vocab]│
└──────────────┬──────────────────────┘
               ↓
Output: logits [n_vocab]
```

### 1.2 关键参数

```cpp
// LLaMA-7B 的超参数
n_vocab     = 32000    // 词表大小
n_embd      = 4096     // 嵌入维度 (隐藏大小)
n_layer     = 32       // 层数
n_head      = 32       // 注意力头数
n_head_kv   = 32       // KV 头数 (GQA: 可以更少)
n_ff        = 11008    // FFN 隐藏大小
n_ctx_train = 2048     // 训练上下文长度
rope_freq_base = 10000.0  // RoPE 频率基数
```

## 2. Self-Attention 实现详解

### 2.1 完整的注意力计算

```cpp
// 位置：src/llama-graph.cpp:456
struct ggml_tensor * llm_build_kqv(
    struct llama_context & lctx,
    struct ggml_tensor * cur,      // 输入 [n_embd, n_tokens]
    struct ggml_tensor * wq,       // Q 权重 [n_embd, n_embd]
    struct ggml_tensor * wk,       // K 权重 [n_embd, n_embd_gqa]
    struct ggml_tensor * wv,       // V 权重 [n_embd, n_embd_gqa]
    struct ggml_tensor * wo,       // 输出权重 [n_embd, n_embd]
    int n_head,
    int n_head_kv,
    int il) {                      // 层索引

    struct ggml_context * ctx = lctx.ctx_compute.get();
    const int n_embd_head = lctx.model.hparams.n_embd_head_k;
    const int n_embd = lctx.model.hparams.n_embd;

    // === Step 1: Q, K, V 投影 ===
    struct ggml_tensor * Qcur = ggml_mul_mat(ctx, wq, cur);
    struct ggml_tensor * Kcur = ggml_mul_mat(ctx, wk, cur);
    struct ggml_tensor * Vcur = ggml_mul_mat(ctx, wv, cur);

    // === Step 2: Reshape 为多头 ===
    // Q: [n_embd, n_tokens] → [n_embd_head, n_head, n_tokens]
    Qcur = ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, n_tokens);

    // K, V: [n_embd_gqa, n_tokens] → [n_embd_head, n_head_kv, n_tokens]
    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

    // === Step 3: 应用 RoPE ===
    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos,
                         n_rot, rope_type, freq_base, freq_scale);
    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos,
                         n_rot, rope_type, freq_base, freq_scale);

    // === Step 4: 存储 K, V 到缓存 ===
    struct ggml_tensor * k = kv_cache.k_l[il];  // [n_embd_head, n_head_kv, kv_size]
    struct ggml_tensor * v = kv_cache.v_l[il];

    // 将 Kcur, Vcur 拷贝到 KV 缓存的相应位置
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx, Kcur,
                 ggml_view_1d(ctx, k, n_tokens * n_embd_head * n_head_kv,
                              (kv_head) * n_embd_head * n_head_kv * ggml_element_size(k))));
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx, Vcur,
                 ggml_view_1d(ctx, v, n_tokens * n_embd_head * n_head_kv,
                              (kv_head) * n_embd_head * n_head_kv * ggml_element_size(v))));

    // === Step 5: Flash Attention（如果启用）===
    struct ggml_tensor * kqv;

    if (use_flash_attn) {
        // Flash Attention: 融合的优化实现
        kqv = ggml_flash_attn_ext(ctx, Qcur, k, v, kq_mask,
                                   1.0f / sqrtf(n_embd_head),  // scale
                                   0.0f);  // max_bias
    } else {
        // === Step 5a: 标准注意力 - 计算 QK^T ===
        // scores = Q @ K^T  [n_head, n_tokens, kv_size]
        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, Qcur);

        // === Step 5b: 缩放 ===
        kq = ggml_scale(ctx, kq, 1.0f / sqrtf(n_embd_head));

        // === Step 5c: 添加注意力掩码 ===
        kq = ggml_add(ctx, kq, kq_mask);

        // === Step 5d: Softmax ===
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, 1.0f, hparams.f_max_alibi_bias);

        // === Step 5e: 应用到 V ===
        // kqv = softmax(scores) @ V  [n_head, n_tokens, n_embd_head]
        kqv = ggml_mul_mat(ctx, v, kq);
    }

    // === Step 6: Reshape 回原始形状 ===
    // [n_head, n_tokens, n_embd_head] → [n_embd, n_tokens]
    kqv = ggml_reshape_2d(ctx, kqv, n_embd, n_tokens);

    // === Step 7: 输出投影 ===
    struct ggml_tensor * attn_out = ggml_mul_mat(ctx, wo, kqv);

    return attn_out;
}
```

### 2.2 分组查询注意力（GQA）

```
标准 Multi-Head Attention (MHA):
    Q: 32 个头
    K: 32 个头  ← 每个头都有独立的 K
    V: 32 个头  ← 每个头都有独立的 V

分组查询注意力 (GQA):
    Q: 32 个头
    K: 4 个头   ← 共享！多个 Q 头共享一个 KV 头
    V: 4 个头

    Q[0-7]   共享 → K[0], V[0]
    Q[8-15]  共享 → K[1], V[1]
    Q[16-23] 共享 → K[2], V[2]
    Q[24-31] 共享 → K[3], V[3]

优势：
    • KV 缓存减少 8 倍
    • 内存使用大幅降低
    • 推理速度提升
    • 质量损失很小
```

**实现**：
```cpp
const int n_head = 32;       // Q 头数
const int n_head_kv = 4;     // KV 头数
const int n_gqa = n_head / n_head_kv;  // 8 (每组 Q 头数)

// K, V 只有 4 个头
Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);

// 广播 KV 到所有 Q 头
K = ggml_repeat(ctx, K, n_gqa);  // [n_embd_head, 4, ...] → [n_embd_head, 32, ...]
```

## 3. FFN (Feed-Forward Network) 实现

### 3.1 SwiGLU 激活

LLaMA 使用 SwiGLU 而非传统的 ReLU：

```
传统 FFN (GPT-2 风格):
    hidden = ReLU(x @ W1 + b1)
    output = hidden @ W2 + b2

SwiGLU FFN (LLaMA 风格):
    gate = x @ W_gate
    up = x @ W_up
    hidden = SiLU(gate) ⊙ up    # ⊙ 是逐元素乘法
    output = hidden @ W_down

公式：
    SwiGLU(x, W) = SiLU(x @ W_gate) ⊙ (x @ W_up)
    其中 SiLU(x) = x * sigmoid(x)
```

### 3.2 实现代码

```cpp
// 位置：src/llama-graph.cpp:1123
struct ggml_tensor * llm_build_ffn(
    struct ggml_context * ctx,
    struct ggml_tensor * cur,      // 输入 [n_embd, n_tokens]
    struct ggml_tensor * ffn_gate, // Gate 权重 [n_embd, n_ff]
    struct ggml_tensor * ffn_up,   // Up 权重 [n_embd, n_ff]
    struct ggml_tensor * ffn_down) // Down 权重 [n_ff, n_embd]
{
    // === Step 1: Gate 和 Up 投影 ===
    struct ggml_tensor * gate = ggml_mul_mat(ctx, ffn_gate, cur);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, ffn_up, cur);

    // === Step 2: SiLU 激活 ===
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    gate = ggml_silu(ctx, gate);

    // === Step 3: 逐元素乘法 ===
    struct ggml_tensor * hidden = ggml_mul(ctx, gate, up);

    // === Step 4: Down 投影 ===
    struct ggml_tensor * output = ggml_mul_mat(ctx, ffn_down, hidden);

    return output;
}
```

**为什么使用 SwiGLU？**
- **性能更好**：在多个基准测试上优于 ReLU/GELU
- **门控机制**：可以动态控制信息流
- **平滑梯度**：比 ReLU 更利于训练

## 4. RMS Norm（Root Mean Square Normalization）

LLaMA 使用 RMS Norm 而非 LayerNorm：

```cpp
// 位置：ggml/src/ggml.c:12678
static void ggml_compute_forward_rms_norm_f32(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    const struct ggml_tensor * src = dst->src[0];
    const float eps = ((float *) dst->op_params)[0];

    const int n_elements = src->ne[0];  // 归一化维度

    for (int64_t i = 0; i < n_rows; i++) {
        const float * x = (float *)((char *) src->data + i * src->nb[1]);
        float * y = (float *)((char *) dst->data + i * dst->nb[1]);

        // 1. 计算平方和
        float sum = 0.0f;
        for (int j = 0; j < n_elements; j++) {
            sum += x[j] * x[j];
        }

        // 2. 计算 RMS
        float rms = sqrtf(sum / n_elements + eps);

        // 3. 归一化
        for (int j = 0; j < n_elements; j++) {
            y[j] = x[j] / rms;
        }
    }
}
```

**RMS Norm vs LayerNorm**：
```
LayerNorm:
    y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

RMS Norm:
    y = x / RMS(x) * gamma
    其中 RMS(x) = sqrt(mean(x²) + eps)

差异：
    • 不减去均值
    • 不需要 bias 项
    • 计算更快
    • 效果相当
```

## 5. 不同模型架构的变体

### 5.1 主要 LLM 架构对比

| 模型 | 注意力 | FFN 激活 | 归一化 | 位置编码 | 特殊之处 |
|------|--------|---------|--------|---------|---------|
| **LLaMA** | MHA/GQA | SwiGLU | RMS Norm | RoPE | 经典架构 |
| **Mistral** | GQA | SwiGLU | RMS Norm | RoPE | SWA (滑窗) |
| **Qwen** | GQA | SwiGLU | RMS Norm | RoPE | YARN 扩展 |
| **Gemma** | MQA | GeGLU | RMS Norm | RoPE | Logit 软上限 |
| **Phi** | MHA | GELU | LayerNorm | Learned | 小模型优化 |
| **GPT-2** | MHA | GELU | LayerNorm | Learned | 经典 GPT |

### 5.2 Mistral 的 Sliding Window Attention

```cpp
// Mistral 特有：滑动窗口注意力
// 位置：src/llama-graph.cpp:678

if (hparams.n_swa > 0) {  // Sliding Window Attention 启用
    // 只注意最近 n_swa 个 token
    for (int i = 0; i < n_tokens; i++) {
        int window_start = max(0, i - n_swa);
        int window_end = i + 1;

        // 掩码只允许窗口内的 token
        for (int j = 0; j < window_start; j++) {
            mask[i * kv_size + j] = -INFINITY;  // 屏蔽窗口外
        }
        for (int j = window_start; j < window_end; j++) {
            mask[i * kv_size + j] = 0.0f;  // 窗口内可见
        }
        for (int j = window_end; j < kv_size; j++) {
            mask[i * kv_size + j] = -INFINITY;  // 未来不可见
        }
    }
}
```

**SWA 优势**：
- 减少 KV 缓存使用
- 可以处理超长上下文
- 推理速度更快

### 5.3 Gemma 的 Logit 软上限

```cpp
// Gemma 特有：限制 logit 范围，防止数值不稳定
// 位置：src/llama-graph.cpp:1789

if (hparams.f_attn_logit_softcapping > 0.0f) {
    // logits = tanh(logits / cap) * cap
    float cap = hparams.f_attn_logit_softcapping;

    logits = ggml_scale(ctx, logits, 1.0f / cap);
    logits = ggml_tanh(ctx, logits);
    logits = ggml_scale(ctx, logits, cap);
}
```

## 6. 完整的 Transformer Block 实现

```cpp
// 位置：src/llama-graph.cpp:2341
struct ggml_tensor * llm_build_llama_layer(
    struct llama_context & lctx,
    int il,
    struct ggml_tensor * inpL,
    struct ggml_tensor * attn_mask) {

    const auto & model = lctx.model;
    const auto & hparams = model.hparams;
    const auto & layer = model.layers[il];
    struct ggml_context * ctx = lctx.ctx_compute.get();

    // ========== 注意力部分 ==========

    // 1. RMS Norm
    struct ggml_tensor * attn_norm_out = ggml_rms_norm(ctx, inpL, hparams.f_norm_rms_eps);
    attn_norm_out = ggml_mul(ctx, attn_norm_out, layer.attn_norm);

    // 2. Self-Attention
    struct ggml_tensor * attn_out = llm_build_kqv(
        lctx, attn_norm_out,
        layer.wq, layer.wk, layer.wv, layer.wo,
        hparams.n_head, hparams.n_head_kv, il);

    // 3. 残差连接
    inpL = ggml_add(ctx, inpL, attn_out);

    // ========== FFN 部分 ==========

    // 4. RMS Norm
    struct ggml_tensor * ffn_norm_out = ggml_rms_norm(ctx, inpL, hparams.f_norm_rms_eps);
    ffn_norm_out = ggml_mul(ctx, ffn_norm_out, layer.ffn_norm);

    // 5. FFN (SwiGLU)
    struct ggml_tensor * ffn_out = llm_build_ffn(
        ctx, ffn_norm_out,
        layer.ffn_gate, layer.ffn_up, layer.ffn_down);

    // 6. 残差连接
    inpL = ggml_add(ctx, inpL, ffn_out);

    return inpL;
}
```

## 7. 性能优化技巧

### 7.1 KV 缓存优化

```cpp
// 使用 FP16 存储 KV 缓存
ctx_params.type_k = GGML_TYPE_F16;  // 而非 F32
ctx_params.type_v = GGML_TYPE_F16;

// 节省 50% 的 KV 缓存内存
// 7B 模型：~1GB (F16) vs ~2GB (F32)
```

### 7.2 批处理优化

```cpp
// 批量处理多个 token
llama_batch batch = llama_batch_init(512, 0, 1);

// 添加多个 token
for (int i = 0; i < n_tokens; i++) {
    llama_batch_add(batch, tokens[i], i, {0}, i == n_tokens - 1);
}

// 一次推理处理整个批次
llama_decode(ctx, batch);
```

### 7.3 Flash Attention

```cpp
// 启用 Flash Attention（如果 GPU 支持）
ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

// 优势：
// • 减少内存访问
// • 2-4x 注意力加速
// • 支持更长的上下文
```

## 8. 总结

今天我们深入学习了 Transformer 架构：

✅ **LLaMA 架构**：完整的模型结构
✅ **Self-Attention**：多头注意力、GQA、Flash Attention
✅ **FFN**：SwiGLU 激活函数
✅ **归一化**：RMS Norm 实现
✅ **架构变体**：Mistral SWA、Gemma 软上限

### 关键要点

1. **GQA**：减少 KV 缓存，提升推理速度
2. **SwiGLU**：门控 FFN，性能更好
3. **RMS Norm**：更快的归一化
4. **RoPE**：旋转位置编码，支持外推

## 下一步

明天我们将学习 **Day 10: 注意力机制优化**：
- Flash Attention 原理
- 多查询注意力（MQA）
- Sliding Window Attention
- 稀疏注意力技术

---

**练习**：
1. 对比 MHA、GQA、MQA 的 KV 缓存大小
2. 实现一个简单的 Transformer Block
3. 分析不同激活函数的性能差异

📚 [Day 10: 注意力机制优化](day10-attention-optimization.md)

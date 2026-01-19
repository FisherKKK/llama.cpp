# Day 9: KV缓存机制详解

> 🎯 **学习目标**：深入理解KV缓存的实现原理、优化技巧和内存管理策略。

## 1. 为什么需要KV缓存？

### 1.1 问题背景

在Transformer解码过程中，每生成一个新token都需要：

```
生成第1个token:
  Q1 @ [K1, V1] -> Output1

生成第2个token:
  Q2 @ [K1, K2, V1, V2] -> Output2  ❌ K1, V1 需要重新计算

生成第3个token:
  Q3 @ [K1, K2, K3, V1, V2, V3] -> Output3  ❌ 前面的K, V都要重新计算

...

时间复杂度: O(n²) - 随序列长度平方增长！
```

### 1.2 KV缓存的解决方案

```
使用KV缓存：

Token 1: 计算K1, V1 -> 缓存
Token 2: 计算K2, V2 -> 缓存，使用缓存的[K1, K2], [V1, V2]  ✅
Token 3: 计算K3, V3 -> 缓存，使用缓存的[K1, K2, K3], [V1, V2, V3]  ✅

时间复杂度: O(n) - 线性增长！
内存开销: O(n * n_layer * n_embd) - 需要额外内存
```

### 1.3 性能对比

```
以7B模型、生成2048 tokens为例：

┌─────────────┬───────────────┬───────────────┬─────────┐
│ 方法        │ 计算量        │ 生成速度      │ 内存    │
├─────────────┼───────────────┼───────────────┼─────────┤
│ 无KV缓存    │ ~4M FLOPs     │ 0.5 tok/s     │ 14GB    │
│ 有KV缓存    │ ~2K FLOPs     │ 50 tok/s      │ 16GB    │
├─────────────┼───────────────┼───────────────┼─────────┤
│ 加速比      │ 2000x         │ 100x          │ +14%    │
└─────────────┴───────────────┴───────────────┴─────────┘

结论：用少量内存换取巨大的性能提升！
```

## 2. KV缓存数据结构

### 2.1 核心结构

```cpp
// 文件: src/llama-kv-cache.h

struct llama_kv_cache {
    const llama_model& model;
    const llama_hparams& hparams;

    // 缓存维度信息
    bool   v_trans;        // V是否转置存储
    uint32_t n_seq_max;    // 最大序列数（批处理）
    uint32_t n_stream;     // 流数量（统一/独立）
    uint32_t n_pad;        // padding大小
    uint32_t n_swa;        // Sliding Window Attention大小
    llama_swa_type swa_type;  // SWA类型

    // 每层的KV张量
    struct layer {
        uint32_t il;                       // 层索引
        ggml_tensor* k;                    // K缓存张量 [n_embd_k, kv_size, n_stream]
        ggml_tensor* v;                    // V缓存张量 [n_embd_v, kv_size, n_stream]
        std::vector<ggml_tensor*> k_stream; // 每个stream的K视图
        std::vector<ggml_tensor*> v_stream; // 每个stream的V视图
    };
    std::vector<layer> layers;             // 所有层的缓存

    // 缓存单元管理
    std::vector<std::vector<llama_kv_cell>> v_cells;  // [stream][pos]
    std::vector<uint32_t> v_heads;                     // 每个stream的当前头位置

    // 序列到stream的映射
    std::vector<uint32_t> seq_to_stream;               // [seq_id] -> stream_id

    // 层ID映射
    std::map<uint32_t, uint32_t> map_layer_ids;       // layer_id -> index

    // 内存管理
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_t>> ctxs_bufs;
};
```

### 2.2 缓存单元

```cpp
// 文件: src/llama-kv-cells.h

struct llama_kv_cell {
    llama_pos pos   = -1;  // 在序列中的位置
    llama_pos delta = 0;   // RoPE delta（用于位置编码）

    int32_t src = -1;      // 源单元（用于复制）
    int32_t tail = -1;     // 链表尾（用于管理）

    std::set<llama_seq_id> seq_id;  // 此单元属于哪些序列

    bool has_seq_id(const llama_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }

    bool is_empty() const {
        return seq_id.empty();
    }

    bool is_same_seq(const llama_kv_cell & other) const {
        return seq_id == other.seq_id;
    }
};
```

### 2.3 内存布局

```
对于一个32层、hidden_dim=4096、n_head=32的模型：

K缓存每层: [n_embd_k, kv_size, n_stream]
         = [4096, 2048, 1]
         = 33MB (FP16)

V缓存每层: [n_embd_v, kv_size, n_stream]
         = [4096, 2048, 1]
         = 33MB (FP16)

总KV缓存: (33MB + 33MB) * 32层 = 2.1GB

量化到Q8_0: 2.1GB * 0.5 = 1.05GB  ✅ 节省50%内存
量化到Q4_0: 2.1GB * 0.25 = 525MB  ✅ 节省75%内存

物理布局（连续内存）：
┌──────────────────────────────────────┐
│ Layer 0                              │
│  ┌────────────────────────────────┐ │
│  │ K: [4096 x 2048]               │ │
│  └────────────────────────────────┘ │
│  ┌────────────────────────────────┐ │
│  │ V: [4096 x 2048]               │ │
│  └────────────────────────────────┘ │
├──────────────────────────────────────┤
│ Layer 1                              │
│  ┌────────────────────────────────┐ │
│  │ K: [4096 x 2048]               │ │
│  └────────────────────────────────┘ │
│  ┌────────────────────────────────┐ │
│  │ V: [4096 x 2048]               │ │
│  └────────────────────────────────┘ │
├──────────────────────────────────────┤
│ ...                                  │
└──────────────────────────────────────┘
```

## 3. KV缓存实现

### 3.1 初始化

```cpp
// 文件: src/llama-kv-cache.cpp

llama_kv_cache::llama_kv_cache(
        const llama_model& model,
        ggml_type type_k,              // K的数据类型
        ggml_type type_v,              // V的数据类型
        bool v_trans,                  // V是否转置
        bool offload,                  // 是否卸载到GPU
        bool unified,                  // 是否统一缓存（多序列）
        uint32_t kv_size,              // 缓存大小
        uint32_t n_seq_max,            // 最大序列数
        uint32_t n_pad,                // padding
        uint32_t n_swa,                // SWA大小
        llama_swa_type swa_type,       // SWA类型
        const layer_filter_cb& filter, // 层过滤
        const layer_reuse_cb& reuse)   // 层复用
    : model(model), hparams(model.hparams),
      v_trans(v_trans), n_seq_max(n_seq_max),
      n_stream(unified ? 1 : n_seq_max),
      n_pad(n_pad), n_swa(n_swa), swa_type(swa_type) {

    // 检查对齐
    GGML_ASSERT(kv_size % n_pad == 0);

    // 初始化stream
    v_heads.resize(n_stream, 0);
    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // 序列到stream映射
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);
    if (n_stream > 1) {
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // 为每层创建K、V张量
    for (uint32_t il = 0; il < hparams.n_layer; il++) {
        if (!hparams.has_kv(il)) continue;
        if (filter && !filter(il)) continue;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ?
            hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        // 选择设备
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (offload) {
            auto* dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);
        }

        // 创建上下文
        ggml_context* ctx = ctx_for_buft(buft);

        // 创建K、V张量
        ggml_tensor* k = ggml_new_tensor_3d(ctx, type_k,
                                           n_embd_k_gqa, kv_size, n_stream);
        ggml_tensor* v = ggml_new_tensor_3d(ctx, type_v,
                                           n_embd_v_gqa, kv_size, n_stream);

        ggml_format_name(k, "cache_k_l%d", il);
        ggml_format_name(v, "cache_v_l%d", il);

        // 为每个stream创建视图
        std::vector<ggml_tensor*> k_stream, v_stream;
        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(ggml_view_2d(ctx, k,
                n_embd_k_gqa, kv_size, k->nb[1], s * k->nb[2]));
            v_stream.push_back(ggml_view_2d(ctx, v,
                n_embd_v_gqa, kv_size, v->nb[1], s * v->nb[2]));
        }

        map_layer_ids[il] = layers.size();
        layers.push_back({il, k, v, k_stream, v_stream});
    }

    // 分配内存并初始化为0
    for (auto& [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx.get(), buft);
        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }
}
```

### 3.2 更新缓存

```cpp
// 文件: src/llama-kv-cache.cpp

void llama_kv_cache_update(llama_kv_cache& cache) {
    // 遍历所有stream
    for (uint32_t s = 0; s < cache.n_stream; ++s) {
        uint32_t& head = cache.v_heads[s];
        auto& cells = cache.v_cells[s];

        // Sliding Window Attention处理
        if (cache.n_swa > 0 && head >= cache.n_swa) {
            // 移动窗口
            const uint32_t swa_offset = head - cache.n_swa + 1;

            // 清理超出窗口的单元
            for (uint32_t i = 0; i < swa_offset; ++i) {
                cells[i].pos = -1;
                cells[i].seq_id.clear();
            }

            // 调整位置
            for (uint32_t i = swa_offset; i < head; ++i) {
                if (cells[i].pos >= 0) {
                    cells[i].pos -= swa_offset;
                }
            }

            head = cache.n_swa - 1;
        }
    }
}
```

### 3.3 序列操作

```cpp
// 添加序列
void llama_kv_cache_seq_add(
        llama_kv_cache& cache,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        llama_pos delta) {

    uint32_t s = cache.seq_to_stream[seq_id];
    auto& cells = cache.v_cells[s];

    for (uint32_t i = 0; i < cells.size(); ++i) {
        auto& cell = cells[i];

        if (cell.has_seq_id(seq_id) &&
            cell.pos >= p0 && cell.pos < p1) {
            cell.delta += delta;
            cell.pos += delta;
        }
    }
}

// 移除序列
void llama_kv_cache_seq_rm(
        llama_kv_cache& cache,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {

    uint32_t s = cache.seq_to_stream[seq_id];
    auto& cells = cache.v_cells[s];

    for (uint32_t i = 0; i < cells.size(); ++i) {
        auto& cell = cells[i];

        if (cell.has_seq_id(seq_id) &&
            (p0 < 0 || (cell.pos >= p0 && cell.pos < p1))) {
            cell.seq_id.erase(seq_id);

            if (cell.is_empty()) {
                cell.pos = -1;
            }
        }
    }
}

// 复制序列
void llama_kv_cache_seq_cp(
        llama_kv_cache& cache,
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) {

    uint32_t s_src = cache.seq_to_stream[seq_id_src];
    uint32_t s_dst = cache.seq_to_stream[seq_id_dst];

    if (s_src == s_dst) {
        // 同一stream，直接添加序列ID
        auto& cells = cache.v_cells[s_src];
        for (auto& cell : cells) {
            if (cell.has_seq_id(seq_id_src) &&
                (p0 < 0 || (cell.pos >= p0 && cell.pos < p1))) {
                cell.seq_id.insert(seq_id_dst);
            }
        }
    } else {
        // 不同stream，需要复制数据
        // 实现略...
    }
}
```

## 4. 计算图中的KV缓存

### 4.1 在注意力中使用

```cpp
// 文件: src/llama-graph.cpp

static struct ggml_tensor * llm_build_kv(
        struct ggml_context * ctx,
        const llama_model& model,
        const llama_hparams& hparams,
        const llama_ubatch& batch,
        struct ggml_tensor * k_cur,  // 当前的K [n_embd_k_gqa, n_tokens]
        struct ggml_tensor * v_cur,  // 当前的V [n_embd_v_gqa, n_tokens]
        struct ggml_tensor * kv_pe,  // 位置编码
        llama_kv_cache& kv_cache,
        int il) {                     // 层索引

    const auto& layer = kv_cache.layers[kv_cache.map_layer_ids.at(il)];
    const uint32_t n_tokens = batch.n_tokens;

    // 1. 应用RoPE到当前K
    if (kv_pe) {
        k_cur = ggml_rope_ext(..., k_cur, kv_pe, ...);
    }

    // 2. 更新KV缓存
    // 这是一个自定义操作，将k_cur和v_cur写入缓存
    struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, layer.k,
        n_tokens * n_embd_k_gqa, /* offset = */ ...);
    struct ggml_tensor * v_cache_view = ggml_view_1d(ctx, layer.v,
        n_tokens * n_embd_v_gqa, /* offset = */ ...);

    ggml_build_forward_expand(&gf,
        ggml_cpy(ctx, k_cur, k_cache_view));
    ggml_build_forward_expand(&gf,
        ggml_cpy(ctx, v_cur, v_cache_view));

    // 3. 获取所有需要的K和V（包括缓存的）
    struct ggml_tensor * k = ggml_view_2d(ctx, layer.k,
        n_embd_k_gqa, n_kv, /* stride = */ ..., /* offset = */ ...);
    struct ggml_tensor * v = ggml_view_2d(ctx, layer.v,
        n_kv, n_embd_v_gqa, /* stride = */ ..., /* offset = */ ...);

    return {k, v};
}

// 使用示例
static struct ggml_tensor * llm_build_attn(
        struct ggml_context * ctx,
        ...,
        struct ggml_tensor * q_cur,  // [n_embd, n_tokens]
        struct ggml_tensor * k_cur,  // [n_embd_k_gqa, n_tokens]
        struct ggml_tensor * v_cur,  // [n_embd_v_gqa, n_tokens]
        ...) {

    // 获取完整的K和V（包括缓存）
    auto [k, v] = llm_build_kv(ctx, model, hparams, batch,
                                k_cur, v_cur, kv_pe, kv_cache, il);

    // Q @ K^T
    struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);  // [n_tokens, n_kv]

    // Mask (可选)
    if (mask) {
        kq = ggml_add(ctx, kq, mask);
    }

    // Softmax
    kq = ggml_soft_max_ext(ctx, kq, ...);

    // @ V
    struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);  // [n_embd_v, n_tokens]

    return kqv;
}
```

## 5. 优化技巧

### 5.1 量化KV缓存

```cpp
// 不同量化级别的权衡
┌─────────┬──────────┬──────────┬──────────┐
│ 类型    │ 精度     │ 大小     │ 质量     │
├─────────┼──────────┼──────────┼──────────┤
│ F16     │ 16-bit   │ 2.1GB    │ 100%     │
│ Q8_0    │ 8-bit    │ 1.05GB   │ 99.5%    │
│ Q6_K    │ 6-bit    │ 800MB    │ 98%      │
│ Q4_0    │ 4-bit    │ 525MB    │ 95%      │
└─────────┴──────────┴──────────┴──────────┘

使用方法：
llama_context_params cparams = llama_context_default_params();
cparams.type_k = GGML_TYPE_Q8_0;  // 量化K
cparams.type_v = GGML_TYPE_Q8_0;  // 量化V
```

### 5.2 V转置存储

```cpp
// 问题：V在注意力计算时需要转置
// attn = softmax(Q @ K^T) @ V
//                            ↑ V需要按列访问

// 方案1：运行时转置（慢）
V_transposed = ggml_transpose(ctx, V);  // 每次都转置

// 方案2：预转置存储（快）
// 在缓存时就转置V，节省运行时开销
cparams.v_trans = true;

// 内存布局对比：
正常: V [n_embd_v, n_kv]    按行存储 [v0_0, v0_1, ..., v1_0, v1_1, ...]
转置: V [n_kv, n_embd_v]    按列存储 [v0_0, v1_0, ..., v0_1, v1_1, ...]
                            ↑ 访问时缓存友好！
```

### 5.3 Sliding Window Attention (SWA)

```cpp
// 长序列问题：KV缓存线性增长
// 2048 tokens: 2.1GB
// 4096 tokens: 4.2GB
// 8192 tokens: 8.4GB  ❌ 内存爆炸

// SWA解决方案：只保留最近的N个tokens
cparams.n_swa = 2048;  // 窗口大小

// 效果：
┌──────────────────────────────────────┐
│ Tokens: [0, 1, 2, ..., 8191]        │
│                                      │
│ Window: [6144, 6145, ..., 8191]     │← 只保留这部分
│         └─────── 2048 ──────┘       │
│                                      │
│ Memory: 固定2.1GB ✅                 │
└──────────────────────────────────────┘

// 性能影响：
• 对长序列：几乎无影响
• 对依赖远距离上下文的任务：可能有影响
```

### 5.4 批处理优化

```cpp
// 问题：多个序列并行时如何管理KV缓存？

// 方案1：独立缓存（每个序列一个stream）
n_stream = n_seq_max;
// 优点：完全隔离，灵活
// 缺点：内存占用大

// 方案2：统一缓存（所有序列共享一个stream）
n_stream = 1;
// 优点：内存节省
// 缺点：需要仔细管理seq_id

// 示例：批处理3个序列
llama_batch batch = llama_batch_init(512, 0, 3);

// 序列0: "Hello"
batch.token[0] = token_hello;
batch.pos[0] = 0;
batch.n_seq_id[0] = 1;
batch.seq_id[0][0] = 0;  // seq_id = 0

// 序列1: "World"
batch.token[1] = token_world;
batch.pos[1] = 0;
batch.n_seq_id[1] = 1;
batch.seq_id[1][0] = 1;  // seq_id = 1

// ...

llama_decode(ctx, batch);
// KV缓存自动管理每个序列的位置
```

## 6. 实践练习

### 练习1：查看KV缓存大小

```cpp
#include "llama.h"
#include <stdio.h>

void print_kv_cache_info(llama_context* ctx) {
    // 获取缓存大小
    size_t size_k = llama_get_state_size(ctx) / 2;  // 近似
    size_t size_v = llama_get_state_size(ctx) / 2;

    printf("KV Cache Info:\n");
    printf("  K size: %.2f MB\n", size_k / 1024.0 / 1024.0);
    printf("  V size: %.2f MB\n", size_v / 1024.0 / 1024.0);
    printf("  Total: %.2f MB\n", (size_k + size_v) / 1024.0 / 1024.0);
}
```

### 练习2：测试不同量化级别

```bash
#!/bin/bash

for type in "f16" "q8_0" "q6_k" "q4_0"; do
    echo "Testing type_k=$type"
    time ./llama-cli -m model.gguf \
        --ctx-size 2048 \
        --cache-type-k $type \
        --cache-type-v $type \
        -p "Hello world" -n 100
done
```

### 练习3：实现简单的缓存管理

```cpp
// 简化版KV缓存管理
struct SimpleKVCache {
    std::vector<std::vector<float>> k_cache;  // [n_layer][...]
    std::vector<std::vector<float>> v_cache;
    size_t current_pos = 0;
    size_t capacity = 2048;

    void update(int layer, const std::vector<float>& k, const std::vector<float>& v) {
        if (current_pos >= capacity) {
            // 简单策略：覆盖最旧的
            current_pos = 0;
        }

        // 复制到缓存
        // ... 实现略
        current_pos++;
    }

    std::pair<std::vector<float>, std::vector<float>> get(int layer) {
        // 返回当前所有缓存的K和V
        // ... 实现略
    }
};
```

## 7. 调试技巧

```cpp
// 1. 打印缓存使用情况
void debug_kv_cache(llama_context* ctx) {
    // 获取使用的单元数
    int used_cells = 0;
    // ... 遍历cells统计

    printf("KV Cache Usage: %d / %d (%.1f%%)\n",
           used_cells, total_cells,
           100.0 * used_cells / total_cells);
}

// 2. 验证缓存一致性
void verify_kv_cache(llama_context* ctx) {
    // 检查每个单元的seq_id是否有效
    // 检查pos是否在合理范围内
    // ... 实现略
}

// 3. 可视化缓存状态
void visualize_kv_cache(llama_context* ctx) {
    // 打印每个位置的占用情况
    // X = 已占用, . = 空闲
    // [XXXXXX.................XXXXX]
    // ... 实现略
}
```

## 8. 常见问题

**Q1: KV缓存为什么这么大？**
- 每个token在每层都要保存K和V
- 7B模型32层，每层hidden_dim=4096
- 2048 tokens × 32层 × 4096 × 2 (K+V) × 2字节(FP16) = 2.1GB

**Q2: 能否只缓存部分层？**
可以，使用layer_filter回调：
```cpp
auto filter = [](int layer) {
    return layer % 2 == 0;  // 只缓存偶数层
};
kv_cache = llama_kv_cache(model, ..., filter, nullptr);
```

**Q3: 如何选择量化级别？**
```
Q8_0: 最佳选择，损失<0.5%，节省50%内存
Q6_K: 激进优化，损失~2%，节省62.5%内存
Q4_0: 极限压缩，损失~5%，节省75%内存
```

**Q4: Unified vs Independent 缓存？**
```
Unified (n_stream=1):
- 适合：单序列或少量序列
- 优点：内存最省
- 缺点：管理复杂

Independent (n_stream=n_seq):
- 适合：大量并行序列
- 优点：管理简单
- 缺点：内存占用大
```

## 9. 下一步

**Day 10 预告：注意力机制优化**
- FlashAttention原理
- 分组查询注意力(GQA)
- 多查询注意力(MQA)
- 位置编码优化

## 作业

1. ✏️ 计算你的模型需要多大KV缓存
2. 🔍 使用gdb查看实际的缓存布局
3. 📖 阅读 `src/llama-kv-cache.cpp` 完整实现
4. 💻 实现一个简单的KV缓存可视化工具

---

**继续学习**: [Day 10: 注意力机制优化](day10-attention-optimization.md) →

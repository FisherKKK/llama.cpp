# Day 7: 计算图构建与调度

## 课程目标

深入理解计算图的构建与执行：
- llama_graph_builder 实现
- 构建完整的 Transformer 计算图
- 后端调度器（Scheduler）原理
- 图优化技术
- 多后端协同工作

## 1. 计算图回顾

### 1.1 为什么需要计算图？

**传统立即执行**：
```cpp
// 每个操作立即计算
Tensor c = add(a, b);    // 立即执行
Tensor d = mul(c, 2);    // 立即执行
Tensor e = relu(d);      // 立即执行
```

**计算图延迟执行**：
```cpp
// 只构建图结构，不计算
Tensor c = ggml_add(ctx, a, b);    // 只记录操作
Tensor d = ggml_mul(ctx, c, 2);    // 只记录操作
Tensor e = ggml_relu(ctx, d);      // 只记录操作

// 统一执行
ggml_graph_compute(graph);         // 一次性执行所有操作
```

**优势**：
- ✅ 全局优化：可以分析整个图，进行融合优化
- ✅ 内存管理：可以重用中间结果的内存
- ✅ 并行调度：可以并行执行无依赖的操作
- ✅ 跨设备：可以智能分配不同设备

## 2. 计算图结构详解

### 2.1 图的数据结构

```c
// 位置：ggml/include/ggml.h:625
struct ggml_cgraph {
    int size;          // 图的最大容量
    int n_nodes;       // 当前节点数
    int n_leafs;       // 叶子节点数（输入）

    struct ggml_tensor ** nodes;   // 操作节点（按拓扑序）
    struct ggml_tensor ** grads;   // 梯度（训练用）
    struct ggml_tensor ** leafs;   // 输入节点

    struct ggml_hash_set visited_hash_set;  // 已访问标记

    // 执行顺序
    enum ggml_cgraph_eval_order order;

    // 性能统计
    int perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;
};

enum ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
    GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
    GGML_CGRAPH_EVAL_ORDER_COUNT
};
```

### 2.2 图的构建流程

```
用户代码：
    a = ggml_new_tensor(ctx)
    b = ggml_new_tensor(ctx)
    c = ggml_add(ctx, a, b)
    d = ggml_mul(ctx, c, 2)

    ↓ 构建图

ggml_build_forward_expand(graph, d)

    ↓ 生成的图结构

Leafs: [a, b]              # 输入
Nodes: [c=add(a,b),        # 操作（拓扑序）
        d=mul(c,2)]

    ↓ 执行

ggml_graph_compute(graph)
```

### 2.3 图构建代码

```c
// 位置：ggml/src/ggml.c:18467
void ggml_build_forward_expand(
    struct ggml_cgraph * cgraph,
    struct ggml_tensor * tensor) {

    if (!tensor) {
        return;
    }

    // 1. 检查是否已访问
    if (ggml_hash_contains(&cgraph->visited_hash_set, tensor)) {
        return;
    }

    // 2. 标记为已访问
    ggml_hash_insert(&cgraph->visited_hash_set, tensor);

    // 3. 递归处理依赖节点
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            ggml_build_forward_expand(cgraph, tensor->src[i]);
        }
    }

    // 4. 添加到图中
    if (tensor->op == GGML_OP_NONE && tensor->grad == NULL) {
        // 叶子节点（输入）
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);
        cgraph->leafs[cgraph->n_leafs++] = tensor;
    } else {
        // 操作节点
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);
        cgraph->nodes[cgraph->n_nodes++] = tensor;
    }
}
```

## 3. llama_graph_builder 实现

### 3.1 Graph Builder 结构

```cpp
// 位置：src/llama-graph.cpp:146
struct llama_graph_builder {
    llama_context & lctx;
    const llama_model & model;
    const llama_ubatch & ubatch;

    // 缓存的张量
    struct ggml_tensor * inp_tokens;   // 输入 token IDs
    struct ggml_tensor * inp_embd;     // 输入 embeddings
    struct ggml_tensor * inp_pos;      // 位置索引
    struct ggml_tensor * inp_KQ_mask;  // 注意力掩码
    struct ggml_tensor * inp_K_shift;  // KV 缓存偏移

    // 中间结果
    struct ggml_tensor * cur;          // 当前激活
    struct ggml_tensor * inpL;         // 层输入

    // KV 缓存视图
    std::vector<struct ggml_tensor *> kv_heads;
};
```

### 3.2 构建 Transformer 层

```cpp
// 位置：src/llama-graph.cpp:823
struct ggml_tensor * llama_graph_builder::build_layer(
    int il,  // 层索引
    struct ggml_tensor * cur,
    struct ggml_tensor * attn_mask) {

    const auto & model = lctx.model;
    const auto & hparams = model.hparams;
    const auto & layer = model.layers[il];

    struct ggml_context * ctx = lctx.ctx_compute.get();

    // === 1. 注意力部分 ===

    // 1.1 LayerNorm
    struct ggml_tensor * attn_norm = ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps);
    attn_norm = ggml_mul(ctx, attn_norm, layer.attn_norm);

    // 1.2 Q, K, V 投影
    struct ggml_tensor * Qcur = ggml_mul_mat(ctx, layer.wq, attn_norm);
    struct ggml_tensor * Kcur = ggml_mul_mat(ctx, layer.wk, attn_norm);
    struct ggml_tensor * Vcur = ggml_mul_mat(ctx, layer.wv, attn_norm);

    // 1.3 Reshape Q, K, V
    // Q: [n_embd, n_tokens] → [n_head, n_embd_head, n_tokens]
    Qcur = ggml_reshape_3d(ctx, Qcur,
                           hparams.n_embd_head,
                           hparams.n_head,
                           ubatch.n_tokens);

    // 1.4 应用 RoPE
    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos,
                         hparams.n_rot, hparams.rope_type,
                         hparams.rope_freq_base, hparams.rope_freq_scale);

    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos,
                         hparams.n_rot, hparams.rope_type,
                         hparams.rope_freq_base, hparams.rope_freq_scale);

    // 1.5 存储 K, V 到缓存
    struct ggml_tensor * k_cache = kv_heads[il * 2 + 0];
    struct ggml_tensor * v_cache = kv_heads[il * 2 + 1];

    ggml_build_forward_expand(lctx.gf,
        ggml_cpy(ctx, Kcur, k_cache));
    ggml_build_forward_expand(lctx.gf,
        ggml_cpy(ctx, Vcur, v_cache));

    // 1.6 计算注意力
    // scores = Q @ K^T / sqrt(d_k)
    struct ggml_tensor * kq = ggml_mul_mat(ctx, k_cache, Qcur);
    kq = ggml_scale(ctx, kq, 1.0f / sqrtf(hparams.n_embd_head));

    // 应用掩码
    kq = ggml_add(ctx, kq, attn_mask);

    // Softmax
    kq = ggml_soft_max(ctx, kq);

    // attn_out = softmax(scores) @ V
    struct ggml_tensor * attn_out = ggml_mul_mat(ctx, v_cache, kq);

    // 1.7 Reshape 回原始形状
    attn_out = ggml_reshape_2d(ctx, attn_out,
                               hparams.n_embd, ubatch.n_tokens);

    // 1.8 输出投影
    attn_out = ggml_mul_mat(ctx, layer.wo, attn_out);

    // 1.9 残差连接
    cur = ggml_add(ctx, cur, attn_out);

    // === 2. FFN 部分 ===

    // 2.1 LayerNorm
    struct ggml_tensor * ffn_norm = ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps);
    ffn_norm = ggml_mul(ctx, ffn_norm, layer.ffn_norm);

    // 2.2 FFN (SwiGLU: gate * silu(up))
    struct ggml_tensor * ffn_gate = ggml_mul_mat(ctx, layer.ffn_gate, ffn_norm);
    struct ggml_tensor * ffn_up = ggml_mul_mat(ctx, layer.ffn_up, ffn_norm);

    ffn_gate = ggml_silu(ctx, ffn_gate);
    struct ggml_tensor * ffn_hidden = ggml_mul(ctx, ffn_gate, ffn_up);

    // 2.3 Down 投影
    struct ggml_tensor * ffn_out = ggml_mul_mat(ctx, layer.ffn_down, ffn_hidden);

    // 2.4 残差连接
    cur = ggml_add(ctx, cur, ffn_out);

    return cur;
}
```

### 3.3 构建完整的模型图

```cpp
// 位置：src/llama-graph.cpp:1456
struct ggml_cgraph * llama_graph_builder::build_llama() {
    const auto & model = lctx.model;
    const auto & hparams = model.hparams;

    struct ggml_context * ctx = lctx.ctx_compute.get();

    // 1. 创建输入张量
    inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ubatch.n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);

    inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ubatch.n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    // 2. Token 嵌入
    inpL = ggml_get_rows(ctx, model.tok_embd, inp_tokens);

    // 3. 构建注意力掩码
    struct ggml_tensor * KQ_mask = build_attention_mask(ubatch);

    // 4. 逐层构建
    for (int il = 0; il < hparams.n_layer; il++) {
        inpL = build_layer(il, inpL, KQ_mask);

        // 可选：中间层输出
        if (ubatch.output_layer[il]) {
            ggml_build_forward_expand(lctx.gf, inpL);
        }
    }

    // 5. 最终 LayerNorm
    inpL = ggml_rms_norm(ctx, inpL, hparams.f_norm_rms_eps);
    inpL = ggml_mul(ctx, inpL, model.output_norm);

    // 6. 输出投影（logits）
    struct ggml_tensor * logits = ggml_mul_mat(ctx, model.output, inpL);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    // 7. 构建图
    ggml_build_forward_expand(lctx.gf, logits);

    return lctx.gf;
}
```

## 4. 后端调度器

### 4.1 调度器结构

```c
// 位置：ggml/src/ggml-backend.cpp:1623
struct ggml_backend_sched {
    int n_backends;
    ggml_backend_t * backends;           // 后端列表

    // 张量 → 后端映射
    struct ggml_hash_set hash_set;
    ggml_backend_t * tensor_backend;     // 张量所在后端

    // 分割信息
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // 同步事件
    struct ggml_backend_event * events;

    // 图副本（每个后端一份）
    struct ggml_cgraph ** graph_copies;
};
```

### 4.2 调度流程

```
┌─────────────────────────────────────────┐
│ 1. 分析图，确定每个节点的后端           │
│    tensor_backend[t] = best_backend(t)  │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 2. 将图分割成多个 split                 │
│    每个 split 在单个后端上执行          │
│                                         │
│    Split 0 (CPU):  [op0, op1]          │
│    Split 1 (GPU0): [op2, op3, op4]     │
│    Split 2 (CPU):  [op5]               │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 3. 插入数据传输操作                     │
│    CPU → GPU: copy_tensor(op1_out)     │
│    GPU → CPU: copy_tensor(op4_out)     │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 4. 顺序执行每个 split                   │
│    for each split:                     │
│      backend.compute(split.ops)        │
│      sync_if_needed()                  │
└─────────────────────────────────────────┘
```

### 4.3 调度器实现

```c
// 位置：ggml/src/ggml-backend.cpp:2156
enum ggml_status ggml_backend_sched_graph_compute(
    struct ggml_backend_sched * sched,
    struct ggml_cgraph * graph) {

    // 1. 分配张量到后端
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // 选择最佳后端
        ggml_backend_t backend = ggml_backend_sched_backend_from_cur(sched, node);

        // 如果张量还没有后端，分配一个
        if (node->backend == NULL || node->backend != backend) {
            ggml_backend_sched_alloc_tensor(sched, backend, node);
        }
    }

    // 2. 分割图
    ggml_backend_sched_split_graph(sched, graph);

    // 3. 执行每个 split
    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        ggml_backend_t backend = split->backend;

        // 3.1 拷贝输入（如果需要跨设备传输）
        for (int j = 0; j < split->n_inputs; j++) {
            struct ggml_tensor * input = split->inputs[j];
            struct ggml_tensor * input_cpy = split->inputs_cpy[j];

            if (input_cpy != NULL) {
                // 跨设备拷贝
                ggml_backend_tensor_copy(input, input_cpy);
            }
        }

        // 3.2 执行计算
        enum ggml_status status = ggml_backend_graph_compute(
            backend, split->graph);

        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }

        // 3.3 同步（如果需要）
        if (split->sync_needed) {
            ggml_backend_synchronize(backend);
        }
    }

    return GGML_STATUS_SUCCESS;
}
```

### 4.4 后端选择策略

```c
// 位置：ggml/src/ggml-backend.cpp:1923
static ggml_backend_t ggml_backend_sched_backend_from_cur(
    struct ggml_backend_sched * sched,
    struct ggml_tensor * tensor) {

    // 1. 检查是否已分配
    ggml_backend_t cur_backend = ggml_get_backend(sched, tensor);
    if (cur_backend != NULL) {
        return cur_backend;
    }

    // 2. 根据操作类型选择
    if (tensor->op == GGML_OP_NONE) {
        // 输入张量，使用第一个后端
        return sched->backends[0];
    }

    // 3. 检查是否有后端支持此操作
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_t backend = sched->backends[i];

        if (ggml_backend_supports_op(backend, tensor)) {
            // 优先选择 GPU（如果可用）
            if (ggml_backend_is_gpu(backend)) {
                return backend;
            }
        }
    }

    // 4. 默认使用 CPU
    return sched->backends[0];
}
```

## 5. 图优化技术

### 5.1 操作融合

将多个操作融合为一个，减少内存访问：

```c
// 优化前：3 个独立操作
y = ggml_add(ctx, x, bias);      // 写 y
y = ggml_mul(ctx, y, scale);     // 读 y，写 y
y = ggml_relu(ctx, y);           // 读 y，写 y

// 优化后：1 个融合操作
y = ggml_add_mul_relu(ctx, x, bias, scale);  // 直接计算
```

**实现**（位置：`ggml/src/ggml.c:17234`）：

```c
static void ggml_graph_optimize(struct ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes - 2; i++) {
        struct ggml_tensor * node0 = graph->nodes[i];
        struct ggml_tensor * node1 = graph->nodes[i + 1];
        struct ggml_tensor * node2 = graph->nodes[i + 2];

        // 检测模式：Add → Mul → ReLU
        if (node0->op == GGML_OP_ADD &&
            node1->op == GGML_OP_MUL &&
            node2->op == GGML_OP_RELU &&
            node1->src[0] == node0 &&
            node2->src[0] == node1) {

            // 替换为融合操作
            node2->op = GGML_OP_ADD_MUL_RELU_FUSED;
            node2->src[0] = node0->src[0];  // x
            node2->src[1] = node0->src[1];  // bias
            node2->src[2] = node1->src[1];  // scale

            // 标记中间节点为无效
            node0->op = GGML_OP_NONE;
            node1->op = GGML_OP_NONE;
        }
    }
}
```

### 5.2 内存复用

重用中间结果的内存：

```c
// 分析每个张量的生命周期
for (int i = 0; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];

    // 最后使用位置
    int last_use = find_last_use(graph, node);

    if (last_use < graph->n_nodes - 1) {
        // 可以复用此张量的内存
        mark_for_reuse(node, last_use);
    }
}
```

### 5.3 常量折叠

预计算常量表达式：

```c
// 优化前
struct ggml_tensor * scale = ggml_new_f32(ctx, 0.5f);
struct ggml_tensor * two = ggml_new_f32(ctx, 2.0f);
struct ggml_tensor * result = ggml_mul(ctx, scale, two);  // 0.5 * 2.0

// 优化后（编译时计算）
struct ggml_tensor * result = ggml_new_f32(ctx, 1.0f);
```

## 6. 实战：构建自定义计算图

### 6.1 简单的 MLP

```cpp
struct ggml_cgraph * build_mlp_graph(
    struct ggml_context * ctx,
    struct ggml_tensor * input,    // [batch, input_dim]
    struct ggml_tensor * w1,       // [input_dim, hidden_dim]
    struct ggml_tensor * b1,       // [hidden_dim]
    struct ggml_tensor * w2,       // [hidden_dim, output_dim]
    struct ggml_tensor * b2) {     // [output_dim]

    // 层 1: hidden = relu(input @ w1 + b1)
    struct ggml_tensor * hidden = ggml_mul_mat(ctx, w1, input);
    hidden = ggml_add(ctx, hidden, b1);
    hidden = ggml_relu(ctx, hidden);

    // 层 2: output = hidden @ w2 + b2
    struct ggml_tensor * output = ggml_mul_mat(ctx, w2, hidden);
    output = ggml_add(ctx, output, b2);

    // 构建图
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    return gf;
}
```

### 6.2 带残差连接的块

```cpp
struct ggml_tensor * build_residual_block(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * w1,
    struct ggml_tensor * w2) {

    // 主路径
    struct ggml_tensor * x = ggml_mul_mat(ctx, w1, input);
    x = ggml_relu(ctx, x);
    x = ggml_mul_mat(ctx, w2, x);

    // 残差连接
    x = ggml_add(ctx, x, input);

    return x;
}
```

### 6.3 多头注意力

```cpp
struct ggml_tensor * build_multi_head_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * input,      // [n_tokens, n_embd]
    struct ggml_tensor * wq,         // [n_embd, n_embd]
    struct ggml_tensor * wk,
    struct ggml_tensor * wv,
    struct ggml_tensor * wo,
    int n_head,
    int n_embd_head) {

    // Q, K, V 投影
    struct ggml_tensor * Q = ggml_mul_mat(ctx, wq, input);
    struct ggml_tensor * K = ggml_mul_mat(ctx, wk, input);
    struct ggml_tensor * V = ggml_mul_mat(ctx, wv, input);

    // Reshape 为多头
    Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head, -1);
    K = ggml_reshape_3d(ctx, K, n_embd_head, n_head, -1);
    V = ggml_reshape_3d(ctx, V, n_embd_head, n_head, -1);

    // 缩放点积注意力
    struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
    KQ = ggml_scale(ctx, KQ, 1.0f / sqrtf(n_embd_head));
    KQ = ggml_soft_max(ctx, KQ);

    struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);

    // Reshape 回原始形状
    KQV = ggml_reshape_2d(ctx, KQV, n_head * n_embd_head, -1);

    // 输出投影
    struct ggml_tensor * output = ggml_mul_mat(ctx, wo, KQV);

    return output;
}
```

## 7. 调试工具

### 7.1 图可视化

```cpp
void dump_graph_to_dot(struct ggml_cgraph * graph, const char * filename) {
    FILE * fp = fopen(filename, "w");
    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  rankdir=TB;\n");

    // 输出节点
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        fprintf(fp, "  node%d [label=\"%s\\n%s\\n[%lld,%lld,%lld,%lld]\"];\n",
                i, node->name,
                ggml_op_name(node->op),
                node->ne[0], node->ne[1], node->ne[2], node->ne[3]);

        // 输出边
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                int src_idx = find_node_index(graph, node->src[j]);
                fprintf(fp, "  node%d -> node%d;\n", src_idx, i);
            }
        }
    }

    fprintf(fp, "}\n");
    fclose(fp);

    printf("Graph dumped to %s\n", filename);
    printf("View with: dot -Tpng %s -o graph.png\n", filename);
}
```

### 7.2 性能分析

```cpp
void profile_graph_execution(struct ggml_cgraph * graph) {
    printf("Graph profiling:\n");
    printf("%-40s %12s %12s\n", "Operation", "Time (ms)", "Percentage");
    printf("─────────────────────────────────────────────────────────────────\n");

    int64_t total_time = graph->perf_time_us;

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        float time_ms = node->perf_time_us / 1000.0f;
        float percentage = 100.0f * node->perf_time_us / total_time;

        printf("%-40s %12.3f %11.1f%%\n",
               node->name, time_ms, percentage);
    }

    printf("─────────────────────────────────────────────────────────────────\n");
    printf("%-40s %12.3f\n", "Total", total_time / 1000.0f);
}
```

## 8. 总结

今天我们深入学习了计算图的构建与调度：

✅ **图构建**：拓扑排序、依赖追踪
✅ **llama_graph_builder**：构建完整的 Transformer 图
✅ **后端调度**：多后端协同、自动分割
✅ **图优化**：操作融合、内存复用、常量折叠
✅ **调试工具**：可视化、性能分析

### 关键要点

1. **延迟执行**：构建时不计算，执行时统一优化
2. **自动调度**：调度器智能分配操作到不同后端
3. **图优化**：全局视角优化性能
4. **可扩展**：易于添加新操作和后端

## 下一步

明天我们将学习 **Day 8: Transformer 架构实现**：
- LLaMA 架构详解
- 注意力层实现
- FFN 层实现
- 不同架构的变体

---

**练习**：
1. 构建一个简单的 3 层 MLP 图
2. 分析 llama-7b 的计算图结构
3. 使用 DOT 可视化你的自定义图

📚 [Day 8: Transformer 架构实现](day08-transformer-impl.md)

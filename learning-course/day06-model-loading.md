# Day 6: 模型加载与权重管理

## 课程目标

深入理解 llama.cpp 的模型加载流程：
- 从 GGUF 文件到 llama_model 结构
- 架构识别与初始化
- 权重映射与管理
- GPU 分层 offload
- 内存优化策略

## 1. 模型加载完整流程

### 1.1 流程概览

```
用户调用
llama_model_load_from_file("model.gguf", params)
    ↓
┌─────────────────────────────────────────┐
│ 1. 打开 GGUF 文件                        │
│    llama_model_loader ml(fname)          │
│    • 解析文件头                          │
│    • 读取元数据                          │
│    • 建立张量索引                        │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 2. 识别模型架构                          │
│    const auto arch = ml.get_arch()       │
│    • 读取 "general.architecture"         │
│    • 匹配架构枚举 (LLM_ARCH_LLAMA)       │
│    • 加载架构特定的键映射                │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 3. 加载超参数                            │
│    llama_hparams hparams                 │
│    • n_vocab, n_embd, n_layer            │
│    • n_head, n_head_kv                   │
│    • RoPE 参数                           │
│    • 归一化参数                          │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 4. 初始化 llama_model 结构               │
│    model = new llama_model()             │
│    model.arch = arch                     │
│    model.hparams = hparams               │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 5. 创建张量骨架                          │
│    llm_load_tensors()                    │
│    • 根据架构创建所有层的张量            │
│    • token_embd, output, norm            │
│    • 32层 × (attn + ffn)                 │
│    • 不分配数据，只设置维度              │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 6. 分配后端缓冲区                        │
│    llama_backend_init()                  │
│    • CPU 后端：malloc/mmap               │
│    • GPU 后端：cudaMalloc/Metal          │
│    • 根据 n_gpu_layers 分配              │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 7. 加载权重数据                          │
│    ml.load_all_data()                    │
│    • 从 GGUF 读取量化数据                │
│    • 拷贝到对应的后端缓冲区              │
│    • GPU 张量：CPU → GPU 传输            │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 8. 初始化词表                            │
│    llama_vocab_init()                    │
│    • 从元数据读取 tokens                 │
│    • 构建 token ↔ ID 映射                │
│    • 初始化 tokenizer                    │
└──────────────┬──────────────────────────┘
               ↓
           返回 model
```

## 2. 核心代码解析

### 2.1 llama_model_load_from_file（入口）

```cpp
// 位置：src/llama-model.cpp:16234
struct llama_model * llama_model_load_from_file(
    const char * path_model,
    struct llama_model_params params) {

    // 1. 初始化后端系统
    ggml_backend_load_all();

    // 2. 创建模型对象
    llama_model * model = new llama_model();

    // 3. 创建加载器
    llama_model_loader ml(path_model, params);

    // 4. 识别架构
    ml.init_mappings();
    model->arch = ml.get_arch();

    // 5. 加载超参数
    llm_load_hparams(ml, model->hparams);

    // 6. 加载词表
    llm_load_vocab(ml, model->vocab);

    // 7. 加载张量
    llm_load_tensors(ml, model, params);

    // 8. 返回
    return model;
}
```

### 2.2 架构识别

```cpp
// 位置：src/llama-model-loader.cpp:456
llm_arch llama_model_loader::get_arch() const {
    // 从元数据读取架构字符串
    const std::string arch_name = get_arch_name();

    // 映射到枚举
    static const std::map<std::string, llm_arch> arch_map = {
        { "llama",    LLM_ARCH_LLAMA },
        { "mistral",  LLM_ARCH_MISTRAL },
        { "qwen",     LLM_ARCH_QWEN },
        { "gemma",    LLM_ARCH_GEMMA },
        { "phi",      LLM_ARCH_PHI },
        // ... 120+ 架构
    };

    auto it = arch_map.find(arch_name);
    if (it == arch_map.end()) {
        throw std::runtime_error("unknown architecture: " + arch_name);
    }

    return it->second;
}

std::string llama_model_loader::get_arch_name() const {
    return get_metadata_string("general.architecture");
}
```

### 2.3 加载超参数

```cpp
// 位置：src/llama-hparams.cpp:78
static void llm_load_hparams(
    llama_model_loader & ml,
    llama_hparams & hparams) {

    const auto arch = ml.get_arch();

    // 基础参数（所有架构通用）
    ml.get_key(LLM_KV_VOCAB_SIZE,        hparams.n_vocab);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,  hparams.n_embd);
    ml.get_key(LLM_KV_BLOCK_COUNT,       hparams.n_layer);
    ml.get_key(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head);

    // KV 头数（多查询注意力）
    if (!ml.get_key(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv, false)) {
        hparams.n_head_kv = hparams.n_head;  // 默认与 Q 相同
    }

    // FFN 参数
    ml.get_key(LLM_KV_FEED_FORWARD_LENGTH, hparams.n_ff);

    // RoPE 参数
    ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE,       hparams.rope_freq_base, false);

    // 归一化参数
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // 架构特定参数
    switch (arch) {
        case LLM_ARCH_LLAMA:
            // LLaMA 特定
            break;

        case LLM_ARCH_MISTRAL:
            // Mistral 支持 Sliding Window Attention
            ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
            break;

        // ... 其他架构
    }

    // 打印加载的参数
    LLAMA_LOG_INFO("n_vocab    = %u\n", hparams.n_vocab);
    LLAMA_LOG_INFO("n_embd     = %u\n", hparams.n_embd);
    LLAMA_LOG_INFO("n_layer    = %u\n", hparams.n_layer);
    LLAMA_LOG_INFO("n_head     = %u\n", hparams.n_head);
    LLAMA_LOG_INFO("n_head_kv  = %u\n", hparams.n_head_kv);
}
```

### 2.4 创建张量骨架

```cpp
// 位置：src/llama-model.cpp:3847
static bool llm_load_tensors(
    llama_model_loader & ml,
    llama_model & model,
    const llama_model_params & params) {

    const auto & hparams = model.hparams;
    const auto arch = model.arch;

    // 1. 创建输入嵌入层
    model.tok_embd = ml.create_tensor(
        "token_embd.weight",
        {hparams.n_embd, hparams.n_vocab},
        GGML_TYPE_F32);

    // 2. 创建输出层
    model.output = ml.create_tensor(
        "output.weight",
        {hparams.n_embd, hparams.n_vocab},
        GGML_TYPE_F32);

    // 3. 创建最终归一化层
    model.output_norm = ml.create_tensor(
        "output_norm.weight",
        {hparams.n_embd},
        GGML_TYPE_F32);

    // 4. 创建所有 Transformer 层
    model.layers.resize(hparams.n_layer);

    for (uint32_t i = 0; i < hparams.n_layer; i++) {
        auto & layer = model.layers[i];

        // 注意力层权重
        layer.attn_norm = ml.create_tensor(
            format("blk.%d.attn_norm.weight", i),
            {hparams.n_embd},
            GGML_TYPE_F32);

        layer.attn_q = ml.create_tensor(
            format("blk.%d.attn_q.weight", i),
            {hparams.n_embd, hparams.n_embd},
            GGML_TYPE_Q4_K);  // 量化类型

        layer.attn_k = ml.create_tensor(
            format("blk.%d.attn_k.weight", i),
            {hparams.n_embd, hparams.n_embd_k},  // KV 头可能不同
            GGML_TYPE_Q4_K);

        layer.attn_v = ml.create_tensor(
            format("blk.%d.attn_v.weight", i),
            {hparams.n_embd, hparams.n_embd_v},
            GGML_TYPE_Q4_K);

        layer.attn_output = ml.create_tensor(
            format("blk.%d.attn_output.weight", i),
            {hparams.n_embd, hparams.n_embd},
            GGML_TYPE_Q4_K);

        // FFN 层权重
        layer.ffn_norm = ml.create_tensor(
            format("blk.%d.ffn_norm.weight", i),
            {hparams.n_embd},
            GGML_TYPE_F32);

        layer.ffn_gate = ml.create_tensor(
            format("blk.%d.ffn_gate.weight", i),
            {hparams.n_embd, hparams.n_ff},
            GGML_TYPE_Q4_K);

        layer.ffn_up = ml.create_tensor(
            format("blk.%d.ffn_up.weight", i),
            {hparams.n_embd, hparams.n_ff},
            GGML_TYPE_Q4_K);

        layer.ffn_down = ml.create_tensor(
            format("blk.%d.ffn_down.weight", i),
            {hparams.n_ff, hparams.n_embd},
            GGML_TYPE_Q4_K);
    }

    return true;
}
```

## 3. GPU 分层 Offload

### 3.1 Offload 策略

```cpp
// 位置：src/llama-model.cpp:4821
static void llm_load_tensors_offload(
    llama_model & model,
    const llama_model_params & params) {

    const int n_gpu = params.n_gpu_layers;
    const int n_layer = model.hparams.n_layer;

    if (n_gpu == 0) {
        // 纯 CPU 模式
        for (auto & tensor : model.tensors) {
            tensor->backend = GGML_BACKEND_CPU;
        }
        return;
    }

    // 决定哪些层在 GPU 上
    int layer_gpu_count = std::min(n_gpu, n_layer);

    LLAMA_LOG_INFO("offloading %d/%d layers to GPU\n",
                   layer_gpu_count, n_layer);

    // 输入/输出层始终在 GPU（如果启用）
    if (n_gpu > 0) {
        model.tok_embd->backend = GGML_BACKEND_GPU;
        model.output->backend = GGML_BACKEND_GPU;
        model.output_norm->backend = GGML_BACKEND_GPU;
    }

    // 分配 Transformer 层
    for (int i = 0; i < n_layer; i++) {
        auto & layer = model.layers[i];

        if (i < layer_gpu_count) {
            // 在 GPU 上
            layer.attn_q->backend = GGML_BACKEND_GPU;
            layer.attn_k->backend = GGML_BACKEND_GPU;
            layer.attn_v->backend = GGML_BACKEND_GPU;
            layer.attn_output->backend = GGML_BACKEND_GPU;
            layer.ffn_gate->backend = GGML_BACKEND_GPU;
            layer.ffn_up->backend = GGML_BACKEND_GPU;
            layer.ffn_down->backend = GGML_BACKEND_GPU;
        } else {
            // 在 CPU 上
            layer.attn_q->backend = GGML_BACKEND_CPU;
            // ... 其他张量
        }
    }
}
```

### 3.2 多 GPU 支持

```cpp
// 位置：src/llama-model.cpp:5123
static void llm_load_tensors_multi_gpu(
    llama_model & model,
    const llama_model_params & params) {

    const int n_gpu = params.devices.size();
    const int n_layer = model.hparams.n_layer;

    if (n_gpu <= 1) {
        // 单 GPU，使用简单策略
        llm_load_tensors_offload(model, params);
        return;
    }

    // 多 GPU 分割策略
    std::vector<int> layer_splits(n_gpu);

    if (params.split_mode == LLAMA_SPLIT_MODE_LAYER) {
        // 按层分割
        int layers_per_gpu = n_layer / n_gpu;
        for (int i = 0; i < n_gpu; i++) {
            layer_splits[i] = layers_per_gpu;
        }
        layer_splits[n_gpu - 1] += n_layer % n_gpu;  // 余数给最后一个GPU

    } else if (params.split_mode == LLAMA_SPLIT_MODE_ROW) {
        // 张量并行（按行分割矩阵）
        for (int i = 0; i < n_layer; i++) {
            auto & layer = model.layers[i];

            // 每个 GPU 负责部分列
            for (int gpu_id = 0; gpu_id < n_gpu; gpu_id++) {
                layer.attn_q_splits[gpu_id] = split_tensor_row(
                    layer.attn_q, gpu_id, n_gpu);
            }
        }
    }

    // 分配张量到不同 GPU
    int current_layer = 0;
    for (int gpu_id = 0; gpu_id < n_gpu; gpu_id++) {
        ggml_backend_t backend = params.devices[gpu_id].backend;

        for (int i = 0; i < layer_splits[gpu_id]; i++) {
            auto & layer = model.layers[current_layer++];

            // 设置后端
            ggml_backend_tensor_set_backend(layer.attn_q, backend);
            // ... 其他张量
        }
    }
}
```

## 4. 内存优化技巧

### 4.1 使用 mmap

```cpp
// 优势：
// • 延迟加载：只加载访问的页面
// • 共享内存：多进程共享同一份数据
// • 节省内存：OS 管理页面缓存

llama_model_params params = llama_model_default_params();
params.use_mmap = true;  // 启用 mmap（默认）

// 如果内存充足，可以锁定内存页（避免swap）
params.use_mlock = true;  // 需要足够的 RAM
```

### 4.2 量化感知加载

```cpp
// 根据可用内存选择量化级别
size_t available_mem = get_available_memory();
size_t model_size_f16 = 14 * 1024 * 1024 * 1024;  // 7B 模型约 14GB (FP16)

if (available_mem < model_size_f16 * 0.5) {
    // 内存不足，使用 Q4_0
    fprintf(stderr, "Low memory, using Q4_0 quantization\n");
    // 提示用户重新量化模型
} else if (available_mem < model_size_f16) {
    // 中等内存，使用 Q4_K
    fprintf(stderr, "Medium memory, Q4_K recommended\n");
} else {
    // 充足内存，可以使用 Q6_K 或 FP16
    fprintf(stderr, "Sufficient memory, using higher precision\n");
}
```

### 4.3 渐进式加载

```cpp
// 对于超大模型，分批加载
llama_model_params params = llama_model_default_params();
params.progress_callback = [](float progress, void * ctx) {
    printf("Loading: %.1f%%\r", progress * 100);
    fflush(stdout);
};

// 预加载重要层
params.n_gpu_layers = 16;  // 先加载前16层到GPU
model = llama_model_load_from_file("model.gguf", params);

// 后续可以动态调整
llama_model_offload_layers(model, 32);  // 加载更多层
```

## 5. 调试与验证

### 5.1 验证张量完整性

```cpp
void verify_model_tensors(const llama_model & model) {
    const auto & hparams = model.hparams;

    // 检查嵌入层
    assert(model.tok_embd != nullptr);
    assert(model.tok_embd->ne[0] == hparams.n_embd);
    assert(model.tok_embd->ne[1] == hparams.n_vocab);

    // 检查每一层
    for (size_t i = 0; i < model.layers.size(); i++) {
        const auto & layer = model.layers[i];

        // 检查注意力权重
        assert(layer.attn_q != nullptr);
        assert(layer.attn_k != nullptr);
        assert(layer.attn_v != nullptr);
        assert(layer.attn_output != nullptr);

        // 检查维度
        assert(layer.attn_q->ne[0] == hparams.n_embd);
        assert(layer.attn_q->ne[1] == hparams.n_embd);

        // 检查 FFN 权重
        assert(layer.ffn_gate != nullptr);
        assert(layer.ffn_up != nullptr);
        assert(layer.ffn_down != nullptr);

        printf("Layer %zu: OK\n", i);
    }

    printf("Model tensors verified successfully!\n");
}
```

### 5.2 内存使用统计

```cpp
void print_memory_usage(const llama_model & model) {
    size_t total_size = 0;
    size_t gpu_size = 0;
    size_t cpu_size = 0;

    for (const auto & [name, tensor] : model.tensors) {
        size_t size = ggml_nbytes(tensor);
        total_size += size;

        if (tensor->backend == GGML_BACKEND_GPU) {
            gpu_size += size;
        } else {
            cpu_size += size;
        }
    }

    printf("Memory Usage:\n");
    printf("  Total: %.2f GB\n", total_size / (1024.0 * 1024 * 1024));
    printf("  GPU:   %.2f GB (%.1f%%)\n",
           gpu_size / (1024.0 * 1024 * 1024),
           100.0 * gpu_size / total_size);
    printf("  CPU:   %.2f GB (%.1f%%)\n",
           cpu_size / (1024.0 * 1024 * 1024),
           100.0 * cpu_size / total_size);
}
```

## 6. 实践练习

### 练习 1：分析模型结构
编写代码统计：
- 总参数量
- 每层的参数分布
- 量化类型统计

### 练习 2：自定义加载策略
实现根据层重要性的智能 offload：
- 前几层和后几层在 GPU
- 中间层在 CPU

### 练习 3：内存预算
给定内存限制，计算最优的 n_gpu_layers 配置。

## 7. 总结

今天我们学习了模型加载的完整流程：

✅ **GGUF → llama_model**：从文件到内存结构
✅ **架构识别**：支持 120+ 种模型架构
✅ **GPU Offload**：灵活的 CPU/GPU 混合部署
✅ **内存优化**：mmap、量化、渐进式加载

### 关键要点

1. **延迟分配**：先创建骨架，后加载数据
2. **灵活 offload**：根据资源动态调整
3. **量化友好**：权重保持量化格式
4. **架构驱动**：元数据决定模型结构

## 下一步

明天我们将学习 **Day 7: 计算图构建与调度**：
- llama_graph_builder 实现
- 构建 Transformer 计算图
- 后端调度策略
- 性能优化技巧

---

📚 [Day 7: 计算图构建与调度](day07-computation-graph.md)

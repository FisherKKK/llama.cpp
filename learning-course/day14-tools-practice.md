# Day 14: 实战项目与总结

## 课程目标

综合应用所学知识：
- 从零搭建推理服务
- llama-server 源码解析
- OpenAI API 兼容实现
- 性能优化实践
- 课程总结与进阶路线

## 1. 从零搭建推理服务

### 1.1 最简单的推理程序

让我们从最基础的开始：

```cpp
// simple_inference.cpp
#include "llama.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    // 1. 初始化后端
    llama_backend_init();

    // 2. 加载模型
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 32;  // 使用 GPU 加速

    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    // 3. 创建上下文
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;         // 上下文大小
    ctx_params.n_batch = 512;        // 批大小
    ctx_params.n_threads = 8;        // CPU 线程数

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        llama_free_model(model);
        return 1;
    }

    // 4. 准备输入
    std::string prompt = "The capital of France is";
    std::cout << "Prompt: " << prompt << std::endl;

    // 5. Tokenize
    std::vector<llama_token> tokens(prompt.size() + 1);
    int n_tokens = llama_tokenize(
        model,
        prompt.c_str(),
        prompt.size(),
        tokens.data(),
        tokens.size(),
        true,   // add_bos
        false   // special tokens
    );
    tokens.resize(n_tokens);

    std::cout << "Tokens: " << n_tokens << std::endl;

    // 6. 创建采样器
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.temp = 0.7f;
    sampler_params.top_p = 0.9f;
    sampler_params.top_k = 40;

    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);

    // 7. 生成循环
    std::cout << "Output: " << prompt;

    for (int i = 0; i < 50; i++) {  // 生成 50 个 token
        // 准备 batch
        llama_batch batch = llama_batch_get_one(
            tokens.data(),
            tokens.size(),
            0,  // position
            0   // sequence id
        );

        // 前向传播
        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "Failed to decode" << std::endl;
            break;
        }

        // 获取 logits
        float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

        // 采样
        llama_token next_token = llama_sampler_sample(sampler, ctx, -1);

        // 检查结束
        if (llama_token_is_eog(model, next_token)) {
            break;
        }

        // 输出
        char piece[256];
        int n = llama_token_to_piece(model, next_token, piece, sizeof(piece), 0, false);
        if (n > 0) {
            std::cout << std::string(piece, n);
            std::cout.flush();
        }

        // 更新 tokens
        tokens.clear();
        tokens.push_back(next_token);
    }

    std::cout << std::endl;

    // 8. 清理
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
```

编译运行：
```bash
g++ -O3 -o simple_inference simple_inference.cpp \
    -I./include \
    -L./build/src -lllama \
    -lpthread -lm

./simple_inference model.gguf
```

### 1.2 添加流式输出

```cpp
// stream_inference.cpp
// ... (前面相同)

// 回调函数
void on_token_generated(const char * token_str, int len, void * user_data) {
    std::cout << std::string(token_str, len);
    std::cout.flush();
}

// 生成循环（流式版本）
llama_generate_stream(
    ctx,
    model,
    sampler,
    tokens.data(),
    tokens.size(),
    50,  // max_tokens
    on_token_generated,
    nullptr  // user_data
);
```

## 2. llama-server 源码解析

### 2.1 服务器架构

llama-server 是一个完整的 HTTP 服务器，提供 OpenAI 兼容的 API。

```
llama-server 架构：

┌─────────────────────────────────────────┐
│ HTTP 服务器 (httplib)                    │
├─────────────────────────────────────────┤
│ 路由层                                   │
│  • /v1/chat/completions                 │
│  • /v1/completions                      │
│  • /v1/embeddings                       │
│  • /health                              │
├─────────────────────────────────────────┤
│ 请求处理层                               │
│  • JSON 解析                            │
│  • 参数验证                             │
│  • 流式响应管理                         │
├─────────────────────────────────────────┤
│ 推理队列                                 │
│  • 多请求并发                           │
│  • 批处理优化                           │
│  • 连续批处理 (continuous batching)      │
├─────────────────────────────────────────┤
│ llama.cpp 核心                          │
│  • 模型加载                             │
│  • 推理执行                             │
│  • KV 缓存管理                          │
└─────────────────────────────────────────┘
```

### 2.2 关键代码片段

#### 主循环（位置：`tools/server/server.cpp:3421`）

```cpp
int main(int argc, char ** argv) {
    // 1. 解析参数
    gpt_params params = parse_server_params(argc, argv);

    // 2. 初始化服务器
    llama_server_context ctx_server;
    ctx_server.load_model(params);

    // 3. 设置路由
    httplib::Server svr;

    // POST /v1/chat/completions
    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        handle_chat_completions(ctx_server, req, res);
    });

    // POST /v1/completions
    svr.Post("/v1/completions", [&](const httplib::Request & req, httplib::Response & res) {
        handle_completions(ctx_server, req, res);
    });

    // GET /health
    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // 4. 启动服务器
    std::cout << "Server listening on http://localhost:" << params.port << std::endl;
    svr.listen("0.0.0.0", params.port);

    return 0;
}
```

#### Chat Completions 处理

```cpp
void handle_chat_completions(
    llama_server_context & ctx,
    const httplib::Request & req,
    httplib::Response & res) {

    // 1. 解析请求
    json body = json::parse(req.body);

    std::string model = body["model"];
    json messages = body["messages"];
    float temperature = body.value("temperature", 0.7f);
    float top_p = body.value("top_p", 0.9f);
    int max_tokens = body.value("max_tokens", 512);
    bool stream = body.value("stream", false);

    // 2. 构建 prompt (apply chat template)
    std::string prompt = ctx.apply_chat_template(messages);

    // 3. Tokenize
    std::vector<llama_token> tokens = ctx.tokenize(prompt, true);

    // 4. 创建任务
    server_task task;
    task.type = TASK_TYPE_COMPLETION;
    task.prompt_tokens = tokens;
    task.params.temp = temperature;
    task.params.top_p = top_p;
    task.params.n_predict = max_tokens;
    task.stream = stream;

    // 5. 提交到队列
    ctx.queue_task(task);

    // 6. 处理响应
    if (stream) {
        // 流式响应
        res.set_content_provider(
            "text/event-stream",
            [task_id = task.id, &ctx](size_t offset, httplib::DataSink & sink) {
                return stream_completion_handler(ctx, task_id, sink);
            }
        );
    } else {
        // 等待完成
        json result = ctx.wait_for_task(task.id);

        // 返回结果
        json response = {
            {"id", generate_id()},
            {"object", "chat.completion"},
            {"created", time(nullptr)},
            {"model", model},
            {"choices", json::array({
                {
                    {"index", 0},
                    {"message", {
                        {"role", "assistant"},
                        {"content", result["content"]}
                    }},
                    {"finish_reason", "stop"}
                }
            })},
            {"usage", {
                {"prompt_tokens", result["prompt_tokens"]},
                {"completion_tokens", result["completion_tokens"]},
                {"total_tokens", result["total_tokens"]}
            }}
        };

        res.set_content(response.dump(), "application/json");
    }
}
```

#### 推理队列（Continuous Batching）

```cpp
void llama_server_context::process_tasks() {
    while (running) {
        // 1. 从队列获取任务
        std::vector<server_task> batch_tasks = get_pending_tasks(batch_size);

        if (batch_tasks.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // 2. 构建批次
        llama_batch batch = llama_batch_init(batch_size, 0, 1);

        for (auto & task : batch_tasks) {
            // 添加 prompt tokens 到 batch
            for (size_t i = 0; i < task.prompt_tokens.size(); i++) {
                llama_batch_add(
                    batch,
                    task.prompt_tokens[i],
                    i,  // position
                    {task.seq_id},
                    false  // logits
                );
            }

            // 最后一个 token 需要 logits
            batch.logits[batch.n_tokens - 1] = true;
        }

        // 3. 推理
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERROR("decode failed");
            continue;
        }

        // 4. 采样
        for (auto & task : batch_tasks) {
            float * logits = llama_get_logits_ith(ctx, task.batch_index);

            llama_token next_token = llama_sampler_sample(
                task.sampler,
                ctx,
                task.batch_index
            );

            // 更新任务
            task.generated_tokens.push_back(next_token);

            // 检查结束条件
            if (llama_token_is_eog(model, next_token) ||
                task.generated_tokens.size() >= task.params.n_predict) {
                task.state = TASK_STATE_DONE;
                complete_task(task);
            } else {
                // 继续生成
                task.prompt_tokens = {next_token};
            }
        }

        llama_batch_free(batch);
    }
}
```

### 2.3 启动服务器

```bash
# 编译
cmake --build build --config Release --target llama-server

# 运行
./build/bin/llama-server \
    -m model.gguf \
    -c 4096 \
    -ngl 32 \
    --port 8080 \
    --threads 8
```

### 2.4 客户端调用

```python
# Python 客户端（OpenAI 兼容）
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy"  # llama-server 不需要 API key
)

# Chat Completion
response = client.chat.completions.create(
    model="llama-3",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)

# 流式响应
stream = client.chat.completions.create(
    model="llama-3",
    messages=[
        {"role": "user", "content": "Write a short poem about AI"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

```bash
# cURL 调用
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7
  }'
```

## 3. 性能优化实践

### 3.1 性能分析

```cpp
// 添加性能计时
#include <chrono>

struct perf_timer {
    std::chrono::high_resolution_clock::time_point start;

    perf_timer() : start(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// 使用
perf_timer timer;
llama_decode(ctx, batch);
printf("Decode: %.2f ms\n", timer.elapsed_ms());
```

### 3.2 批处理优化

```cpp
// 批量处理多个请求
llama_batch batch = llama_batch_init(512, 0, 4);  // 支持4个序列

// 序列 0: "The capital of France"
llama_batch_add(batch, token_ids[0], 0, {0}, false);
llama_batch_add(batch, token_ids[1], 1, {0}, false);
// ...

// 序列 1: "What is 2+2?"
llama_batch_add(batch, token_ids[10], 0, {1}, false);
llama_batch_add(batch, token_ids[11], 1, {1}, false);
// ...

// 一次推理，两个序列同时生成
llama_decode(ctx, batch);
```

### 3.3 KV 缓存复用

```cpp
// 复用公共 prompt
// 例如：system prompt 对所有请求相同

// 1. 首次计算 system prompt，保存 KV 缓存
std::vector<llama_token> system_tokens = tokenize("You are a helpful assistant.");
llama_decode_with_cache_save(ctx, system_tokens, "system_cache.bin");

// 2. 后续请求直接加载
llama_decode_with_cache_load(ctx, "system_cache.bin");

// 3. 只处理用户输入（节省计算）
std::vector<llama_token> user_tokens = tokenize("What is AI?");
llama_decode(ctx, user_tokens);
```

## 4. 课程总结

### 4.1 你已经掌握的核心技能

经过 14 天的学习，你现在能够：

#### 基础理解（Day 1-4）
- ✅ 理解 llama.cpp 的分层架构
- ✅ 掌握 GGML 张量库的核心概念
- ✅ 理解计算图的构建与执行
- ✅ 掌握张量操作的实现细节

#### 模型加载（Day 5-6）
- ✅ 解析 GGUF 文件格式
- ✅ 理解模型加载的完整流程
- ✅ 实现自定义的架构支持
- ✅ 优化内存使用和 GPU offload

#### 推理核心（Day 7-10）
- ✅ 构建 Transformer 计算图
- ✅ 理解注意力机制实现
- ✅ 掌握 KV 缓存优化
- ✅ 实现 RoPE 位置编码

#### 高性能技术（Day 11-12）
- ✅ CPU SIMD 优化
- ✅ GPU 加速（CUDA/Metal）
- ✅ 后端抽象层设计
- ✅ 多 GPU 并行策略

#### 应用开发（Day 13-14）
- ✅ 采样策略的实现
- ✅ 搭建推理服务
- ✅ OpenAI API 兼容
- ✅ 性能调优实践

### 4.2 关键技术点回顾

| 技术 | 核心文件 | 重要性 | 难度 |
|------|---------|-------|------|
| **GGML 张量** | ggml/src/ggml.c | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **GGUF 格式** | ggml/src/gguf.c | ⭐⭐⭐⭐ | ⭐⭐ |
| **模型加载** | src/llama-model.cpp | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **计算图构建** | src/llama-graph.cpp | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **KV 缓存** | src/llama-kv-cache.cpp | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **量化系统** | ggml/src/ggml-quants.c | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **采样策略** | src/llama-sampling.cpp | ⭐⭐⭐⭐ | ⭐⭐ |
| **后端实现** | ggml/src/ggml-cuda/ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 5. 进阶路线

### 5.1 深入方向

#### 方向 1：架构扩展
- 添加新模型架构支持（如 Llama 4, Qwen 3）
- 实现自定义层类型
- 优化特定架构的推理

**学习资源**：
- `src/llama-arch.cpp` - 架构注册
- `docs/development/HOWTO-add-model.md`

#### 方向 2：性能优化
- 实现自定义 CUDA 内核
- 优化量化算法
- 改进 KV 缓存策略

**学习资源**：
- `ggml/src/ggml-cuda/` - CUDA 实现
- CUDA Programming Guide
- FlashAttention 论文

#### 方向 3：量化研究
- 实现新的量化方法
- 优化量化误差
- 量化感知训练

**学习资源**：
- `ggml/src/ggml-quants.c`
- LLM.int8() 论文
- GPTQ, AWQ 论文

#### 方向 4：应用开发
- 构建生产级服务
- 实现 RAG 系统
- 多模态支持

**学习资源**：
- `tools/server/` - 服务器实现
- LangChain, LlamaIndex

### 5.2 推荐项目

#### 初级项目
1. **自定义采样器**：实现一个新的采样策略
2. **性能监控工具**：可视化推理性能
3. **模型转换工具**：支持新格式转换

#### 中级项目
1. **分布式推理**：多机多卡推理系统
2. **量化工具链**：自动化量化流程
3. **RAG 系统**：集成向量数据库

#### 高级项目
1. **自定义后端**：支持新硬件（如 NPU）
2. **推测解码**：实现 Speculative Decoding
3. **混合精度推理**：动态精度调整

### 5.3 持续学习资源

#### 论文阅读
- **Transformer**: "Attention Is All You Need"
- **LLaMA**: "LLaMA: Open and Efficient Foundation Language Models"
- **FlashAttention**: "FlashAttention: Fast and Memory-Efficient Exact Attention"
- **GPTQ**: "GPTQ: Accurate Post-Training Quantization for GPTs"
- **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"

#### 开源项目
- **GGML**: https://github.com/ggerganov/ggml
- **vLLM**: 高性能推理服务器
- **TensorRT-LLM**: NVIDIA 的优化库
- **Text Generation Inference**: Hugging Face 的推理引擎

#### 社区资源
- llama.cpp GitHub Discussions
- r/LocalLLaMA 社区
- GGML Discord 服务器

## 6. 最后的话

恭喜你完成了这个 14 天的深度学习课程！🎉

你已经从一个 llama.cpp 的初学者成长为能够：
- 阅读和理解 50K+ 行核心代码
- 修改和扩展 llama.cpp 功能
- 优化推理性能
- 构建生产级应用

### 继续前进

记住，真正的掌握来自实践：

1. **多动手**：尝试修改代码，观察效果
2. **多阅读**：阅读最新论文和代码
3. **多交流**：参与社区讨论，分享经验
4. **多思考**：深入理解背后的原理

### 保持联系

如果你在学习过程中遇到问题或有任何改进建议，欢迎：
- 在 GitHub 提 Issue
- 参与社区讨论
- 分享你的项目

祝你在 LLM 推理领域取得成功！🚀

---

**最终练习**：
1. 部署一个 llama-server 实例
2. 实现一个自定义的采样策略
3. 测试并优化推理性能
4. 分享你的学习心得

📚 **课程完成！** 回顾：[README](README.md)

---

*感谢你的坚持与努力！希望这个课程对你有所帮助。*

# llama.cpp 常见问题解答（FAQ）

## 目录

1. [入门问题](#1-入门问题)
2. [编译与安装](#2-编译与安装)
3. [模型相关](#3-模型相关)
4. [性能优化](#4-性能优化)
5. [GPU 加速](#5-gpu-加速)
6. [内存问题](#6-内存问题)
7. [推理问题](#7-推理问题)
8. [高级话题](#8-高级话题)
9. [故障排查](#9-故障排查)
10. [最佳实践](#10-最佳实践)

---

## 1. 入门问题

### Q: llama.cpp 是什么？
**A**: llama.cpp 是一个用 C/C++ 编写的 LLM 推理引擎，特点是：
- 零依赖，纯 C/C++ 实现
- 支持 CPU 和 GPU 推理
- 高性能量化（4-bit, 8-bit 等）
- 支持 120+ 种模型架构
- 跨平台（Linux/Windows/macOS/iOS/Android）

### Q: llama.cpp 与 PyTorch/TensorFlow 有什么区别？
**A**:
| 特性 | llama.cpp | PyTorch/TF |
|------|-----------|-----------|
| **用途** | 推理 | 训练 + 推理 |
| **依赖** | 无 | 大量依赖 |
| **内存** | 低（量化） | 高（FP32） |
| **速度** | 快 | 中等 |
| **易用性** | C++ API | Python API |

**总结**：llama.cpp 专注于高效推理，PyTorch/TF 更适合训练。

### Q: 我应该使用哪个量化级别？
**A**: 推荐按需选择：

| 场景 | 推荐量化 | 理由 |
|------|---------|------|
| **生产环境** | Q4_K_M | 最佳平衡（4GB, 质量损失<7%） |
| **高质量** | Q6_K 或 Q8_0 | 接近原始质量 |
| **资源受限** | Q3_K_M | 3GB，质量可接受 |
| **实验测试** | Q2_K | 2GB，质量下降明显 |
| **开发调试** | F16 | 最高质量，但大 |

### Q: 如何快速开始？
**A**: 三步开始：
```bash
# 1. 编译
cmake -B build && cmake --build build --config Release

# 2. 下载模型
# 从 HuggingFace: huggingface.co/models?library=gguf
# 或使用 convert_hf_to_gguf.py 转换

# 3. 运行
./build/bin/llama-cli -m model.gguf -p "Hello" -n 50
```

---

## 2. 编译与安装

### Q: 编译失败，找不到 CUDA？
**A**: 确保安装了 CUDA Toolkit 并设置路径：
```bash
# 检查 CUDA
nvcc --version

# 编译时指定路径
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Q: macOS 上如何启用 Metal？
**A**: Metal 默认启用，无需额外配置：
```bash
# 自动检测 Metal
cmake -B build
cmake --build build --config Release

# 验证
./build/bin/llama-cli -m model.gguf -ngl 1 2>&1 | grep -i metal
```

### Q: ARM 平台（树莓派、手机）如何编译？
**A**:
```bash
# ARM64
cmake -B build -DGGML_CPU=ON
cmake --build build

# 启用 NEON 优化（自动检测）
# 在 Android 上，使用 NDK：
export ANDROID_NDK=/path/to/ndk
cmake -B build -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake
```

### Q: 如何启用 AVX512？
**A**:
```bash
# 编译时自动检测，或强制启用
cmake -B build -DGGML_AVX512=ON
```

### Q: Python bindings 如何使用？
**A**: llama.cpp 有官方 Python 包：
```bash
pip install llama-cpp-python

# 使用
from llama_cpp import Llama
llm = Llama(model_path="model.gguf", n_gpu_layers=32)
output = llm("Hello", max_tokens=50)
```

---

## 3. 模型相关

### Q: 如何转换 HuggingFace 模型？
**A**:
```bash
python convert_hf_to_gguf.py \
  --outfile model-f16.gguf \
  --outtype f16 \
  /path/to/huggingface/model

# 然后量化
./build/bin/llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

### Q: 支持哪些模型？
**A**: 120+ 种架构，包括：
- **LLaMA 系列**: LLaMA 1/2/3, LLaMA 3.1/3.2/4
- **Mistral 系列**: Mistral 7B, Mixtral 8x7B/8x22B
- **Qwen 系列**: Qwen 1.5/2/2.5/3
- **Gemma**: Gemma 2B/7B/9B/27B
- **Phi**: Phi 2/3/3.5
- **其他**: GPT-2, BERT, T5, Mamba, Jamba...

完整列表见：`src/llama-arch.cpp`

### Q: 如何查看模型信息？
**A**:
```bash
# 使用 llama-cli
./build/bin/llama-cli -m model.gguf --verbose-prompt 2>&1 | head -50

# 或使用 Python
python gguf-py/scripts/gguf_dump.py model.gguf
```

### Q: GGUF 和 GGML 有什么区别？
**A**:
- **GGML**（旧）：简单格式，缺少元数据
- **GGUF**（新）：统一格式，自描述，可扩展
- **建议**：使用 GGUF，GGML 已弃用

### Q: 如何修改模型元数据？
**A**:
```bash
python gguf-py/scripts/gguf_set_metadata.py \
  model.gguf \
  --kv "general.name=My Custom Model" \
  --kv "general.description=Description here"
```

---

## 4. 性能优化

### Q: 推理速度慢怎么办？
**A**: 按优先级检查：

1. **启用 GPU 加速**
   ```bash
   ./llama-cli -m model.gguf -ngl -1  # -1 = 全部层
   ```

2. **使用量化模型**
   ```bash
   # Q4_K 比 F16 快 4x+
   ./llama-quantize model-f16.gguf model-q4.gguf Q4_K_M
   ```

3. **增加批大小**
   ```bash
   ./llama-cli -m model.gguf -b 512  # 默认 512
   ```

4. **调整线程数**
   ```bash
   # 设置为 CPU 核心数
   ./llama-cli -m model.gguf -t 8
   ```

5. **启用 Flash Attention**（如果 GPU 支持）
   ```cpp
   ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
   ```

### Q: 如何测试性能？
**A**:
```bash
./build/bin/llama-bench \
  -m model.gguf \
  -p 512 \    # 提示长度
  -n 128 \    # 生成长度
  -ngl 32 \   # GPU 层数
  -r 5        # 重复次数
```

### Q: CPU 和 GPU 性能差距有多大？
**A**: 典型对比（7B 模型）：

| 硬件 | 速度 (tokens/s) | 倍数 |
|------|----------------|------|
| i9-13900K (CPU) | ~5 t/s | 1x |
| RTX 3060 (GPU) | ~40 t/s | 8x |
| RTX 4090 (GPU) | ~120 t/s | 24x |
| M2 Max (Metal) | ~40 t/s | 8x |

**结论**：GPU 加速提升 8-25x。

### Q: 多 GPU 如何配置？
**A**:
```bash
# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 分层模式（推荐）
./llama-cli -m model.gguf -ngl -1 -sm layer

# 张量并行模式
./llama-cli -m model.gguf -ngl -1 -sm row
```

---

## 5. GPU 加速

### Q: 如何确认 GPU 正在使用？
**A**:
```bash
# 方法 1: 查看日志
./llama-cli -m model.gguf -ngl 1 -p "test" 2>&1 | grep -i cuda

# 方法 2: 监控 GPU
watch -n 1 nvidia-smi

# 方法 3: 检查内存
nvidia-smi --query-gpu=memory.used --format=csv
```

### Q: CUDA OOM（内存不足）怎么办？
**A**: 减少内存使用：

1. **减少 GPU 层数**
   ```bash
   ./llama-cli -m model.gguf -ngl 24  # 而不是 -1
   ```

2. **减少上下文大小**
   ```bash
   ./llama-cli -m model.gguf -c 2048  # 而不是 4096
   ```

3. **使用更激进的量化**
   ```bash
   # Q4_0 比 Q6_K 省 30% 内存
   ```

4. **使用 FP16 KV 缓存**（默认）
   ```cpp
   ctx_params.type_k = GGML_TYPE_F16;
   ctx_params.type_v = GGML_TYPE_F16;
   ```

### Q: Metal 性能不如预期？
**A**: 检查：
1. 确保使用 Apple Silicon（M1/M2/M3）
2. 更新到最新 macOS
3. 检查 Xcode Command Line Tools
4. 使用 `-ngl -1` 全部 offload

### Q: 如何在 Jetson/嵌入式设备上运行？
**A**:
```bash
# Jetson (ARM + CUDA)
cmake -B build -DGGML_CUDA=ON
cmake --build build

# 使用低内存配置
./llama-cli -m model-q4.gguf -c 1024 -b 256 -ngl 16
```

---

## 6. 内存问题

### Q: 模型需要多少内存？
**A**: 参考表（7B 模型）：

| 量化 | 模型大小 | +KV缓存(ctx=2048) | +KV缓存(ctx=4096) |
|------|---------|------------------|------------------|
| F16 | 14GB | +1GB | +2GB |
| Q8_0 | 8GB | +500MB | +1GB |
| Q6_K | 6GB | +500MB | +1GB |
| Q4_K | 4GB | +500MB | +1GB |
| Q4_0 | 4GB | +500MB | +1GB |

**公式**：总内存 ≈ 模型大小 + KV缓存 + 1GB（系统开销）

### Q: 内存不足怎么办？
**A**: 优先级排序：

1. **使用量化**：Q4_K 减少 75% 内存
2. **减少上下文**：`-c 1024` 而不是 4096
3. **部分 offload**：`-ngl 16` 而不是全部
4. **使用 mmap**（默认启用）：共享内存
5. **关闭 mlock**（默认关闭）：允许 swap

### Q: mmap 是什么？应该使用吗？
**A**:
- **mmap**: 将文件直接映射到内存，零拷贝加载
- **优势**: 快速启动、节省内存、多进程共享
- **劣势**: 依赖操作系统页面缓存
- **建议**: 默认启用（`use_mmap = true`）

### Q: 什么时候使用 mlock？
**A**:
- **mlock**: 锁定内存页，防止 swap 到磁盘
- **使用场景**: 对延迟极度敏感的应用
- **前提**: 有足够的 RAM（> 模型大小 × 2）
- **注意**: 可能需要 root 权限或调整 ulimit
- **建议**: 一般不需要

---

## 7. 推理问题

### Q: 生成的文本质量差/重复？
**A**: 调整采样参数：

```bash
# 重复严重 → 增加重复惩罚
--repeat-penalty 1.2  # 默认 1.1

# 输出混乱 → 降低温度
--temp 0.5  # 默认 0.8

# 太保守 → 增加温度/top-p
--temp 1.0 --top-p 0.95

# 尝试 Mirostat
--mirostat 2 --mirostat-tau 5.0
```

### Q: 如何强制模型输出 JSON？
**A**: 使用语法约束：
```bash
./llama-cli -m model.gguf \
  -p "Generate JSON" \
  --grammar-file grammars/json.gbnf
```

或在代码中：
```cpp
llama_sampler * grammar_sampler = llama_sampler_init_grammar(
    model, json_grammar_str, "root");
llama_sampler_chain_add(chain, grammar_sampler);
```

### Q: 如何使用 Chat 模板？
**A**: llama.cpp 自动应用模板：
```cpp
// 自动从模型元数据读取 chat template
// 大多数模型已内置

// 或手动应用
std::string apply_chat_template(
    const std::vector<Message> & messages) {
    // Llama 3 格式示例
    std::string prompt;
    for (const auto & msg : messages) {
        prompt += "<|start_header_id|>" + msg.role + "<|end_header_id|>\n";
        prompt += msg.content + "<|eot_id|>";
    }
    return prompt;
}
```

### Q: 推理结果不可复现？
**A**: 设置随机种子：
```bash
./llama-cli -m model.gguf -s 42  # 固定种子

# 或在代码中
sampler_params.seed = 42;
```

### Q: 如何实现批量推理？
**A**: 使用 `llama_batch` 和多序列：
```cpp
llama_batch batch = llama_batch_init(512, 0, 4);  // 4 个序列

// 添加多个序列的 token
llama_batch_add(batch, token1, pos, {0}, false);  // 序列 0
llama_batch_add(batch, token2, pos, {1}, false);  // 序列 1
// ...

llama_decode(ctx, batch);
```

---

## 8. 高级话题

### Q: 如何添加新的模型架构？
**A**: 参考文档：
```bash
# 1. 阅读指南
cat docs/development/HOWTO-add-model.md

# 2. 在 llama-arch.cpp 中注册架构
# 3. 实现 llm_load_<arch>() 和 llm_build_<arch>()
# 4. 测试并提交 PR
```

### Q: 如何实现自定义采样器？
**A**:
```cpp
struct my_sampler_context {
    // 自定义状态
};

static void my_sampler_apply(
    struct llama_sampler * sampler,
    llama_token_data_array * cur_p) {

    // 自定义采样逻辑
    for (size_t i = 0; i < cur_p->size; i++) {
        cur_p->data[i].logit *= my_factor;
    }
}

static struct llama_sampler_i my_sampler_i = {
    /* .apply = */ my_sampler_apply,
    // ... 其他回调
};

llama_sampler * llama_sampler_init_my_sampler() {
    return new llama_sampler {
        /* .iface = */ &my_sampler_i,
        /* .ctx = */ new my_sampler_context(),
    };
}
```

### Q: 如何实现流式服务器？
**A**: 参考 `tools/server/server.cpp`：
```cpp
// 使用 SSE (Server-Sent Events)
res.set_content_provider(
    "text/event-stream",
    [](size_t offset, httplib::DataSink & sink) {
        // 每生成一个 token
        std::string event = "data: " + json_chunk + "\n\n";
        sink.write(event.c_str(), event.size());
        return true;  // 继续
    });
```

### Q: 如何实现 RAG（检索增强生成）？
**A**:
```cpp
// 1. 使用 embeddings 模型获取文档嵌入
llama_model * embed_model = llama_model_load_from_file("embed-model.gguf", params);

// 2. 检索相关文档
std::vector<Document> relevant_docs = retrieve(query, doc_embeddings, top_k=5);

// 3. 构建增强提示
std::string prompt = "Context:\n";
for (const auto & doc : relevant_docs) {
    prompt += doc.content + "\n\n";
}
prompt += "Question: " + query + "\nAnswer:";

// 4. 生成
generate(chat_model, prompt);
```

---

## 9. 故障排查

### Q: 编译时出现 "undefined reference" 错误？
**A**:
```bash
# 清理并重新构建
rm -rf build
cmake -B build
cmake --build build --config Release -j
```

### Q: 运行时 Segmentation Fault？
**A**: 检查：
1. 模型文件完整性：`md5sum model.gguf`
2. 内存是否足够
3. 指针是否有效：`if (model == NULL) ...`
4. 使用调试器：`gdb ./llama-cli`

### Q: 输出乱码？
**A**:
1. 检查 tokenizer 是否正确
2. 确认模型支持目标语言
3. 检查终端编码：`export LANG=en_US.UTF-8`

### Q: Windows 上编译失败？
**A**:
```powershell
# 使用 Visual Studio 2019+
cmake -B build -G "Visual Studio 16 2019" -A x64
cmake --build build --config Release

# 或使用 MSYS2/MinGW
```

### Q: 日志太多/太少？
**A**:
```bash
# 设置日志级别
export LLAMA_LOG_LEVEL=error    # error/warn/info/debug
export LLAMA_LOG_FILE=llama.log  # 输出到文件
```

---

## 10. 最佳实践

### Q: 生产环境部署建议？
**A**:

**硬件选择**：
- **CPU**: 至少 8 核，支持 AVX2
- **RAM**: 模型大小 × 2（例如 7B Q4_K = 8GB RAM）
- **GPU**: NVIDIA（CUDA）或 Apple Silicon（Metal）
- **存储**: SSD（快速加载）

**软件配置**：
```cpp
// 稳定性优先
model_params.use_mmap = true;
model_params.use_mlock = false;
ctx_params.n_ctx = 2048;
ctx_params.n_batch = 512;

// 性能优先
model_params.n_gpu_layers = -1;
ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
ctx_params.offload_kqv = true;
```

**监控**：
- GPU 使用率：`nvidia-smi`
- CPU/内存：`htop`
- 推理延迟：记录 tokens/s
- 错误率：日志分析

### Q: 如何选择模型大小？
**A**:

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **嵌入式设备** | 1B-3B | 低资源消耗 |
| **个人电脑** | 7B-13B | 平衡质量和速度 |
| **服务器** | 13B-70B | 高质量输出 |
| **专业应用** | 70B+ | 最佳性能 |

### Q: 安全注意事项？
**A**:
1. **输入验证**: 限制提示长度
2. **输出过滤**: 检测有害内容
3. **速率限制**: 防止滥用
4. **资源限制**: 设置 timeout
5. **沙箱运行**: 隔离进程

### Q: 如何优化首次生成延迟（TTFT）？
**A**:
1. **使用 mmap**: 快速加载
2. **增加批大小**: 并行处理提示
3. **缓存提示**: 重用 system prompt
4. **预热模型**: 启动时运行一次推理

### Q: 如何提高吞吐量？
**A**:
1. **连续批处理**: llama-server 的 `-cb` 选项
2. **增加并发槽位**: `--slots N`
3. **使用量化**: Q4_K 减少内存带宽
4. **GPU 加速**: 全层 offload

---

## 11. 社区与支持

### Q: 如何获取帮助？
**A**:
1. **文档**: 查看 `docs/` 目录
2. **GitHub Issues**: 搜索已有问题
3. **Discussions**: 提问讨论
4. **Discord**: 实时交流
5. **Reddit**: r/LocalLLaMA 社区

### Q: 如何报告 Bug？
**A**:
```markdown
**环境**:
- OS: Ubuntu 22.04
- GPU: RTX 4090
- CUDA: 12.2
- llama.cpp commit: abc123

**复现步骤**:
1. ...
2. ...

**期望行为**: ...
**实际行为**: ...
**日志**: ...
```

### Q: 如何贡献代码？
**A**:
1. 阅读 `CONTRIBUTING.md` 和 `AGENTS.md`
2. Fork 仓库
3. 创建分支
4. 提交 PR（小改动优先）
5. 响应 review

**注意**: 不接受 AI 生成的 PR（见 AGENTS.md）

---

## 12. 性能参考

### 典型硬件性能（7B Q4_K 模型）

| 硬件 | 提示处理 | 生成速度 | 备注 |
|------|---------|---------|------|
| i9-13900K | 50 t/s | 5 t/s | CPU only |
| RTX 3060 12GB | 300 t/s | 40 t/s | CUDA |
| RTX 4090 24GB | 1000 t/s | 120 t/s | CUDA |
| M1 Max 64GB | 200 t/s | 35 t/s | Metal |
| M2 Max 96GB | 250 t/s | 40 t/s | Metal |
| 4× RTX 4090 | 3000 t/s | 400 t/s | 多 GPU |

t/s = tokens per second

---

## 快速诊断流程图

```
推理问题？
    ↓
速度慢？
    ├─ YES → 启用 GPU（-ngl -1）
    ├─ YES → 使用量化（Q4_K）
    └─ YES → 增加批大小（-b 512）
    ↓
质量差？
    ├─ YES → 检查量化级别（升级到 Q6_K）
    ├─ YES → 调整温度（--temp 0.7）
    └─ YES → 增加 top-p（--top-p 0.9）
    ↓
内存不足？
    ├─ YES → 减少上下文（-c 2048）
    ├─ YES → 部分 offload（-ngl 16）
    └─ YES → 使用更小量化（Q4_0）
    ↓
仍有问题？
    └─ 查看日志、提 Issue
```

---

**持续更新中**。如果你有其他问题，欢迎在 GitHub Discussions 提出！

**相关资源**：
- [快速参考手册](quick-reference.md)
- [课程总览](README.md)
- [官方文档](https://github.com/ggml-org/llama.cpp)

# llama.cpp 快速参考手册

## 概览

本手册提供 llama.cpp 的常用 API、命令、参数的快速查询。适合作为日常开发的参考工具。

---

## 1. 核心 API 速查

### 1.1 模型加载

```cpp
// 基础加载
llama_model_params model_params = llama_model_default_params();
model_params.n_gpu_layers = 32;      // GPU 层数（-1 = 全部）
model_params.use_mmap = true;        // 使用 mmap
model_params.use_mlock = false;      // 锁定内存

llama_model * model = llama_model_load_from_file("model.gguf", model_params);
```

**常用参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `n_gpu_layers` | int | 0 | GPU 层数，-1=全部 |
| `split_mode` | enum | NONE | 多 GPU 分割模式 |
| `main_gpu` | int | 0 | 主 GPU ID |
| `use_mmap` | bool | true | 使用 mmap 加载 |
| `use_mlock` | bool | false | 锁定内存防止 swap |

### 1.2 上下文创建

```cpp
llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx = 2048;             // 上下文大小
ctx_params.n_batch = 512;            // 逻辑批大小
ctx_params.n_ubatch = 512;           // 物理批大小
ctx_params.n_threads = 8;            // CPU 线程数
ctx_params.rope_freq_base = 10000.0; // RoPE 基频
ctx_params.rope_freq_scale = 1.0;    // RoPE 缩放
ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

llama_context * ctx = llama_new_context_with_model(model, ctx_params);
```

**常用参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `n_ctx` | uint32 | 512 | 上下文窗口大小 |
| `n_batch` | uint32 | 512 | 提示处理批大小 |
| `n_ubatch` | uint32 | 512 | GPU 批大小 |
| `n_threads` | int32 | -1 | CPU 线程（-1=自动） |
| `n_threads_batch` | int32 | -1 | 批处理线程 |
| `rope_scaling_type` | enum | UNSPECIFIED | RoPE 缩放类型 |
| `offload_kqv` | bool | true | KQV 卸载到 GPU |
| `flash_attn_type` | enum | AUTO | Flash Attention |

### 1.3 Tokenization

```cpp
// 编码（文本 → token IDs）
std::vector<llama_token> tokens(text.size() + 1);
int n_tokens = llama_tokenize(
    model,
    text.c_str(),
    text.size(),
    tokens.data(),
    tokens.size(),
    true,   // add_bos：添加开始符
    false   // special：允许特殊 token
);
tokens.resize(n_tokens);

// 解码（token ID → 文本）
char piece[256];
int n = llama_token_to_piece(
    model,
    token_id,
    piece,
    sizeof(piece),
    0,      // lstrip：移除前导空格
    false   // special：特殊 token 处理
);
std::string token_str(piece, n);
```

### 1.4 推理（Decode）

```cpp
// 创建 batch
llama_batch batch = llama_batch_init(n_tokens, 0, 1);

// 添加 token
for (int i = 0; i < n_tokens; i++) {
    llama_batch_add(
        batch,
        tokens[i],     // token ID
        i,             // position
        {0},           // sequence IDs
        i == n_tokens - 1  // logits：只有最后一个输出
    );
}

// 执行推理
int ret = llama_decode(ctx, batch);

// 获取 logits
float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

// 清理
llama_batch_free(batch);
```

### 1.5 采样

```cpp
// 创建采样器链
llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
sampler_params.temp = 0.7f;
sampler_params.top_p = 0.9f;
sampler_params.top_k = 40;
sampler_params.penalty_repeat = 1.1f;

llama_sampler * sampler = llama_sampler_chain_init(sampler_params);

// 采样
llama_token next_token = llama_sampler_sample(sampler, ctx, -1);

// 清理
llama_sampler_free(sampler);
```

### 1.6 清理资源

```cpp
llama_sampler_free(sampler);
llama_free(ctx);
llama_free_model(model);
llama_backend_free();
```

---

## 2. 命令行工具速查

### 2.1 llama-cli

```bash
# 基础推理
llama-cli -m model.gguf -p "Hello"

# 常用参数
llama-cli \
  -m model.gguf \              # 模型路径
  -p "Your prompt" \           # 提示文本
  -n 128 \                     # 生成 token 数
  -c 4096 \                    # 上下文大小
  -ngl 32 \                    # GPU 层数
  -t 8 \                       # CPU 线程数
  --temp 0.7 \                 # 温度
  --top-p 0.9 \                # Top-P
  --top-k 40 \                 # Top-K
  --repeat-penalty 1.1 \       # 重复惩罚
  -b 512 \                     # 批大小
  --color                      # 彩色输出
```

**参数速查**：
```
模型与硬件：
  -m, --model PATH           模型文件路径
  -ngl, --n-gpu-layers N     GPU 层数（-1=全部）
  -sm, --split-mode MODE     多 GPU 模式（none/layer/row）
  -ts, --tensor-split SPLIT  张量分割比例

上下文与批处理：
  -c, --ctx-size N           上下文大小（默认 512）
  -b, --batch-size N         批大小（默认 512）
  -ub, --ubatch-size N       物理批大小

生成控制：
  -n, --n-predict N          生成 token 数（默认 -1=无限）
  -p, --prompt TEXT          提示文本
  -f, --file PATH            从文件读取提示
  --temp N                   温度（默认 0.8）
  --top-p N                  Top-P（默认 0.95）
  --top-k N                  Top-K（默认 40）
  --repeat-penalty N         重复惩罚（默认 1.1）

性能：
  -t, --threads N            CPU 线程数
  -tb, --threads-batch N     批处理线程数
  --mlock                    锁定内存
  --no-mmap                  不使用 mmap

输出：
  --color                    彩色输出
  --verbose-prompt           显示处理后的提示
  -s, --seed N               随机种子
```

### 2.2 llama-server

```bash
# 启动服务器
llama-server \
  -m model.gguf \
  -c 4096 \
  -ngl 32 \
  --host 0.0.0.0 \
  --port 8080 \
  -t 8

# 常用参数
--host HOST                  监听地址（默认 127.0.0.1）
--port PORT                  端口（默认 8080）
--path PATH                  HTTP 路径前缀
-to, --timeout N             服务器超时（秒）
--threads-http N             HTTP 线程数
--metrics                    启用 /metrics 端点
--log-format FORMAT          日志格式（text/json）
--slots N                    并发槽位数（默认 1）
-cb, --cont-batching         连续批处理
```

**API 端点**：
```
POST /v1/chat/completions    Chat 补全（OpenAI 兼容）
POST /v1/completions         文本补全
POST /v1/embeddings          获取嵌入
GET  /v1/models              列出模型
GET  /health                 健康检查
GET  /metrics                Prometheus 指标
GET  /props                  服务器属性
```

### 2.3 llama-quantize

```bash
# 量化模型
llama-quantize \
  model-f16.gguf \           # 输入（FP16）
  model-q4_k_m.gguf \        # 输出（量化）
  Q4_K_M                     # 量化类型

# 量化类型
Q4_0, Q4_1                   # 古典 4-bit
Q5_0, Q5_1                   # 古典 5-bit
Q8_0                         # 8-bit
Q2_K, Q3_K_S, Q3_K_M, Q3_K_L # K-量化 2/3-bit
Q4_K_S, Q4_K_M               # K-量化 4-bit（推荐）
Q5_K_S, Q5_K_M               # K-量化 5-bit
Q6_K                         # K-量化 6-bit
IQ2_XXS, IQ2_XS              # 超小量化
```

### 2.4 llama-bench

```bash
# 性能基准测试
llama-bench \
  -m model.gguf \
  -p 512 \                   # 提示长度
  -n 128 \                   # 生成长度
  -ngl 32 \                  # GPU 层数
  -t 8                       # 线程数

# 输出示例
model_filename          : model.gguf
model_size              : 7.16 GiB
model_n_params          : 6738.42 M
n_batch                 : 512
n_threads               : 8
n_gpu_layers            : 32

pp 512                  : 234.56 ± 2.34 ms (2183 t/s)
tg 128                  : 5.67 ± 0.12 ms (177 t/s)

pp = prompt processing (提示处理)
tg = token generation (生成)
t/s = tokens per second
```

### 2.5 Python 脚本

```bash
# 转换 HuggingFace 模型
python convert_hf_to_gguf.py \
  --outfile model-f16.gguf \
  --outtype f16 \
  /path/to/huggingface/model

# 查看 GGUF 信息
python gguf-py/scripts/gguf_dump.py model.gguf

# 修改元数据
python gguf-py/scripts/gguf_set_metadata.py \
  model.gguf \
  --kv "general.name=My Model"
```

---

## 3. 常用代码模板

### 3.1 完整推理示例

```cpp
#include "llama.h"
#include <vector>
#include <string>

int main() {
    // 1. 初始化
    llama_backend_init();

    // 2. 加载模型
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 32;
    auto * model = llama_model_load_from_file("model.gguf", model_params);

    // 3. 创建上下文
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    auto * ctx = llama_new_context_with_model(model, ctx_params);

    // 4. Tokenize
    std::string prompt = "Hello, world!";
    std::vector<llama_token> tokens(prompt.size() + 1);
    int n_tokens = llama_tokenize(model, prompt.c_str(), prompt.size(),
                                   tokens.data(), tokens.size(), true, false);
    tokens.resize(n_tokens);

    // 5. 创建采样器
    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.temp = 0.7f;
    auto * sampler = llama_sampler_chain_init(sampler_params);

    // 6. 生成循环
    for (int i = 0; i < 50; i++) {
        // Decode
        auto batch = llama_batch_get_one(tokens.data(), tokens.size(), 0, 0);
        llama_decode(ctx, batch);

        // Sample
        auto next = llama_sampler_sample(sampler, ctx, -1);
        if (llama_token_is_eog(model, next)) break;

        // Detokenize
        char piece[256];
        int n = llama_token_to_piece(model, next, piece, sizeof(piece), 0, false);
        printf("%.*s", n, piece);
        fflush(stdout);

        // Update
        tokens.clear();
        tokens.push_back(next);
    }
    printf("\n");

    // 7. 清理
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
```

### 3.2 流式生成回调

```cpp
// 回调函数类型
typedef void (*token_callback_t)(const char * piece, int len, void * user_data);

// 生成函数
void generate_with_callback(
    llama_context * ctx,
    llama_model * model,
    llama_sampler * sampler,
    const std::vector<llama_token> & prompt_tokens,
    int max_tokens,
    token_callback_t callback,
    void * user_data) {

    std::vector<llama_token> tokens = prompt_tokens;

    for (int i = 0; i < max_tokens; i++) {
        auto batch = llama_batch_get_one(tokens.data(), tokens.size(), 0, 0);
        llama_decode(ctx, batch);

        auto next = llama_sampler_sample(sampler, ctx, -1);
        if (llama_token_is_eog(model, next)) break;

        char piece[256];
        int n = llama_token_to_piece(model, next, piece, sizeof(piece), 0, false);

        // 调用回调
        if (callback) {
            callback(piece, n, user_data);
        }

        tokens.clear();
        tokens.push_back(next);
    }
}

// 使用
void my_callback(const char * piece, int len, void * user_data) {
    std::cout << std::string(piece, len);
    std::cout.flush();
}

generate_with_callback(ctx, model, sampler, prompt_tokens, 100, my_callback, nullptr);
```

---

## 4. 性能参数速查表

### 4.1 内存使用估算

| 模型大小 | FP16 | Q8_0 | Q6_K | Q4_K | Q4_0 | IQ2_XXS |
|---------|------|------|------|------|------|---------|
| **7B** | 14GB | 8GB | 6GB | 4GB | 4GB | 1.5GB |
| **13B** | 26GB | 15GB | 11GB | 8GB | 7GB | 3GB |
| **33B** | 66GB | 38GB | 28GB | 19GB | 18GB | 7GB |
| **70B** | 140GB | 80GB | 60GB | 40GB | 38GB | 15GB |

**KV 缓存**（额外）：
```
7B, ctx=2048, FP16:  ~1 GB
7B, ctx=4096, FP16:  ~2 GB
7B, ctx=2048, Q8_0:  ~500 MB
```

### 4.2 推荐配置

**场景：对话（7B 模型）**
```cpp
n_ctx = 2048              // 足够的对话历史
n_batch = 512             // 快速提示处理
temp = 0.7                // 略有创意
top_p = 0.9               // 标准过滤
repeat_penalty = 1.1      // 轻微重复惩罚
```

**场景：代码生成**
```cpp
n_ctx = 4096              // 长代码上下文
temp = 0.2                // 高确定性
top_p = 0.5               // 严格过滤
top_k = 20                // 少量候选
repeat_penalty = 1.05     // 轻微惩罚（关键字可重复）
```

**场景：创意写作**
```cpp
n_ctx = 4096              // 长文本
temp = 1.2                // 高多样性
top_p = 0.95              // 宽松过滤
top_k = 100               // 大量候选
repeat_penalty = 1.2      // 避免重复
```

---

## 5. 常见错误代码

| 错误 | 代码 | 原因 | 解决方案 |
|------|------|------|---------|
| 模型加载失败 | NULL | 文件不存在/格式错误 | 检查路径和文件完整性 |
| 上下文创建失败 | NULL | 内存不足 | 减少 n_ctx 或增加 RAM |
| Decode 失败 | != 0 | 批次超出限制 | 检查 batch size ≤ n_batch |
| CUDA OOM | ENOMEM | GPU 内存不足 | 减少 n_gpu_layers 或 n_ctx |
| 分段错误 | SIGSEGV | 无效指针 | 检查模型/上下文有效性 |

---

## 6. 环境变量

```bash
# 启用详细日志
export LLAMA_LOG_LEVEL=debug

# CUDA 设备
export CUDA_VISIBLE_DEVICES=0,1

# Metal 性能优化
export GGML_METAL_PATH_RESOURCES=/path/to/ggml-metal.metal

# 线程数
export OMP_NUM_THREADS=8

# CPU 亲和性
export GOMP_CPU_AFFINITY="0-7"
```

---

## 7. 快速诊断

### 检查模型信息
```bash
./llama-cli -m model.gguf --verbose-prompt 2>&1 | head -50
```

### 测试性能
```bash
./llama-bench -m model.gguf -p 512 -n 128
```

### 验证 GPU 加速
```bash
./llama-cli -m model.gguf -ngl 1 -p "test" -n 10 2>&1 | grep -i cuda
```

### 查看内存使用
```bash
# Linux
watch -n 1 'nvidia-smi; free -h'

# macOS
while true; do echo "$(date) - $(vm_stat | grep free)"; sleep 1; done
```

---

## 8. 有用的宏和常量

```cpp
// Token 特殊值
#define LLAMA_TOKEN_NULL -1

// 文件魔数
#define LLAMA_FILE_MAGIC_GGUF 0x46554747  // "GGUF"

// 默认值
#define LLAMA_DEFAULT_SEED 0xFFFFFFFF

// 量化块大小
#define QK4_0 32
#define QK4_1 32
#define QK_K  256

// 线程推荐值
int n_threads_recommended = std::thread::hardware_concurrency();
```

---

## 9. 性能优化检查清单

### 模型加载
- [x] 使用 mmap（默认启用）
- [x] 避免 mlock（除非必要）
- [x] 使用适当的量化级别

### GPU 优化
- [x] n_gpu_layers = -1（全部层）
- [x] offload_kqv = true
- [x] flash_attn_type = ENABLED（如果支持）
- [x] 使用 FP16 KV 缓存

### CPU 优化
- [x] 编译时启用 AVX2/NEON
- [x] 设置合适的线程数
- [x] 使用批处理（n_batch ≥ 512）

### 推理优化
- [x] 缓存 system prompt
- [x] 使用连续批处理（server）
- [x] 调整采样参数

---

## 10. 相关资源

**官方文档**：
- 构建指南: `docs/build.md`
- 服务器开发: `tools/server/README-dev.md`
- 添加模型: `docs/development/HOWTO-add-model.md`

**代码位置**：
- 主 API: `include/llama.h`
- GGML API: `ggml/include/ggml.h`
- 示例代码: `examples/`

**在线资源**：
- GitHub: https://github.com/ggml-org/llama.cpp
- Discord: https://discord.gg/llama-cpp
- Reddit: r/LocalLLaMA

---

**快速开始**：
```bash
# 1. 编译
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j

# 2. 下载模型（或转换自己的）
# huggingface-cli download ggml-org/gemma-3-1b-it-GGUF

# 3. 运行
./build/bin/llama-cli -m model.gguf -p "Hello" -n 50

# 4. 或启动服务器
./build/bin/llama-server -m model.gguf -c 4096 --port 8080
```

---

*本手册持续更新。建议收藏以便随时查阅。*

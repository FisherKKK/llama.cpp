# Day 5: GGUF 文件格式详解

## 课程目标

深入理解 GGUF（GGML Universal File）格式：
- GGUF 文件结构
- 元数据系统
- 张量存储格式
- 文件解析实现
- 实战：解析 GGUF 文件

## 1. GGUF 格式概述

GGUF 是 llama.cpp 使用的模型文件格式，设计目标：
- **自描述**：文件包含完整的模型信息
- **可扩展**：支持添加新的元数据
- **高效**：支持 mmap 直接映射
- **跨平台**：统一的二进制格式

### 1.1 为什么需要 GGUF？

旧格式（GGML）的问题：
- 缺少版本信息
- 元数据不完整
- 难以扩展

GGUF 的改进：
- ✅ 完整的元数据系统
- ✅ 清晰的版本管理
- ✅ 灵活的扩展机制
- ✅ 更好的工具支持

## 2. GGUF 文件结构

```
┌────────────────────────────────────────┐
│ GGUF File Structure                    │
├────────────────────────────────────────┤
│                                        │
│  ┌──────────────────────────────────┐ │
│  │ Header (固定大小)                 │ │
│  ├──────────────────────────────────┤ │
│  │ • Magic: 0x46554747 ("GGUF")    │ │
│  │ • Version: 3                     │ │
│  │ • Tensor Count: N                │ │
│  │ • Metadata KV Count: M           │ │
│  └──────────────────────────────────┘ │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │ Metadata (可变大小)               │ │
│  ├──────────────────────────────────┤ │
│  │ KV Pair 1:                       │ │
│  │   key: "general.architecture"    │ │
│  │   type: STRING                   │ │
│  │   value: "llama"                 │ │
│  ├──────────────────────────────────┤ │
│  │ KV Pair 2:                       │ │
│  │   key: "llama.context_length"    │ │
│  │   type: UINT32                   │ │
│  │   value: 4096                    │ │
│  ├──────────────────────────────────┤ │
│  │ ... (M-2 more pairs)             │ │
│  └──────────────────────────────────┘ │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │ Tensor Info (N个张量的元信息)     │ │
│  ├──────────────────────────────────┤ │
│  │ Tensor 1:                        │ │
│  │   name: "token_embd.weight"      │ │
│  │   n_dims: 2                      │ │
│  │   dims: [4096, 32000]            │ │
│  │   type: Q4_K                     │ │
│  │   offset: 1048576                │ │
│  ├──────────────────────────────────┤ │
│  │ Tensor 2:                        │ │
│  │   name: "blk.0.attn_q.weight"    │ │
│  │   ...                            │ │
│  └──────────────────────────────────┘ │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │ Alignment Padding                │ │
│  │ (对齐到 32/64 字节)               │ │
│  └──────────────────────────────────┘ │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │ Tensor Data (实际权重数据)        │ │
│  ├──────────────────────────────────┤ │
│  │ [Token Embedding Data]           │ │
│  │ Size: 4096 * 32000 * 0.547字节   │ │
│  │ (Q4_K 格式)                      │ │
│  ├──────────────────────────────────┤ │
│  │ [Layer 0 Q Weight Data]          │ │
│  │ ...                              │ │
│  ├──────────────────────────────────┤ │
│  │ [More tensor data...]            │ │
│  └──────────────────────────────────┘ │
│                                        │
└────────────────────────────────────────┘
```

## 3. 文件头解析

### 3.1 头部结构

```c
// 位置：ggml/include/gguf.h:48
#define GGUF_MAGIC 0x46554747  // "GGUF" in little endian
#define GGUF_VERSION 3

struct gguf_header {
    uint32_t magic;             // 魔数 GGUF_MAGIC
    uint32_t version;           // 版本号（当前为 3）
    uint64_t n_tensors;         // 张量数量
    uint64_t n_kv;              // 元数据键值对数量
};
```

### 3.2 读取文件头

```c
// 位置：ggml/src/gguf.c:1523
struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params) {
    FILE * file = fopen(fname, "rb");
    if (!file) {
        return NULL;
    }

    // 读取文件头
    struct gguf_header header;
    fread(&header, sizeof(header), 1, file);

    // 验证魔数
    if (header.magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: 0x%08x\n", header.magic);
        fclose(file);
        return NULL;
    }

    // 检查版本
    if (header.version != GGUF_VERSION) {
        fprintf(stderr, "Unsupported GGUF version: %u\n", header.version);
        fclose(file);
        return NULL;
    }

    printf("GGUF file: %llu tensors, %llu metadata\n",
           header.n_tensors, header.n_kv);

    // ... 继续解析
}
```

## 4. 元数据系统

### 4.1 元数据类型

```c
// 位置：ggml/include/gguf.h:28
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};
```

### 4.2 键值对结构

```c
// 每个元数据条目的结构（内存中）
struct gguf_kv {
    char * key;              // 键名（如 "llama.context_length"）
    enum gguf_type type;     // 值类型
    union {
        uint8_t   uint8;
        int8_t    int8;
        uint16_t  uint16;
        int16_t   int16;
        uint32_t  uint32;
        int32_t   int32;
        float     float32;
        uint64_t  uint64;
        int64_t   int64;
        double    float64;
        bool      bool_;
        struct {
            char * data;
            uint64_t len;
        } str;
        struct {
            enum gguf_type type;
            uint64_t n;
            void * data;
        } arr;
    } value;
};
```

### 4.3 常见元数据键

```python
# 通用元数据
"general.architecture"       # 模型架构 ("llama", "mistral", ...)
"general.name"               # 模型名称
"general.author"             # 作者
"general.version"            # 版本
"general.file_type"          # 文件类型（量化级别）
"general.quantization_version" # 量化版本

# LLaMA 特定元数据
"llama.context_length"       # 上下文长度（如 4096）
"llama.embedding_length"     # 嵌入维度（如 4096）
"llama.block_count"          # 层数（如 32）
"llama.feed_forward_length"  # FFN 隐藏大小（如 11008）
"llama.attention.head_count" # 注意力头数（如 32）
"llama.attention.head_count_kv" # KV 头数（MQA/GQA）
"llama.rope.dimension_count" # RoPE 维度
"llama.rope.freq_base"       # RoPE 频率基数（如 10000.0）

# 词表元数据
"tokenizer.ggml.model"       # tokenizer 类型
"tokenizer.ggml.tokens"      # token 列表（数组）
"tokenizer.ggml.scores"      # token 得分（数组）
"tokenizer.ggml.token_type"  # token 类型（数组）
```

### 4.4 解析元数据

```c
// 位置：ggml/src/gguf.c:1623
static void gguf_read_kv(struct gguf_context * ctx, FILE * file) {
    for (uint64_t i = 0; i < ctx->header.n_kv; i++) {
        struct gguf_kv * kv = &ctx->kv[i];

        // 读取键名
        uint64_t key_len;
        fread(&key_len, sizeof(key_len), 1, file);
        kv->key = malloc(key_len + 1);
        fread(kv->key, 1, key_len, file);
        kv->key[key_len] = '\0';

        // 读取值类型
        fread(&kv->type, sizeof(kv->type), 1, file);

        // 根据类型读取值
        switch (kv->type) {
            case GGUF_TYPE_UINT32:
                fread(&kv->value.uint32, sizeof(uint32_t), 1, file);
                break;

            case GGUF_TYPE_FLOAT32:
                fread(&kv->value.float32, sizeof(float), 1, file);
                break;

            case GGUF_TYPE_STRING: {
                uint64_t str_len;
                fread(&str_len, sizeof(str_len), 1, file);
                kv->value.str.data = malloc(str_len + 1);
                fread(kv->value.str.data, 1, str_len, file);
                kv->value.str.data[str_len] = '\0';
                kv->value.str.len = str_len;
                break;
            }

            case GGUF_TYPE_ARRAY: {
                fread(&kv->value.arr.type, sizeof(enum gguf_type), 1, file);
                fread(&kv->value.arr.n, sizeof(uint64_t), 1, file);
                // 读取数组数据...
                break;
            }

            // ... 其他类型
        }
    }
}
```

## 5. 张量信息解析

### 5.1 张量元信息结构

```c
// 位置：ggml/src/gguf.c
struct gguf_tensor_info {
    char * name;                    // 张量名称
    uint32_t n_dims;                // 维度数（1-4）
    uint64_t ne[GGML_MAX_DIMS];     // 每个维度的大小
    enum ggml_type type;            // 数据类型（F32/Q4_K/...）
    uint64_t offset;                // 在文件中的偏移量
    void * data;                    // 指向实际数据（mmap后）
    size_t size;                    // 数据大小（字节）
};
```

### 5.2 解析张量信息

```c
// 位置：ggml/src/gguf.c:1712
static void gguf_read_tensor_info(struct gguf_context * ctx, FILE * file) {
    for (uint64_t i = 0; i < ctx->header.n_tensors; i++) {
        struct gguf_tensor_info * info = &ctx->infos[i];

        // 读取张量名称
        uint64_t name_len;
        fread(&name_len, sizeof(name_len), 1, file);
        info->name = malloc(name_len + 1);
        fread(info->name, 1, name_len, file);
        info->name[name_len] = '\0';

        // 读取维度数
        fread(&info->n_dims, sizeof(uint32_t), 1, file);

        // 读取每个维度的大小
        for (uint32_t j = 0; j < info->n_dims; j++) {
            fread(&info->ne[j], sizeof(uint64_t), 1, file);
        }

        // 读取数据类型
        fread(&info->type, sizeof(enum ggml_type), 1, file);

        // 读取数据偏移
        fread(&info->offset, sizeof(uint64_t), 1, file);

        // 计算数据大小
        size_t type_size = ggml_type_size(info->type);
        info->size = ggml_row_size(info->type, info->ne[0]);
        for (uint32_t j = 1; j < info->n_dims; j++) {
            info->size *= info->ne[j];
        }

        printf("Tensor %llu: %s [", i, info->name);
        for (uint32_t j = 0; j < info->n_dims; j++) {
            printf("%llu%s", info->ne[j], j < info->n_dims-1 ? ", " : "");
        }
        printf("], type=%s, size=%zu bytes\n",
               ggml_type_name(info->type), info->size);
    }
}
```

## 6. 张量数据加载

### 6.1 数据对齐

GGUF 要求张量数据对齐到 32 字节（某些平台 64 字节）：

```c
// 位置：src/llama-model-loader.cpp:234
static const size_t GGUF_DEFAULT_ALIGNMENT = 32;

// 计算对齐后的偏移
size_t tensor_data_offset = /* 头部 + 元数据 + 张量信息大小 */;
tensor_data_offset = (tensor_data_offset + GGUF_DEFAULT_ALIGNMENT - 1)
                     & ~(GGUF_DEFAULT_ALIGNMENT - 1);
```

### 6.2 使用 mmap 加载

```c
// 位置：src/llama-mmap.cpp:347
struct llama_mmap {
    void * addr;      // 映射地址
    size_t size;      // 映射大小

    llama_mmap(const char * fname, size_t prefetch = 0) {
        int fd = open(fname, O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("cannot open file");
        }

        struct stat st;
        fstat(fd, &st);
        size = st.st_size;

        // 内存映射
        addr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("mmap failed");
        }

        // 建议预取（可选）
        if (prefetch > 0) {
            madvise(addr, size, MADV_WILLNEED);
        }

        close(fd);
    }

    ~llama_mmap() {
        munmap(addr, size);
    }
};
```

### 6.3 访问张量数据

```c
// 位置：src/llama-model-loader.cpp:587
void * llama_model_loader::get_tensor_data(const char * name) {
    // 查找张量信息
    struct gguf_tensor_info * info = gguf_find_tensor(ctx_gguf, name);
    if (!info) {
        return NULL;
    }

    // 计算在文件中的实际地址
    uint8_t * data = (uint8_t *) mmap->addr + tensor_data_offset + info->offset;

    return data;
}
```

## 7. 实战：解析 GGUF 文件

### 7.1 简单的 GGUF 读取工具

```c
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>

void print_gguf_info(const char * fname) {
    // 初始化 GGUF 上下文
    struct gguf_init_params params = {
        .no_alloc = true,   // 不分配内存，只读取元数据
    };
    struct gguf_context * ctx = gguf_init_from_file(fname, params);
    if (!ctx) {
        printf("Failed to open %s\n", fname);
        return;
    }

    // 打印文件信息
    printf("=== GGUF File Info ===\n");
    printf("Version: %u\n", gguf_get_version(ctx));
    printf("Tensors: %zu\n", gguf_get_n_tensors(ctx));
    printf("Metadata pairs: %zu\n", gguf_get_n_kv(ctx));
    printf("\n");

    // 打印关键元数据
    printf("=== Key Metadata ===\n");
    const char * arch = gguf_get_val_str(ctx, gguf_find_key(ctx, "general.architecture"));
    printf("Architecture: %s\n", arch);

    int n_layers = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.block_count"));
    int n_embd = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.embedding_length"));
    int n_heads = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.attention.head_count"));
    printf("Layers: %d\n", n_layers);
    printf("Embedding dim: %d\n", n_embd);
    printf("Attention heads: %d\n", n_heads);
    printf("\n");

    // 打印张量列表（前10个）
    printf("=== Tensor List (first 10) ===\n");
    int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors && i < 10; i++) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);

        int n_dims = gguf_get_tensor_n_dims(ctx, i);
        printf("[%d] %s: ", i, name);

        for (int j = 0; j < n_dims; j++) {
            int64_t dim = gguf_get_tensor_size(ctx, i, j);
            printf("%lld%s", dim, j < n_dims-1 ? " x " : "");
        }
        printf(", type=%s\n", ggml_type_name(type));
    }
    if (n_tensors > 10) {
        printf("... (%d more tensors)\n", n_tensors - 10);
    }

    // 清理
    gguf_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    print_gguf_info(argv[1]);
    return 0;
}
```

编译运行：
```bash
gcc -O3 -o gguf_info gguf_info.c \
    -I./ggml/include \
    -L./build/ggml/src -lggml

./gguf_info model.gguf
```

### 7.2 使用 Python 解析

llama.cpp 提供了 Python 工具：

```bash
# 打印 GGUF 文件信息
python gguf-py/scripts/gguf_dump.py model.gguf

# 修改元数据
python gguf-py/scripts/gguf_set_metadata.py \
    model.gguf \
    --kv "general.name=My Model"

# 转换字节序
python gguf-py/scripts/gguf_convert_endian.py \
    model.gguf model-be.gguf
```

## 8. GGUF vs 其他格式对比

| 特性 | GGUF | PyTorch (.pt) | SafeTensors | HF (.bin) |
|------|------|---------------|-------------|-----------|
| **自描述** | ✅ 完整 | ❌ 需要配置文件 | ⚠️ 部分 | ❌ 需要配置文件 |
| **mmap 支持** | ✅ 原生 | ❌ 不支持 | ✅ 支持 | ❌ 不支持 |
| **量化支持** | ✅ 原生 | ❌ 需要额外工具 | ❌ 需要额外工具 | ❌ 需要额外工具 |
| **跨平台** | ✅ 统一格式 | ⚠️ Python依赖 | ✅ 好 | ⚠️ Python依赖 |
| **文件大小** | 小（量化） | 大（全精度） | 大（全精度） | 大（全精度） |
| **加载速度** | 快（mmap） | 慢（反序列化） | 中等 | 慢（反序列化） |

## 9. 总结

今天我们深入学习了 GGUF 文件格式：

✅ **文件结构**：头部 → 元数据 → 张量信息 → 张量数据
✅ **元数据系统**：灵活的键值对系统
✅ **张量存储**：高效的数据对齐和 mmap 支持
✅ **实战工具**：C 和 Python 的解析示例

### 关键要点

1. **GGUF 是自描述的**：包含完整的模型信息
2. **支持 mmap**：快速加载，节省内存
3. **元数据驱动**：架构信息从文件中读取
4. **量化友好**：直接存储量化权重

## 下一步

明天我们将学习 **Day 6: 模型加载与权重管理**：
- 从 GGUF 到 llama_model
- 权重映射策略
- 多 GPU 分配
- 内存优化技巧

---

**练习**：
1. 使用示例代码解析一个 GGUF 文件
2. 统计不同量化类型的张量占比
3. 计算模型的总参数量和内存占用

📚 [Day 6: 模型加载与权重管理](day06-model-loading.md)

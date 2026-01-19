# Day 4: GGML 内存管理机制

## 课程目标

深入理解 GGML 的内存管理：
- 内存分配器（Allocator）原理
- 上下文（Context）内存池
- mmap 文件映射技术
- 后端缓冲区管理
- 内存优化策略

## 1. GGML 内存管理概览

### 1.1 为什么需要自定义内存管理？

GGML 不使用标准的 `malloc/free`，而是实现了自己的内存管理系统：

**原因**：
1. **性能**：减少频繁的内存分配开销
2. **可控性**：精确控制内存布局和生命周期
3. **优化**：支持 mmap、mlock 等高级特性
4. **跨平台**：统一的内存管理接口

### 1.2 内存管理层次

```
┌─────────────────────────────────────────┐
│ 应用层                                   │
│  ggml_new_tensor_*()                    │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ GGML Context 层                         │
│  • 内存池管理                           │
│  • 张量分配                             │
│  • 对象追踪                             │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ Backend Buffer 层                       │
│  • CPU 缓冲区                           │
│  • GPU 缓冲区                           │
│  • 跨设备传输                           │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 系统层                                   │
│  • malloc/free (CPU)                    │
│  • cudaMalloc (GPU)                     │
│  • mmap (文件映射)                      │
└─────────────────────────────────────────┘
```

## 2. GGML Context（上下文）

### 2.1 Context 结构

```c
// 位置：ggml/src/ggml.c:1823
struct ggml_context {
    size_t mem_size;           // 内存池总大小
    void * mem_buffer;         // 内存池指针
    bool   mem_buffer_owned;   // 是否拥有内存

    int    n_objects;          // 已分配对象数

    struct ggml_object * objects_begin;  // 对象链表头
    struct ggml_object * objects_end;    // 对象链表尾

    struct ggml_scratch scratch;         // 临时内存区域
    struct ggml_scratch scratch_save;    // 保存的临时内存

    // 对齐配置
    size_t alignment;
};
```

### 2.2 初始化 Context

```c
// 位置：ggml/src/ggml.c:2847
struct ggml_context * ggml_init(struct ggml_init_params params) {
    // 1. 静态初始化（只执行一次）
    ggml_critical_section_start();
    static bool is_first_call = true;
    if (is_first_call) {
        ggml_setup_op_has_task_pass();  // 初始化操作表
        is_first_call = false;
    }
    ggml_critical_section_end();

    // 2. 计算所需内存大小
    const size_t mem_size = params.mem_size;

    // 对齐到 GGML_MEM_ALIGN (16字节)
    size_t ctx_size = GGML_OBJECT_SIZE;
    ctx_size = GGML_PAD(ctx_size, GGML_MEM_ALIGN);

    // 3. 分配上下文结构
    struct ggml_context * ctx = NULL;

    if (params.mem_buffer != NULL) {
        // 用户提供的缓冲区
        ctx = (struct ggml_context *)(params.mem_buffer);
        ctx->mem_buffer_owned = false;
        ctx->mem_buffer = (char *)params.mem_buffer + ctx_size;
    } else {
        // 自动分配
        ctx = (struct ggml_context *)malloc(ctx_size + mem_size);
        ctx->mem_buffer_owned = true;
        ctx->mem_buffer = (char *)ctx + ctx_size;
    }

    // 4. 初始化字段
    ctx->mem_size = mem_size;
    ctx->n_objects = 0;
    ctx->objects_begin = NULL;
    ctx->objects_end = NULL;
    ctx->scratch = (struct ggml_scratch) {
        .offs = 0,
        .size = 0,
        .data = NULL,
    };
    ctx->alignment = GGML_MEM_ALIGN;

    GGML_PRINT_DEBUG("ggml_init: context initialized with %zu bytes\n", mem_size);

    return ctx;
}
```

### 2.3 使用示例

```c
// 方式 1: 自动分配
struct ggml_init_params params = {
    .mem_size   = 128*1024*1024,  // 128 MB
    .mem_buffer = NULL,
    .no_alloc   = false,
};
struct ggml_context * ctx = ggml_init(params);

// 方式 2: 用户提供缓冲区
void * buffer = malloc(128*1024*1024);
struct ggml_init_params params = {
    .mem_size   = 128*1024*1024,
    .mem_buffer = buffer,
    .no_alloc   = false,
};
struct ggml_context * ctx = ggml_init(params);

// 方式 3: 延迟分配（只分配元数据）
struct ggml_init_params params = {
    .mem_size   = 0,
    .mem_buffer = NULL,
    .no_alloc   = true,  // 不分配数据，只分配结构
};
struct ggml_context * ctx = ggml_init(params);
```

## 3. 对象分配机制

### 3.1 对象结构

```c
// 位置：ggml/src/ggml.c:1811
struct ggml_object {
    size_t offs;               // 在内存池中的偏移
    size_t size;               // 对象大小
    struct ggml_object * next; // 链表下一个
    enum ggml_object_type type;

    char padding[8];           // 对齐填充
};

enum ggml_object_type {
    GGML_OBJECT_TYPE_TENSOR,
    GGML_OBJECT_TYPE_GRAPH,
    GGML_OBJECT_TYPE_WORK_BUFFER,
};
```

### 3.2 分配张量

```c
// 位置：ggml/src/ggml.c:3124
static struct ggml_tensor * ggml_new_tensor_impl(
    struct ggml_context * ctx,
    enum ggml_type type,
    int n_dims,
    const int64_t * ne,
    struct ggml_tensor * view_src,
    size_t view_offs) {

    // 1. 计算张量结构大小
    size_t size_needed = sizeof(struct ggml_tensor);
    size_needed = GGML_PAD(size_needed, GGML_MEM_ALIGN);

    // 2. 计算数据大小
    size_t data_size = ggml_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }
    data_size = GGML_PAD(data_size, GGML_MEM_ALIGN);

    // 3. 分配对象头
    struct ggml_object * obj_new = ggml_new_object(
        ctx,
        GGML_OBJECT_TYPE_TENSOR,
        size_needed + data_size
    );

    // 4. 初始化张量
    struct ggml_tensor * result = (struct ggml_tensor *)((char *)ctx->mem_buffer + obj_new->offs);

    *result = (struct ggml_tensor) {
        .type = type,
        .backend = GGML_BACKEND_TYPE_CPU,
        .n_dims = n_dims,
        .ne = { 1, 1, 1, 1 },
        .nb = { 0, 0, 0, 0 },
        .op = GGML_OP_NONE,
        .op_params = { 0 },
        .flags = 0,
        .src = { NULL },
        .view_src = view_src,
        .view_offs = view_offs,
        .data = (view_src == NULL) ? (char *)result + size_needed : NULL,
        .name = { 0 },
        .extra = NULL,
    };

    // 5. 设置维度和步长
    for (int i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = ggml_type_size(type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1] * result->ne[i - 1];
    }

    return result;
}
```

### 3.3 内存布局示例

```
Context 内存池布局：

┌────────────────────────────────────────┐
│ Object 1: Tensor                       │  offs=0
│  ┌──────────────────────────────────┐ │
│  │ struct ggml_tensor (120 bytes)   │ │
│  └──────────────────────────────────┘ │
│  ┌──────────────────────────────────┐ │
│  │ Tensor data (aligned)            │ │
│  │ [4096 x 4096 x 4 bytes]          │ │
│  └──────────────────────────────────┘ │
├────────────────────────────────────────┤
│ Object 2: Tensor                       │  offs=67108992
│  ┌──────────────────────────────────┐ │
│  │ struct ggml_tensor               │ │
│  └──────────────────────────────────┘ │
│  ┌──────────────────────────────────┐ │
│  │ Tensor data                      │ │
│  └──────────────────────────────────┘ │
├────────────────────────────────────────┤
│ Object 3: Graph                        │
│  ...                                   │
├────────────────────────────────────────┤
│ Free space                             │
│  (未使用的内存池空间)                  │
└────────────────────────────────────────┘
```

## 4. Scratch Memory（临时内存）

### 4.1 Scratch 机制

Scratch 是一块可重用的临时内存区域，用于存储中间计算结果。

```c
// 位置：ggml/src/ggml.c:4217
struct ggml_scratch {
    size_t offs;   // 当前使用的偏移
    size_t size;   // 总大小
    void * data;   // 数据指针
};

// 设置 scratch
void ggml_scratch_set(struct ggml_context * ctx, struct ggml_scratch scratch) {
    ctx->scratch = scratch;
}

// 保存当前 scratch 状态
void ggml_scratch_save(struct ggml_context * ctx) {
    ctx->scratch_save = ctx->scratch;
}

// 恢复 scratch 状态（重用内存）
void ggml_scratch_load(struct ggml_context * ctx) {
    ctx->scratch = ctx->scratch_save;
}
```

### 4.2 使用示例

```c
// 1. 分配 scratch 缓冲区
size_t scratch_size = 256*1024*1024;  // 256 MB
void * scratch_buffer = malloc(scratch_size);

struct ggml_scratch scratch = {
    .offs = 0,
    .size = scratch_size,
    .data = scratch_buffer,
};

// 2. 设置到 context
ggml_scratch_set(ctx, scratch);

// 3. 构建计算图（会使用 scratch）
struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
struct ggml_tensor * c = ggml_mul_mat(ctx, a, b);  // 中间结果在 scratch

// 4. 保存状态
ggml_scratch_save(ctx);

// 5. 执行计算
ggml_graph_compute(ctx, graph);

// 6. 恢复状态（重用 scratch）
ggml_scratch_load(ctx);

// 再次使用相同的 scratch 空间
struct ggml_tensor * d = ggml_mul_mat(ctx, c, b);
```

## 5. mmap 文件映射

### 5.1 mmap 原理

mmap 将文件直接映射到进程的虚拟地址空间，避免数据拷贝。

```
传统文件读取：
┌────────┐   read()   ┌────────┐   memcpy   ┌────────┐
│ 磁盘   │ ────────> │ 内核   │ ─────────> │ 用户   │
│ 文件   │           │ 缓冲区 │            │ 内存   │
└────────┘           └────────┘            └────────┘
 2 次拷贝

mmap 映射：
┌────────┐           ┌────────┐
│ 磁盘   │ ◄───────> │ 用户   │
│ 文件   │   mmap    │ 内存   │
└────────┘           └────────┘
 0 次拷贝（页面故障时自动加载）
```

### 5.2 mmap 实现

```cpp
// 位置：src/llama-mmap.cpp:347
struct llama_mmap {
    void * addr;      // 映射地址
    size_t size;      // 映射大小

#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMapping;
#else
    int fd;
#endif

    // 构造函数：打开并映射文件
    llama_mmap(const char * fname, size_t prefetch = 0, bool numa = false) {
#ifdef _WIN32
        // Windows 实现
        hFile = CreateFileA(fname, GENERIC_READ, FILE_SHARE_READ,
                           NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Cannot open file");
        }

        LARGE_INTEGER file_size;
        GetFileSizeEx(hFile, &file_size);
        size = file_size.QuadPart;

        hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMapping == NULL) {
            CloseHandle(hFile);
            throw std::runtime_error("CreateFileMapping failed");
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        if (addr == NULL) {
            CloseHandle(hMapping);
            CloseHandle(hFile);
            throw std::runtime_error("MapViewOfFile failed");
        }
#else
        // Linux/macOS 实现
        fd = open(fname, O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Cannot open file");
        }

        struct stat st;
        if (fstat(fd, &st) != 0) {
            close(fd);
            throw std::runtime_error("fstat failed");
        }
        size = st.st_size;

        // 内存映射
        addr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("mmap failed");
        }

        // 建议内核预取（可选）
        if (prefetch > 0) {
            size_t prefetch_size = std::min(prefetch, size);
            if (madvise(addr, prefetch_size, MADV_WILLNEED) != 0) {
                // 警告但不失败
                fprintf(stderr, "Warning: madvise WILLNEED failed\n");
            }
        }

        // NUMA 优化（可选）
        if (numa) {
#ifdef __linux__
            // 分布到所有 NUMA 节点
            if (madvise(addr, size, MADV_SEQUENTIAL) == 0) {
                // 提示顺序访问
            }
#endif
        }
#endif
    }

    // 析构函数：取消映射
    ~llama_mmap() {
#ifdef _WIN32
        UnmapViewOfFile(addr);
        CloseHandle(hMapping);
        CloseHandle(hFile);
#else
        munmap(addr, size);
        close(fd);
#endif
    }
};
```

### 5.3 mlock（锁定内存）

```cpp
// 位置：src/llama-mmap.cpp:512
struct llama_mlock {
    void * addr;
    size_t size;

    llama_mlock(void * ptr, size_t len) : addr(ptr), size(len) {
#ifdef _WIN32
        // Windows: VirtualLock
        if (!VirtualLock(addr, size)) {
            fprintf(stderr, "Warning: VirtualLock failed\n");
        }
#else
        // Linux/macOS: mlock
        if (mlock(addr, size) != 0) {
            fprintf(stderr, "Warning: mlock failed: %s\n", strerror(errno));
            // 不抛出异常，只是警告
        }
#endif
    }

    ~llama_mlock() {
#ifdef _WIN32
        VirtualUnlock(addr, size);
#else
        munlock(addr, size);
#endif
    }
};
```

**mlock 的作用**：
- 防止内存页被 swap 到磁盘
- 保证访问延迟一致
- 适用于对延迟敏感的场景

**使用建议**：
- 只在有足够 RAM 时使用
- 需要 root 权限或配置 ulimit

## 6. Backend Buffer（后端缓冲区）

### 6.1 Buffer 抽象

```c
// 位置：ggml/include/ggml-backend.h:78
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

struct ggml_backend_buffer {
    ggml_backend_buffer_type_t type;  // 缓冲区类型
    ggml_backend_t backend;           // 所属后端

    void * context;                   // 后端特定的上下文

    size_t size;                      // 缓冲区大小
    void * data;                      // 数据指针（CPU 可访问）
};

// 缓冲区操作接口
struct ggml_backend_buffer_i {
    void (*free_buffer)(ggml_backend_buffer_t buffer);
    void * (*get_base)(ggml_backend_buffer_t buffer);
    void (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    void (*set_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                       const void * data, size_t offset, size_t size);
    void (*get_tensor)(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                       void * data, size_t offset, size_t size);
    bool (*cpy_tensor)(ggml_backend_buffer_t buffer, const struct ggml_tensor * src,
                       struct ggml_tensor * dst);
};
```

### 6.2 CPU Buffer

```c
// 位置：ggml/src/ggml-backend.cpp:124
static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->data;
}

static void ggml_backend_cpu_buffer_set_tensor(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor,
    const void * data,
    size_t offset,
    size_t size) {

    memcpy((char *)tensor->data + offset, data, size);
}

static void ggml_backend_cpu_buffer_get_tensor(
    ggml_backend_buffer_t buffer,
    const struct ggml_tensor * tensor,
    void * data,
    size_t offset,
    size_t size) {

    memcpy(data, (const char *)tensor->data + offset, size);
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_alloc(size_t size) {
    // 分配对齐的内存
    void * data = aligned_alloc(GGML_MEM_ALIGN, size);

    ggml_backend_buffer_t buffer = malloc(sizeof(struct ggml_backend_buffer));
    buffer->type = ggml_backend_cpu_buffer_type();
    buffer->context = NULL;
    buffer->size = size;
    buffer->data = data;

    return buffer;
}
```

### 6.3 GPU Buffer（CUDA 示例）

```cpp
// 位置：ggml/src/ggml-cuda/ggml-cuda.cpp:2847
static void ggml_backend_cuda_buffer_set_tensor(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor,
    const void * data,
    size_t offset,
    size_t size) {

    // CPU → GPU 传输
    cudaMemcpy(
        (char *)tensor->data + offset,
        data,
        size,
        cudaMemcpyHostToDevice
    );
}

static void ggml_backend_cuda_buffer_get_tensor(
    ggml_backend_buffer_t buffer,
    const struct ggml_tensor * tensor,
    void * data,
    size_t offset,
    size_t size) {

    // GPU → CPU 传输
    cudaMemcpy(
        data,
        (const char *)tensor->data + offset,
        size,
        cudaMemcpyDeviceToHost
    );
}

ggml_backend_buffer_t ggml_backend_cuda_buffer_alloc(int device, size_t size) {
    cudaSetDevice(device);

    // 分配 GPU 内存
    void * dev_ptr;
    cudaMalloc(&dev_ptr, size);

    ggml_backend_buffer_t buffer = malloc(sizeof(struct ggml_backend_buffer));
    buffer->type = ggml_backend_cuda_buffer_type(device);
    buffer->context = (void *)(intptr_t)device;
    buffer->size = size;
    buffer->data = dev_ptr;

    return buffer;
}
```

## 7. 内存优化策略

### 7.1 计算内存需求

```cpp
size_t estimate_model_memory(const llama_hparams & hparams) {
    size_t mem_size = 0;

    // 1. Token 嵌入层: [n_vocab, n_embd]
    mem_size += hparams.n_vocab * hparams.n_embd * sizeof(float);

    // 2. 每个 Transformer 层
    for (int i = 0; i < hparams.n_layer; i++) {
        // Q, K, V 投影: [n_embd, n_embd] * 3
        mem_size += 3 * hparams.n_embd * hparams.n_embd * sizeof(float);

        // 输出投影: [n_embd, n_embd]
        mem_size += hparams.n_embd * hparams.n_embd * sizeof(float);

        // FFN: [n_embd, n_ff] * 3 (gate, up, down)
        mem_size += 3 * hparams.n_embd * hparams.n_ff * sizeof(float);

        // Layer Norm 权重: [n_embd] * 2
        mem_size += 2 * hparams.n_embd * sizeof(float);
    }

    // 3. 输出层: [n_embd, n_vocab]
    mem_size += hparams.n_embd * hparams.n_vocab * sizeof(float);

    // 4. 额外开销（元数据、对齐等）
    mem_size = (size_t)(mem_size * 1.1);

    return mem_size;
}
```

### 7.2 动态内存管理

```cpp
// 根据可用内存动态调整
size_t available_mem = get_available_system_memory();
size_t model_mem = estimate_model_memory(hparams);

if (available_mem < model_mem) {
    // 内存不足，使用更激进的策略

    // 策略 1: 减少 KV 缓存大小
    ctx_params.n_ctx = 1024;  // 而不是 4096

    // 策略 2: 部分 offload 到 GPU
    model_params.n_gpu_layers = 16;  // 而不是全部

    // 策略 3: 使用更多的 CPU，减少批大小
    ctx_params.n_batch = 128;  // 而不是 512

    fprintf(stderr, "Warning: Limited memory, using conservative settings\n");
}
```

### 7.3 内存池复用

```cpp
// 多个 context 共享一个内存池
void * shared_buffer = malloc(512*1024*1024);  // 512 MB

struct ggml_init_params params1 = {
    .mem_size   = 256*1024*1024,
    .mem_buffer = shared_buffer,
    .no_alloc   = false,
};
struct ggml_context * ctx1 = ggml_init(params1);

// 使用 ctx1...

ggml_free(ctx1);

// 复用相同的 buffer
struct ggml_init_params params2 = {
    .mem_size   = 256*1024*1024,
    .mem_buffer = shared_buffer,
    .no_alloc   = false,
};
struct ggml_context * ctx2 = ggml_init(params2);
```

## 8. 实践练习

### 练习 1：内存使用分析

编写工具分析模型的内存使用：

```cpp
void analyze_memory_usage(struct ggml_context * ctx) {
    size_t total_size = 0;
    int n_tensors = 0;

    std::map<ggml_type, size_t> size_by_type;

    for (struct ggml_object * obj = ctx->objects_begin;
         obj != NULL; obj = obj->next) {

        if (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            struct ggml_tensor * tensor = (struct ggml_tensor *)
                ((char *)ctx->mem_buffer + obj->offs);

            size_t tensor_size = ggml_nbytes(tensor);
            total_size += tensor_size;
            n_tensors++;

            size_by_type[tensor->type] += tensor_size;
        }
    }

    printf("Memory Analysis:\n");
    printf("  Total tensors: %d\n", n_tensors);
    printf("  Total size: %.2f MB\n", total_size / (1024.0 * 1024));

    for (auto & [type, size] : size_by_type) {
        printf("  %s: %.2f MB (%.1f%%)\n",
               ggml_type_name(type),
               size / (1024.0 * 1024),
               100.0 * size / total_size);
    }
}
```

### 练习 2：自定义内存分配器

实现一个简单的 arena 分配器：

```cpp
class ArenaAllocator {
    char * buffer;
    size_t size;
    size_t offset;

public:
    ArenaAllocator(size_t sz) : size(sz), offset(0) {
        buffer = (char *)malloc(sz);
    }

    ~ArenaAllocator() {
        free(buffer);
    }

    void * allocate(size_t sz, size_t align = 16) {
        // 对齐
        offset = (offset + align - 1) & ~(align - 1);

        if (offset + sz > size) {
            throw std::bad_alloc();
        }

        void * ptr = buffer + offset;
        offset += sz;
        return ptr;
    }

    void reset() {
        offset = 0;  // 重用所有内存
    }

    size_t used() const { return offset; }
    size_t available() const { return size - offset; }
};
```

## 9. 总结

今天我们深入学习了 GGML 的内存管理：

✅ **Context 机制**：内存池管理与对象分配
✅ **Scratch 内存**：临时内存的高效复用
✅ **mmap 技术**：零拷贝文件映射
✅ **Backend Buffer**：统一的后端缓冲区接口
✅ **优化策略**：动态调整与内存复用

### 关键要点

1. **内存池**：减少频繁分配，提高性能
2. **mmap**：大文件快速加载，节省内存
3. **对齐**：保证 SIMD 和 GPU 性能
4. **复用**：Scratch 机制避免重复分配

## 下一步

明天我们将学习 **Day 7: 计算图构建与调度**：
- llama_graph_builder 实现
- 构建完整的 Transformer 计算图
- 后端调度器原理
- 图优化技术

---

**练习**：
1. 分析一个模型的内存使用分布
2. 对比 mmap 和传统加载的性能差异
3. 实现一个简单的内存池管理器

📚 [Day 7: 计算图构建与调度](day07-computation-graph.md)

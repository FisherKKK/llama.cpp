# Day 12: GPU 后端实现（CUDA/Metal）

## 课程目标

掌握 GPU 加速的核心技术：
- CUDA 编程基础
- cuBLAS 矩阵乘法
- 自定义 CUDA 内核
- Metal Shaders（macOS）
- 性能优化技巧

## 1. GPU 加速原理

### 1.1 CPU vs GPU

```
CPU:
    • 核心数: 8-64
    • 时钟频率: 3-5 GHz
    • 适合: 复杂逻辑、分支密集
    • 峰值性能: ~1 TFLOPS

GPU (NVIDIA RTX 4090):
    • 核心数: 16,384 (CUDA cores)
    • 时钟频率: ~2.5 GHz
    • 适合: 大规模并行计算
    • 峰值性能: ~82 TFLOPS (FP32)
                ~330 TFLOPS (FP16 TensorCore)

→ GPU 在矩阵运算上有 100x+ 优势
```

### 1.2 llama.cpp 中的 GPU 使用

```
模型推理的计算分布：

┌─────────────────────────────────┐
│ Token 嵌入 (GPU)                 │ 2%
├─────────────────────────────────┤
│ 32x Transformer Layer:          │
│   • Q,K,V 投影 (GPU)             │ 40%
│   • 注意力计算 (GPU)             │ 30%
│   • FFN (GPU)                    │ 25%
├─────────────────────────────────┤
│ 输出投影 (GPU)                   │ 2%
├─────────────────────────────────┤
│ Softmax/采样 (CPU)               │ 1%
└─────────────────────────────────┘

→ 95%+ 计算在 GPU 上
```

## 2. CUDA 后端

### 2.1 CUDA 基础

**线程层次**：
```
Grid (整个计算)
  ↓
Block (工作组, 如 256 线程)
  ↓
Thread (单个线程)
  ↓
Warp (32 线程, 硬件调度单位)
```

**内存层次**：
```
Global Memory (GPU VRAM)    ~16-24 GB, 慢 (~1000 GB/s)
  ↓
L2 Cache                    ~MB 级, 中速
  ↓
Shared Memory (per block)   ~100 KB, 快 (~20 TB/s)
  ↓
Registers (per thread)      ~64 KB, 极快
```

### 2.2 简单的 CUDA 内核

```cuda
// 位置：ggml/src/ggml-cuda/vector-add.cu
__global__ void vector_add_kernel(
    const float * x,
    const float * y,
    float * z,
    int n) {

    // 全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}

// 主机端调用
void ggml_cuda_op_add(
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst) {

    const int n = ggml_nelements(dst);

    // 配置线程块
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    // 启动内核
    vector_add_kernel<<<grid_size, block_size>>>(
        (float *)src0->data,
        (float *)src1->data,
        (float *)dst->data,
        n);

    // 同步等待完成
    cudaDeviceSynchronize();
}
```

### 2.3 cuBLAS 矩阵乘法

```cpp
// 位置：ggml/src/ggml-cuda/ggml-cuda.cpp:1234
void ggml_cuda_mul_mat_cublas(
    const ggml_tensor * src0,  // [K, M]
    const ggml_tensor * src1,  // [K, N]
    ggml_tensor * dst) {       // [M, N]

    const int M = src0->ne[1];
    const int N = src1->ne[1];
    const int K = src0->ne[0];

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS 矩阵乘法：C = alpha * A @ B + beta * C
    cublasStatus_t status = cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T,    // A 转置
        CUBLAS_OP_N,    // B 不转置
        M, N, K,
        &alpha,
        (float *)src0->data, K,
        (float *)src1->data, K,
        &beta,
        (float *)dst->data, M);

    CUDA_CHECK(status);
}
// 性能: ~15 TFLOPS (RTX 4090)
// cuBLAS 已经高度优化，通常不需要自己写
```

### 2.4 量化矩阵乘法内核

llama.cpp 的核心优化：自定义量化 CUDA 内核

```cuda
// 位置：ggml/src/ggml-cuda/mmq.cuh:456
// Q4_K × Q8_0 矩阵乘法

template<int qk, int qr, int qi>
__global__ void mul_mat_q4_K(
    const void * __restrict__ vx,  // Q4_K 权重 [K, M]
    const void * __restrict__ vy,  // Q8_0 激活 [K, N]
    float * __restrict__ dst,      // 输出 [M, N]
    const int ncols_x,
    const int nrows_x) {

    // 每个 block 处理一行
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // 共享内存：减少 global memory 访问
    __shared__ float tmp[WARP_SIZE];

    const block_q4_K * x = (const block_q4_K *)vx + row * (ncols_x / QK_K);
    const block_q8_0 * y = (const block_q8_0 *)vy;

    float sumf = 0.0f;

    // 遍历列方向（K 维度）
    for (int i = 0; i < ncols_x; i += QK_K) {
        const int ib = i / QK_K;

        // 1. 加载量化参数
        const float d = x[ib].d;
        const float dmin = x[ib].dmin;

        // 2. 反量化并点积
        int sumi = 0;
        for (int j = 0; j < QK_K/2; j++) {
            // 从 4-bit 提取
            const int q = get_q4_value(x[ib].qs, j);

            // Q4 × Q8 整数乘法
            const int y_val = y[i/QK8_0 + j].qs[tid];
            sumi += q * y_val;
        }

        // 3. 缩放并累加
        sumf += d * sumi - dmin;
    }

    // 4. Warp 内归约
    sumf = warp_reduce_sum(sumf);

    // 5. 写回结果
    if (tid == 0) {
        dst[row] = sumf;
    }
}
```

**性能对比**：
```
FP32 MatMul (cuBLAS):  15 TFLOPS
Q4_K MatMul (custom):  40+ TFLOPS (有效)
    • 计算量减少 8x (4-bit vs 32-bit)
    • 内存带宽减少 8x
    • 总体性能提升 2-3x
```

### 2.5 Flash Attention CUDA 实现

```cuda
// 位置：ggml/src/ggml-cuda/fattn.cuh:823
__global__ void flash_attn_kernel(
    const float * Q,           // [n_head, n_tokens, d_head]
    const float * K,           // [n_head, kv_size, d_head]
    const float * V,           // [n_head, kv_size, d_head]
    float * O,                 // [n_head, n_tokens, d_head]
    const int n_tokens,
    const int kv_size,
    const int d_head,
    const float scale) {

    const int head_idx = blockIdx.y;
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // 共享内存
    __shared__ float Q_shared[128];
    __shared__ float K_shared[128];
    __shared__ float V_shared[128];

    // 1. 加载 Q 到共享内存
    if (tid < d_head) {
        Q_shared[tid] = Q[head_idx * n_tokens * d_head + token_idx * d_head + tid];
    }
    __syncthreads();

    // 在线 softmax
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float output[128] = {0};

    // 2. 分块处理 K, V
    const int n_blocks = (kv_size + 31) / 32;
    for (int block = 0; block < n_blocks; block++) {
        int kv_idx = block * 32 + tid;

        // 2a. 加载 K 块
        if (kv_idx < kv_size && tid < d_head) {
            K_shared[tid] = K[head_idx * kv_size * d_head + kv_idx * d_head + tid];
        }
        __syncthreads();

        // 2b. Q @ K^T
        float score = 0.0f;
        if (kv_idx < kv_size) {
            for (int i = 0; i < d_head; i++) {
                score += Q_shared[i] * K_shared[i];
            }
            score *= scale;

            // 2c. 在线更新最大值
            float new_max = fmaxf(row_max, score);
            float exp_diff = expf(row_max - new_max);

            row_sum = row_sum * exp_diff + expf(score - new_max);
            row_max = new_max;
        }

        // 2d. 加载 V 并累积
        if (kv_idx < kv_size && tid < d_head) {
            V_shared[tid] = V[head_idx * kv_size * d_head + kv_idx * d_head + tid];
        }
        __syncthreads();

        if (kv_idx < kv_size) {
            float attn = expf(score - row_max) / row_sum;
            for (int i = 0; i < d_head; i++) {
                output[i] += attn * V_shared[i];
            }
        }
        __syncthreads();
    }

    // 3. 写回输出
    if (tid < d_head) {
        O[head_idx * n_tokens * d_head + token_idx * d_head + tid] = output[tid];
    }
}
```

## 3. Metal 后端（macOS）

### 3.1 Metal Shading Language

```metal
// 位置：ggml/src/ggml-metal/ggml-metal.metal:234
kernel void kernel_add(
    device const float * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {

    dst[id] = src0[id] + src1[id];
}
```

### 3.2 Metal 矩阵乘法

```metal
// 使用 Metal Performance Shaders (MPS)
kernel void kernel_mul_mat_f32(
    device const float * src0 [[buffer(0)]],  // [K, M]
    device const float * src1 [[buffer(1)]],  // [K, N]
    device float * dst [[buffer(2)]],         // [M, N]
    constant int & M [[buffer(3)]],
    constant int & N [[buffer(4)]],
    constant int & K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {

    const int row = gid.y;  // M 维度
    const int col = gid.x;  // N 维度

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += src0[row * K + k] * src1[k * N + col];
    }

    dst[row * N + col] = sum;
}
```

**调用代码**：
```objc
// 位置：ggml/src/ggml-metal/ggml-metal.m:567
id<MTLComputePipelineState> pipeline = ...;
id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

[encoder setComputePipelineState:pipeline];
[encoder setBuffer:src0_buf offset:0 atIndex:0];
[encoder setBuffer:src1_buf offset:0 atIndex:1];
[encoder setBuffer:dst_buf offset:0 atIndex:2];

MTLSize gridSize = MTLSizeMake(N, M, 1);
MTLSize threadGroupSize = MTLSizeMake(8, 8, 1);

[encoder dispatchThreads:gridSize
  threadsPerThreadgroup:threadGroupSize];

[encoder endEncoding];
[commandBuffer commit];
[commandBuffer waitUntilCompleted];
```

## 4. 性能优化技巧

### 4.1 内存合并访问

```cuda
// ❌ 坏例子：非合并访问
__global__ void bad_kernel(float * data, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx * stride] = ...;  // 跨步访问，无法合并
}

// ✅ 好例子：合并访问
__global__ void good_kernel(float * data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = ...;  // 连续访问，可以合并
}
// 性能差异：10x+
```

### 4.2 共享内存优化

```cuda
// 使用共享内存缓存数据
__global__ void optimized_matmul(
    const float * A,
    const float * B,
    float * C,
    int M, int N, int K) {

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 分块加载到共享内存
    for (int tile = 0; tile < K / BLOCK_SIZE; tile++) {
        // 加载 A 块
        As[threadIdx.y][threadIdx.x] = A[row * K + tile * BLOCK_SIZE + threadIdx.x];

        // 加载 B 块
        Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * N + col];

        __syncthreads();

        // 计算
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}
// 性能提升：5-10x
```

### 4.3 占用率优化

```cpp
// 检查内核占用率
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks,
    my_kernel,
    block_size,
    dynamic_smem_size);

printf("Occupancy: %.2f%%\n", 100.0 * num_blocks / max_blocks_per_sm);

// 优化目标：>50% 占用率
```

### 4.4 流水线并行

```cpp
// 使用 CUDA Streams 并行执行
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Layer 0 在 stream1
kernel_layer_0<<<grid, block, 0, stream1>>>(input, layer0_out);

// Layer 1 在 stream2（与 layer0 并行）
kernel_layer_1<<<grid, block, 0, stream2>>>(layer0_out, layer1_out);

// 同步
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

## 5. 多 GPU 支持

### 5.1 张量并行

```cpp
// 位置：src/llama-model.cpp:5123
void distribute_tensor_across_gpus(
    ggml_tensor * tensor,
    const std::vector<int> & gpu_ids) {

    const int n_gpu = gpu_ids.size();
    const int rows_per_gpu = tensor->ne[1] / n_gpu;

    for (int i = 0; i < n_gpu; i++) {
        cudaSetDevice(gpu_ids[i]);

        // 为每个 GPU 分配一部分数据
        void * gpu_data;
        size_t slice_size = rows_per_gpu * tensor->ne[0] * ggml_element_size(tensor);
        cudaMalloc(&gpu_data, slice_size);

        // 拷贝数据片段
        cudaMemcpy(gpu_data,
                   (char*)tensor->data + i * slice_size,
                   slice_size,
                   cudaMemcpyHostToDevice);

        tensor->extra_gpu[i] = gpu_data;
    }
}
```

### 5.2 流水线并行

```
GPU 0: Layer 0-10
GPU 1: Layer 11-21
GPU 2: Layer 22-31

Input → GPU0 → (Transfer) → GPU1 → (Transfer) → GPU2 → Output
```

## 6. 性能对比

| 平台 | 硬件 | FP32 (TFLOPS) | Q4_K (有效TFLOPS) | 7B模型速度 |
|------|------|--------------|-----------------|-----------|
| **CPU** | i9-13900K | 1.0 | 2 | ~5 t/s |
| **GPU (CUDA)** | RTX 4090 | 82 | 150+ | ~120 t/s |
| **GPU (Metal)** | M2 Max | 13 | 25 | ~40 t/s |
| **多GPU** | 4× RTX 4090 | 328 | 600+ | ~400 t/s |

t/s = tokens per second (每秒生成token数)

## 7. 总结

今天我们学习了 GPU 后端实现：

✅ **CUDA 基础**：线程模型、内存层次
✅ **cuBLAS**：高性能矩阵乘法
✅ **自定义内核**：量化、Flash Attention
✅ **Metal**：macOS GPU 加速
✅ **性能优化**：内存访问、共享内存、并行

### 关键要点

1. **cuBLAS 用于标准 MatMul**：已优化到极致
2. **自定义内核用于量化**：特殊操作需要定制
3. **内存访问是关键**：合并访问、共享内存
4. **多 GPU 扩展**：张量并行 + 流水线并行

## 课程完成！

恭喜你完成了 llama.cpp 的 14 天深度学习课程！

回顾一下你已经掌握的技能：
- ✅ llama.cpp 架构设计
- ✅ GGML 张量库
- ✅ GGUF 文件格式
- ✅ 模型加载与推理
- ✅ 注意力机制优化
- ✅ KV 缓存系统
- ✅ 量化技术
- ✅ 采样策略
- ✅ CPU/GPU 后端优化

**下一步**：开始实践项目，为 llama.cpp 贡献代码！

---

📚 [回到课程总览](README.md) | [Day 14: 实战项目与总结](day14-tools-practice.md)

# 故障排查指南

本指南帮助你快速定位和解决 llama.cpp 使用中的常见问题。

---

## 🔍 诊断流程

### 第一步：确定问题类别

```
┌─ 编译/安装问题 → 第 1 节
├─ 加载/启动问题 → 第 2 节
├─ 性能问题 → 第 3 节
├─ 内存问题 → 第 4 节
├─ 输出质量问题 → 第 5 节
└─ GPU 相关问题 → 第 6 节
```

---

## 1. 编译/安装问题

### 问题：找不到 CUDA

**症状**：
```
CMake Error: CUDA not found
```

**诊断**：
```bash
# 检查 CUDA 安装
nvcc --version
which nvcc

# 检查环境变量
echo $CUDA_HOME
echo $PATH | grep cuda
```

**解决方案**：
```bash
# 安装 CUDA Toolkit
# Ubuntu:
sudo apt install nvidia-cuda-toolkit

# 设置环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 重新编译
rm -rf build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

### 问题：编译错误 "undefined reference"

**症状**：
```
undefined reference to `llama_model_load_from_file'
```

**诊断**：
```bash
# 检查链接库
ldd ./build/bin/llama-cli | grep llama
```

**解决方案**：
```bash
# 完全清理重建
rm -rf build
cmake -B build
cmake --build build --config Release -j

# 如果仍然失败，检查 CMakeLists.txt
```

### 问题：Metal 在 macOS 上不工作

**症状**：
```
ggml_metal_init: error: failed to load Metal library
```

**诊断**：
```bash
# 检查 Xcode Command Line Tools
xcode-select -p

# 检查 Metal 文件
ls ggml/src/ggml-metal/ggml-metal.metal
```

**解决方案**：
```bash
# 安装 Xcode Command Line Tools
xcode-select --install

# 重新编译
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release
```

---

## 2. 加载/启动问题

### 问题：模型加载失败

**症状**：
```
error: failed to load model from 'model.gguf'
llama_model_load: failed to load model
```

**诊断步骤**：

1. **检查文件存在**
```bash
ls -lh model.gguf
file model.gguf
```

2. **验证文件完整性**
```bash
# 检查文件头
hexdump -C model.gguf | head -20
# 应该看到 "GGUF" (47 47 55 46)

# 或使用 Python 工具
python gguf-py/scripts/gguf_dump.py model.gguf | head -50
```

3. **检查权限**
```bash
chmod 644 model.gguf
```

**常见原因与解决**：

| 原因 | 症状 | 解决方案 |
|------|------|---------|
| 文件损坏 | 加载到一半失败 | 重新下载 |
| 格式不兼容 | 魔数错误 | 使用新版 llama.cpp |
| 权限不足 | Permission denied | `chmod 644` |
| 磁盘已满 | mmap 失败 | 清理空间 |

### 问题：上下文创建失败

**症状**：
```
llama_new_context_with_model: failed to create context
```

**诊断**：
```cpp
// 检查返回值
llama_context * ctx = llama_new_context_with_model(model, ctx_params);
if (ctx == NULL) {
    fprintf(stderr, "Failed to create context\n");
    // 检查具体原因
}
```

**解决方案**：
```bash
# 减少内存需求
./llama-cli -m model.gguf -c 1024  # 减小上下文
./llama-cli -m model.gguf -ngl 16  # 部分 GPU
```

---

## 3. 性能问题

### 问题：推理速度慢

**症状**：
```
生成速度 < 5 tokens/s（7B 模型在现代硬件上）
```

**诊断清单**：

```bash
# 1. 检查是否使用 GPU
./llama-cli -m model.gguf -ngl 1 -p "test" -n 10 2>&1 | grep -i "cuda\|metal"

# 2. 检查 GPU 利用率
nvidia-smi  # 或 sudo powermetrics（macOS）

# 3. 运行基准测试
./llama-bench -m model.gguf -p 512 -n 128 -ngl 32

# 4. 检查 CPU 利用率
htop  # 查看线程是否充分利用
```

**优化步骤**（按优先级）：

1. **启用 GPU 全层 offload**
```bash
./llama-cli -m model.gguf -ngl -1
```

2. **使用量化模型**
```bash
./llama-quantize model-f16.gguf model-q4.gguf Q4_K_M
./llama-cli -m model-q4.gguf
```

3. **增加批大小**
```bash
./llama-cli -m model.gguf -b 512 -ub 512
```

4. **调整线程数**
```bash
# 设置为物理核心数
./llama-cli -m model.gguf -t $(nproc)
```

5. **启用编译优化**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
```

### 问题：提示处理慢

**症状**：
```
Prompt processing: 5 tokens/s（应该 > 100 t/s）
```

**诊断**：
```bash
# 检查批大小
./llama-cli -m model.gguf -p "long prompt..." -n 1 --verbose-prompt
```

**解决方案**：
```bash
# 增加批大小
./llama-cli -m model.gguf -b 2048 -ub 512

# 或在代码中
ctx_params.n_batch = 2048;
ctx_params.n_ubatch = 512;
```

### 问题：GPU 未充分利用

**症状**：
```
nvidia-smi 显示 GPU 利用率 < 50%
```

**诊断**：
```bash
# 检查配置
./llama-cli -m model.gguf -ngl -1 --verbose-prompt 2>&1 | grep "offload"
```

**可能原因**：

1. **未全层 offload**
```bash
# 解决：使用 -ngl -1
./llama-cli -m model.gguf -ngl -1
```

2. **批大小太小**
```bash
# 解决：增加批大小
./llama-cli -m model.gguf -ngl -1 -b 512
```

3. **未启用 Flash Attention**
```cpp
// 解决：在代码中启用
ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
```

---

## 4. 内存问题

### 问题：CUDA Out of Memory

**症状**：
```
CUDA error: out of memory
```

**诊断**：
```bash
# 检查可用内存
nvidia-smi --query-gpu=memory.free --format=csv

# 估算需求
# 7B Q4_K: ~4GB 模型 + ~500MB KV缓存(ctx=2048) = ~4.5GB
```

**解决方案（优先级排序）**：

1. **减少 GPU 层数**
```bash
./llama-cli -m model.gguf -ngl 24  # 而不是 -1
```

2. **减少上下文**
```bash
./llama-cli -m model.gguf -ngl -1 -c 1024  # 而不是 4096
```

3. **使用更激进的量化**
```bash
# Q3_K_M 比 Q4_K_M 省 25%
./llama-quantize model.gguf model-q3.gguf Q3_K_M
```

4. **使用 Q8_0 KV 缓存**（损失质量）
```cpp
ctx_params.type_k = GGML_TYPE_Q8_0;
ctx_params.type_v = GGML_TYPE_Q8_0;
```

### 问题：系统内存不足

**症状**：
```
std::bad_alloc
或系统 OOM killer 杀死进程
```

**诊断**：
```bash
# 检查可用内存
free -h

# 监控内存使用
watch -n 1 'free -h; ps aux | grep llama'
```

**解决方案**：

1. **启用 mmap**（默认）
```cpp
model_params.use_mmap = true;
```

2. **不使用 mlock**（默认）
```cpp
model_params.use_mlock = false;
```

3. **部分 offload 到 GPU**
```bash
./llama-cli -m model.gguf -ngl 16  # 将部分转移到 GPU
```

### 问题：内存泄漏

**症状**：
```
内存使用持续增长
```

**诊断**：
```bash
# 使用 valgrind
valgrind --leak-check=full --show-leak-kinds=all \
  ./build/bin/llama-cli -m model.gguf -p "test" -n 10
```

**常见原因**：
- 未释放 context
- 未释放 model
- 未释放 sampler
- 未释放 batch

**解决方案**：
```cpp
// 确保清理资源
llama_sampler_free(sampler);
llama_free(ctx);
llama_free_model(model);
llama_backend_free();
```

---

## 5. 输出质量问题

### 问题：输出重复

**症状**：
```
"The cat is cute. The cat is cute. The cat is cute..."
```

**解决方案**：
```bash
# 增加重复惩罚
./llama-cli -m model.gguf --repeat-penalty 1.2

# 调整采样参数
./llama-cli -m model.gguf \
  --repeat-penalty 1.2 \
  --frequency-penalty 0.5 \
  --presence-penalty 0.5
```

### 问题：输出混乱/无意义

**症状**：
```
"asdf jkl; qwer..."（乱码）
```

**可能原因**：

1. **温度太高**
```bash
# 解决：降低温度
./llama-cli -m model.gguf --temp 0.7  # 或更低
```

2. **量化过度**
```bash
# 解决：使用更高质量量化
# Q2_K → Q4_K_M → Q6_K
```

3. **tokenizer 错误**
```bash
# 诊断：检查模型元数据
python gguf-py/scripts/gguf_dump.py model.gguf | grep tokenizer
```

### 问题：输出截断

**症状**：
```
输出提前结束，未达到 max_tokens
```

**诊断**：
```bash
# 检查 EOS token
./llama-cli -m model.gguf -p "test" -n 100 --log-disable 2>&1 | grep -i "eos"
```

**解决方案**：
```cpp
// 检查是否遇到 EOS
if (llama_token_is_eog(model, next_token)) {
    fprintf(stderr, "Hit EOS token\n");
    break;
}

// 或忽略 EOS（小心使用）
if (next_token == llama_token_eos(model) && i < max_tokens - 1) {
    continue;  // 跳过 EOS
}
```

---

## 6. GPU 相关问题

### 问题：GPU 未被识别

**症状**：
```
no CUDA-capable device is detected
```

**诊断**：
```bash
# 检查 GPU
nvidia-smi
lspci | grep -i nvidia

# 检查驱动
cat /proc/driver/nvidia/version

# 检查 CUDA
nvcc --version
```

**解决方案**：
```bash
# 安装/更新驱动
sudo ubuntu-drivers autoinstall

# 或手动安装 NVIDIA 驱动
# https://www.nvidia.com/Download/index.aspx
```

### 问题：CUDA 版本不匹配

**症状**：
```
CUDA driver version is insufficient for CUDA runtime version
```

**诊断**：
```bash
# 检查版本
nvidia-smi  # Driver version
nvcc --version  # Runtime version
```

**解决方案**：
- 升级驱动以匹配 CUDA runtime
- 或降级 CUDA runtime 以匹配驱动

### 问题：多 GPU 不工作

**症状**：
```
只有一个 GPU 被使用
```

**诊断**：
```bash
# 检查可见 GPU
echo $CUDA_VISIBLE_DEVICES
nvidia-smi -L

# 监控所有 GPU
watch -n 1 nvidia-smi
```

**解决方案**：
```bash
# 设置可见设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 使用分层模式
./llama-cli -m model.gguf -ngl -1 -sm layer

# 验证
./llama-cli -m model.gguf -ngl -1 --verbose-prompt 2>&1 | grep "split"
```

---

## 7. 调试技巧

### 启用详细日志

```bash
# 环境变量
export LLAMA_LOG_LEVEL=debug
export LLAMA_LOG_FILE=debug.log

# 运行
./llama-cli -m model.gguf -p "test" -n 10

# 查看日志
cat debug.log
```

### 使用 GDB 调试

```bash
# 编译调试版本
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# 运行 GDB
gdb --args ./build/bin/llama-cli -m model.gguf -p "test"

# GDB 命令
(gdb) run
(gdb) backtrace  # 查看调用栈
(gdb) print variable  # 打印变量
(gdb) continue
```

### 性能分析

```bash
# Linux: perf
perf record -g ./llama-cli -m model.gguf -p "test" -n 100
perf report

# macOS: Instruments
xcrun xctrace record --template 'Time Profiler' \
  --launch ./llama-cli -- -m model.gguf -p "test" -n 100
```

### 内存分析

```bash
# Valgrind
valgrind --leak-check=full \
  --track-origins=yes \
  ./llama-cli -m model.gguf -p "test" -n 10

# AddressSanitizer
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address"
cmake --build build
./build/bin/llama-cli -m model.gguf -p "test"
```

---

## 8. 常见错误信息

| 错误信息 | 含义 | 解决方案 |
|---------|------|---------|
| `failed to load model` | 模型文件问题 | 检查文件路径和完整性 |
| `out of memory` | 内存不足 | 减少 n_ctx 或使用量化 |
| `CUDA error` | GPU 错误 | 检查驱动和 CUDA 版本 |
| `invalid model file` | 格式不支持 | 使用兼容版本或转换 |
| `context size exceeded` | 超出上下文限制 | 增加 n_ctx 或截断输入 |
| `batch size exceeded` | 批次太大 | 减少 batch size |
| `tensor not found` | 权重缺失 | 重新下载/转换模型 |

---

## 9. 求助清单

当需要寻求帮助时，请提供：

```markdown
**环境信息**:
- OS: [Ubuntu 22.04 / macOS 14.0 / Windows 11]
- CPU: [i9-13900K]
- GPU: [RTX 4090 24GB / M2 Max / None]
- RAM: [32GB]
- llama.cpp 版本: [commit hash 或 release tag]
- CUDA版本: [12.2 / N/A]

**编译选项**:
```bash
cmake -B build -DGGML_CUDA=ON
```

**命令/代码**:
```bash
./llama-cli -m model.gguf -ngl -1 -p "test" -n 10
```

**完整错误输出**:
```
[粘贴完整错误信息]
```

**已尝试的解决方案**:
1. ...
2. ...

**其他信息**:
- 模型文件: [model.gguf, 4.2GB, Q4_K_M]
- 问题是否可复现: [是/否]
```

---

## 10. 快速诊断脚本

```bash
#!/bin/bash
# diagnose.sh - llama.cpp 诊断脚本

echo "=== System Info ==="
uname -a
cat /etc/os-release | grep PRETTY_NAME

echo -e "\n=== CPU Info ==="
lscpu | grep "Model name"
lscpu | grep "CPU(s):"

echo -e "\n=== Memory ==="
free -h

echo -e "\n=== GPU Info ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "No NVIDIA GPU detected"
fi

echo -e "\n=== CUDA Info ==="
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "CUDA not found"
fi

echo -e "\n=== llama.cpp Build Info ==="
./build/bin/llama-cli --version 2>&1 || echo "llama-cli not found"

echo -e "\n=== Model Info ==="
if [ -f "$1" ]; then
    ls -lh "$1"
    python gguf-py/scripts/gguf_dump.py "$1" 2>&1 | head -30
else
    echo "Usage: $0 <model.gguf>"
fi
```

使用：
```bash
chmod +x diagnose.sh
./diagnose.sh model.gguf > diagnostic_report.txt
```

---

**相关资源**：
- [FAQ](FAQ.md)
- [快速参考](quick-reference.md)
- [GitHub Issues](https://github.com/ggml-org/llama.cpp/issues)

**提示**：90% 的问题都可以通过本指南解决。如果仍有问题，请在 GitHub 提 Issue。

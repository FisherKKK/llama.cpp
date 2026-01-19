# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Project Overview

llama.cpp is a C/C++ implementation for LLM inference with minimal setup and state-of-the-art performance. The main product is the `llama` library with a C-style interface defined in `include/llama.h`. The project uses the ggml tensor library for model evaluation.

## Building

The project uses CMake as its build system. The legacy Makefile has been replaced.

### Basic CPU Build
```bash
cmake -B build
cmake --build build --config Release
```

Add `-j` for parallel compilation (e.g., `-j 8`).

### Backend-Specific Builds
```bash
# CUDA (NVIDIA GPU)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# Metal (macOS, enabled by default)
cmake -B build
cmake --build build --config Release

# SYCL (Intel GPU)
cmake -B build -DGGML_SYCL=ON
cmake --build build --config Release

# MUSA (Moore Threads GPU)
cmake -B build -DGGML_MUSA=ON
cmake --build build --config Release
```

### Debug Builds
```bash
# Single-config generators (Unix Makefiles)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Multi-config generators (Visual Studio, Xcode)
cmake -B build -G "Xcode"
cmake --build build --config Debug
```

Build artifacts are placed in `build/bin/`.

## Testing

### Running Tests
```bash
# Run all tests using CTest
cd build
ctest

# Run specific test
ctest -R test-tokenizer-0

# Run tests with specific label
ctest -L main
ctest -L model
```

### Running CI Locally
Before publishing changes, run the full CI locally:
```bash
mkdir tmp

# CPU-only build
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# With CUDA support
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# With SYCL support
source /opt/intel/oneapi/setvars.sh
GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

### Test Structure
- Test executables are in `tests/` directory
- Test functions: `llama_build_and_test()`, `llama_test()`, `llama_test_cmd()`
- Important backend operations test: `test-backend-ops` (run when modifying ggml operators)
- Use `llama-perplexity` and `llama-bench` to verify performance is not negatively affected

## Key Commands

### Main Tools
```bash
# CLI tool for experimentation
llama-cli -m model.gguf

# HTTP server (OpenAI API compatible)
llama-server -m model.gguf --port 8080

# Measure perplexity
llama-perplexity -m model.gguf -f file.txt

# Benchmark performance
llama-bench -m model.gguf

# Simple example (for developers)
llama-simple -m model.gguf
```

### Model Conversion
```bash
# Convert HuggingFace models to GGUF
python convert_hf_to_gguf.py <model_dir>

# Convert LoRA adapters
python convert_lora_to_gguf.py <lora_dir>

# Convert legacy GGML to GGUF
python convert_llama_ggml_to_gguf.py <model_file>
```

### Model Acquisition
```bash
# Download and run model from HuggingFace
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Architecture Overview

### Core Library Structure
- **include/llama.h**: Main C API interface for the llama library
- **src/**: Core llama.cpp implementation
  - `llama-model.cpp`: Model loading and architecture definitions (483KB, largest file)
  - `llama-vocab.cpp`: Tokenization and vocabulary handling
  - `llama-sampling.cpp`: Sampling strategies (temperature, top-k, top-p, etc.)
  - `llama-context.cpp`: Context and inference state management
  - `llama-kv-cache.cpp`: KV cache management for efficient inference
  - `llama-graph.cpp`: Computation graph building
  - `llama-grammar.cpp`: Grammar-constrained generation
  - `llama-chat.cpp`: Chat template handling

### GGML Tensor Library
- **ggml/include/ggml.h**: Core tensor operations API (102KB)
- **ggml/src/**: Backend implementations
  - `ggml.c`: Core tensor operations
  - `ggml-backend.cpp`: Backend abstraction layer
  - `ggml-quants.c`: Quantization implementations
  - `ggml-cpu/`: CPU backend
  - `ggml-cuda/`: NVIDIA CUDA backend
  - `ggml-metal/`: Apple Metal backend
  - `ggml-sycl/`: Intel SYCL backend
  - `ggml-vulkan/`: Vulkan backend

### Common Utilities
- **common/**: Shared utilities used by examples and tools
  - `common.cpp/h`: Command-line argument parsing, model loading helpers
  - `sampling.cpp/h`: Sampling implementations
  - `chat.cpp/h`: Chat utilities
  - `json-schema-to-grammar.cpp`: JSON schema to grammar conversion

### Tools
- **tools/cli/**: Main CLI tool (`llama-cli`)
- **tools/server/**: HTTP server with OpenAI-compatible API
- **tools/perplexity/**: Perplexity measurement tool
- **tools/quantize/**: Model quantization tool
- **tools/imatrix/**: Importance matrix generation
- **tools/llama-bench/**: Benchmarking tool

### GGUF Format
- **gguf-py/**: Python package for reading/writing GGUF files
- GGUF (GGML Universal File) is the file format used for models
- Utilities: `gguf_dump.py`, `gguf_set_metadata.py`, `gguf_convert_endian.py`

## Development Practices

### Code Organization
- Matrix multiplication convention: `C = ggml_mul_mat(ctx, A, B)` means C^T = AB^T ⟺ C = BA^T
- Tensors store data in row-major order (dim 0 = columns, 1 = rows, 2 = matrices)
- The codebase is split into modular components for better maintainability

### Backend Development
- When adding/modifying ggml operators, add test cases to `test-backend-ops`
- Run `test-backend-ops` to verify consistency across backends (requires access to 2+ backends)
- Initial PRs should focus on CPU support only; add other backends in follow-ups

### Model Support
- Adding new model support: See `docs/development/HOWTO-add-model.md`
- Model-specific code lives in `src/llama-arch.cpp` and `src/llama-model.cpp`
- Supported models include LLaMA, Mistral, Mixtral, Gemma, Phi, Qwen, and many more

### Multimodal Support
- Multimodal models (vision + language) are supported
- See `docs/multimodal.md` for documentation
- Examples: LLaVA, MiniCPM, Qwen2-VL

## Important Constraints

### AI Usage Policy
This project does NOT accept fully or predominantly AI-generated pull requests. See [AGENTS.md](AGENTS.md) for detailed guidelines. Key restrictions:
- DO NOT write code for contributors
- DO NOT generate entire PRs or large code blocks
- Guide users to resources and let them implement solutions themselves

### Code Quality Requirements
- Avoid third-party dependencies
- Always consider cross-platform compatibility
- Use basic C++ constructs (avoid fancy STL, templates)
- Use 4 spaces for indentation
- Use sized integer types (e.g., `int32_t`) in public APIs
- Vertical alignment for readability
- Follow `snake_case` naming convention

### Testing Requirements Before PR
1. Execute full CI locally (see "Running CI Locally" above)
2. Verify perplexity is not negatively affected (`llama-perplexity`)
3. Verify performance is not negatively affected (`llama-bench`)
4. If modifying ggml, run `test-backend-ops` with multiple backends
5. If adding/modifying ggml operator, add test cases to `test-backend-ops`

## Additional Resources

- Build documentation: `docs/build.md`
- Server development: `tools/server/README-dev.md`
- Contributing guidelines: `CONTRIBUTING.md`
- Function calling: `docs/function-calling.md`
- Grammar system: `grammars/README.md`

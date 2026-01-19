# Day 13: 采样策略与生成控制

## 课程目标

掌握 LLM 文本生成的核心技术：
- 采样算法原理与实现
- Temperature、Top-K、Top-P 详解
- Mirostat 自适应采样
- 重复惩罚机制
- 实战：定制采样策略

## 1. 采样问题概述

### 1.1 从 Logits 到 Token

```
模型输出 logits: [n_vocab] 个原始得分
    ↓
如何选择下一个 token？

策略1: Greedy（贪心）
   → 总是选最高分的
   → 输出确定但单调

策略2: 随机采样
   → 按概率随机选择
   → 输出多样但可能混乱

策略3: 智能采样 ⭐
   → 平衡质量与多样性
   → 本课程重点
```

## 2. 核心采样算法

### 2.1 Temperature Scaling（温度缩放）

**原理**：调整概率分布的"尖锐度"

```python
# 伪代码
logits_scaled = logits / temperature

if temperature < 1.0:
    # 更确定性（概率集中）
    # 高分 token 概率进一步提升
    probs = softmax([10, 8, 2]) @ T=0.5
        = [0.9, 0.09, 0.01]  # 更集中

elif temperature > 1.0:
    # 更随机（概率平缓）
    # 低分 token 也有机会
    probs = softmax([10, 8, 2]) @ T=2.0
        = [0.6, 0.35, 0.05]  # 更平均

elif temperature == 1.0:
    # 原始分布
    probs = softmax([10, 8, 2])
        = [0.73, 0.24, 0.03]
```

**代码实现**（位置：`src/llama-sampling.cpp:478`）：

```cpp
void llama_sampler_sample_temp(
    struct llama_token_data_array * cur_p,
    float temp) {

    if (temp <= 0.0f) {
        // Temperature = 0 等同于 greedy
        cur_p->selected = cur_p->data[0].id;
        return;
    }

    // 对所有 logits 缩放
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].logit /= temp;
    }
}
```

**效果对比**：
- `temp=0.0`: 完全确定，适合代码生成、数学推理
- `temp=0.7`: 略有创意，适合一般对话
- `temp=1.0`: 原始分布
- `temp=1.5`: 很有创意，适合创意写作
- `temp=2.0+`: 非常随机，可能产生无意义输出

### 2.2 Top-K Sampling

**原理**：只保留概率最高的 K 个 token

```python
# 伪代码
def top_k_sampling(logits, k=40):
    # 1. 排序
    sorted_indices = argsort(logits, descending=True)

    # 2. 只保留前 k 个
    keep_indices = sorted_indices[:k]

    # 3. 其他设为 -inf（概率为0）
    logits[~keep_indices] = -float('inf')

    # 4. 重新归一化并采样
    probs = softmax(logits)
    token = sample(probs)
    return token
```

**代码实现**（位置：`src/llama-sampling.cpp:548`）：

```cpp
void llama_sampler_sample_top_k(
    struct llama_token_data_array * cur_p,
    int32_t k) {

    if (k <= 0 || k >= (int32_t) cur_p->size) {
        return;  // 不过滤
    }

    // 1. 部分排序，找到第 k 大的元素
    std::nth_element(
        cur_p->data,
        cur_p->data + k,
        cur_p->data + cur_p->size,
        [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        });

    // 2. 截断到前 k 个
    cur_p->size = k;
}
```

**适用场景**：
- `k=1`: 等同于 greedy
- `k=10`: 非常保守
- `k=40`: 常用默认值
- `k=100+`: 几乎不过滤

### 2.3 Top-P (Nucleus) Sampling ⭐

**原理**：动态选择累计概率达到 P 的最小 token 集合

```python
# 伪代码
def top_p_sampling(logits, p=0.9):
    # 1. 计算概率并排序
    probs = softmax(logits)
    sorted_probs, sorted_indices = sort(probs, descending=True)

    # 2. 计算累计概率
    cumsum_probs = cumsum(sorted_probs)

    # 3. 找到累计概率刚超过 p 的位置
    cutoff_index = find_first(cumsum_probs > p)

    # 4. 保留前 cutoff_index 个 token
    keep_indices = sorted_indices[:cutoff_index]
    probs[~keep_indices] = 0

    # 5. 重新归一化并采样
    probs = probs / sum(probs)
    token = sample(probs)
    return token
```

**示例**：
```
原始 logits: [10.0, 9.8, 7.0, 6.5, 3.0, 2.5, ...]
    ↓ softmax
概率:        [0.53, 0.25, 0.08, 0.06, 0.02, 0.01, ...]
    ↓ 累计
累计概率:    [0.53, 0.78, 0.86, 0.92, 0.94, 0.95, ...]
             ^     ^     ^     ^
             |     |     |     +-- 超过 p=0.9，截断这里

保留前 4 个 token: [10.0, 9.8, 7.0, 6.5]
忽略其余 token
```

**代码实现**（位置：`src/llama-sampling.cpp:590`）：

```cpp
void llama_sampler_sample_top_p(
    struct llama_token_data_array * cur_p,
    float p) {

    if (p >= 1.0f) {
        return;  // 不过滤
    }

    // 1. 先 softmax
    llama_sampler_softmax_impl(cur_p);

    // 2. 按概率降序排序
    std::sort(cur_p->data, cur_p->data + cur_p->size,
        [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

    // 3. 计算累计概率
    float cum_sum = 0.0f;
    size_t last_idx = 0;

    for (size_t i = 0; i < cur_p->size; ++i) {
        cum_sum += cur_p->data[i].p;
        last_idx = i;

        if (cum_sum >= p) {
            break;
        }
    }

    // 4. 截断
    cur_p->size = last_idx + 1;
}
```

**Top-P vs Top-K**：
- Top-K: 固定数量，可能太保守或太宽松
- Top-P: 自适应，始终保留"足够好"的 token 集合

**推荐配置**：
- 对话：`p=0.9, temp=0.7`
- 创意写作：`p=0.95, temp=1.2`
- 代码生成：`p=0.5, temp=0.2`

### 2.4 Min-P Sampling

**原理**：过滤掉概率低于 `max_prob * min_p` 的 token

```cpp
// 位置：src/llama-sampling.cpp:653
void llama_sampler_sample_min_p(
    struct llama_token_data_array * cur_p,
    float min_p) {

    if (min_p <= 0.0f || cur_p->size == 0) {
        return;
    }

    // 先 softmax
    llama_sampler_softmax_impl(cur_p);

    // 找最大概率
    float max_prob = cur_p->data[0].p;  // 已排序

    // 计算阈值
    float threshold = min_p * max_prob;

    // 过滤
    size_t keep_count = 0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].p >= threshold) {
            cur_p->data[keep_count++] = cur_p->data[i];
        }
    }

    cur_p->size = keep_count;
}
```

**用途**：配合 Top-P 使用，进一步过滤低质量 token。

### 2.5 Mirostat ⭐ 高级算法

**原理**：自适应调整温度，控制输出的"惊喜度"（perplexity）

Mirostat 试图维持目标 perplexity（困惑度），动态调整采样参数。

```cpp
// 位置：src/llama-sampling.cpp:747
struct llama_sampler_mirostat_v2 {
    float tau;        // 目标 perplexity (通常 5.0)
    float eta;        // 学习率 (通常 0.1)
    float mu;         // 当前状态（动态调整）
};

void llama_sampler_sample_mirostat_v2(
    struct llama_sampler_mirostat_v2 * mirostat,
    struct llama_token_data_array * cur_p) {

    // 1. Softmax
    llama_sampler_softmax_impl(cur_p);

    // 2. 计算当前 entropy
    float entropy = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        float p = cur_p->data[i].p;
        if (p > 0) {
            entropy -= p * logf(p);
        }
    }

    // 3. 计算误差
    float error = entropy - logf(mirostat->tau);

    // 4. 更新 mu（软阈值）
    mirostat->mu -= mirostat->eta * error;

    // 5. 根据 mu 过滤 token
    // 保留 log(prob) >= -mu 的 token
    size_t keep_count = 0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (-logf(cur_p->data[i].p) <= mirostat->mu) {
            cur_p->data[keep_count++] = cur_p->data[i];
        }
    }
    cur_p->size = keep_count;

    // 6. 重新归一化并采样
    llama_sampler_softmax_impl(cur_p);
    // ... sample
}
```

**Mirostat 优势**：
- 自动平衡质量与多样性
- 减少重复和发散
- 适合长文本生成

**推荐参数**：
- `tau=5.0`: 目标困惑度
- `eta=0.1`: 调整速度

## 3. 重复惩罚

### 3.1 简单重复惩罚

**原理**：降低已出现 token 的概率

```cpp
// 位置：src/llama-sampling.cpp:398
void llama_sampler_sample_repetition_penalty(
    struct llama_token_data_array * cur_p,
    const llama_token * last_tokens,
    size_t last_tokens_size,
    float penalty) {  // 通常 1.1 - 1.5

    if (penalty == 1.0f) {
        return;  // 无惩罚
    }

    // 构建已出现 token 的集合
    std::unordered_set<llama_token> token_set(
        last_tokens, last_tokens + last_tokens_size);

    // 应用惩罚
    for (size_t i = 0; i < cur_p->size; ++i) {
        llama_token token = cur_p->data[i].id;

        if (token_set.count(token)) {
            if (cur_p->data[i].logit >= 0) {
                // 正 logit 除以 penalty
                cur_p->data[i].logit /= penalty;
            } else {
                // 负 logit 乘以 penalty
                cur_p->data[i].logit *= penalty;
            }
        }
    }
}
```

**效果**：
- `penalty=1.0`: 无惩罚
- `penalty=1.1`: 轻微惩罚，适合对话
- `penalty=1.3`: 中等惩罚，减少重复
- `penalty=1.5+`: 强惩罚，可能影响连贯性

### 3.2 频率/出现惩罚（OpenAI 风格）

```cpp
void llama_sampler_sample_frequency_presence_penalty(
    struct llama_token_data_array * cur_p,
    const std::map<llama_token, int> & token_count,
    float alpha_frequency,   // 频率惩罚系数
    float alpha_presence) {  // 出现惩罚系数

    for (size_t i = 0; i < cur_p->size; ++i) {
        llama_token token = cur_p->data[i].id;

        auto it = token_count.find(token);
        if (it != token_count.end()) {
            int count = it->second;

            // 惩罚 = 频率惩罚 * count + 出现惩罚
            float penalty = alpha_frequency * count + alpha_presence;
            cur_p->data[i].logit -= penalty;
        }
    }
}
```

**参数说明**：
- `alpha_frequency`: 与出现次数成正比（抑制高频词）
- `alpha_presence`: 固定惩罚（抑制已出现的词）

**推荐配置**：
- 对话：`freq=0.5, presence=0.5`
- 创作：`freq=0.3, presence=0.3`

## 4. 采样链（Sampler Chain）

llama.cpp 使用"采样链"组合多个采样器：

```cpp
// 位置：src/llama-sampling.cpp:1234
struct llama_sampler * llama_sampler_chain_init(
    struct llama_sampler_chain_params params) {

    auto * chain = new llama_sampler_chain();

    // 1. 重复惩罚（作用于 logits）
    llama_sampler_chain_add(chain,
        llama_sampler_init_repetition_penalty(params.penalty_repeat));

    // 2. 频率/出现惩罚
    llama_sampler_chain_add(chain,
        llama_sampler_init_frequencies(
            params.penalty_freq, params.penalty_present));

    // 3. Top-K
    llama_sampler_chain_add(chain,
        llama_sampler_init_top_k(params.top_k));

    // 4. Top-P
    llama_sampler_chain_add(chain,
        llama_sampler_init_top_p(params.top_p));

    // 5. Min-P
    llama_sampler_chain_add(chain,
        llama_sampler_init_min_p(params.min_p));

    // 6. Temperature
    llama_sampler_chain_add(chain,
        llama_sampler_init_temp(params.temp));

    // 7. Mirostat（可选）
    if (params.mirostat == 2) {
        llama_sampler_chain_add(chain,
            llama_sampler_init_mirostat_v2(params.mirostat_tau, params.mirostat_eta));
    }

    return chain;
}
```

**执行顺序**：
```
logits
  ↓
重复惩罚 (修改 logits)
  ↓
频率惩罚 (修改 logits)
  ↓
Top-K (过滤)
  ↓
Top-P (过滤)
  ↓
Min-P (过滤)
  ↓
Temperature (缩放)
  ↓
Softmax (归一化)
  ↓
Mirostat (动态过滤，可选)
  ↓
采样 (multinomial/greedy)
  ↓
selected token
```

## 5. 实战：定制采样策略

### 5.1 代码生成采样器

```cpp
struct llama_sampler * create_code_sampler() {
    auto params = llama_sampler_chain_default_params();

    // 代码需要确定性
    params.temp = 0.2;              // 低温度
    params.top_p = 0.5;             // 严格过滤
    params.top_k = 20;              // 少量候选
    params.penalty_repeat = 1.05;   // 轻微重复惩罚
    params.penalty_freq = 0.0;      // 不惩罚常用关键字
    params.penalty_present = 0.0;

    return llama_sampler_chain_init(params);
}
```

### 5.2 创意写作采样器

```cpp
struct llama_sampler * create_creative_sampler() {
    auto params = llama_sampler_chain_default_params();

    // 创意需要多样性
    params.temp = 1.3;              // 高温度
    params.top_p = 0.95;            // 宽松过滤
    params.top_k = 100;             // 大量候选
    params.penalty_repeat = 1.2;    // 中等重复惩罚
    params.penalty_freq = 0.5;      // 避免高频词
    params.penalty_present = 0.3;

    return llama_sampler_chain_init(params);
}
```

### 5.3 Mirostat 采样器

```cpp
struct llama_sampler * create_mirostat_sampler() {
    auto params = llama_sampler_chain_default_params();

    // 使用 Mirostat 自适应
    params.mirostat = 2;
    params.mirostat_tau = 5.0;      // 目标困惑度
    params.mirostat_eta = 0.1;      // 学习率
    params.temp = 1.0;              // Mirostat 会自动调整

    // 禁用其他采样器（Mirostat 已包含）
    params.top_k = 0;
    params.top_p = 1.0;

    return llama_sampler_chain_init(params);
}
```

## 6. 高级技巧

### 6.1 动态调整采样参数

```cpp
// 根据生成长度动态调整
float get_temperature_for_length(int current_len, int max_len) {
    // 开始时保守，后期更自由
    float progress = (float)current_len / max_len;

    if (progress < 0.3) {
        return 0.7;  // 前30%较保守
    } else if (progress < 0.7) {
        return 1.0;  // 中间正常
    } else {
        return 1.2;  // 最后更自由
    }
}
```

### 6.2 基于困惑度的早停

```cpp
bool should_stop_generation(
    const std::vector<float> & recent_probs) {

    // 计算最近几个 token 的平均概率
    float avg_prob = std::accumulate(
        recent_probs.begin(), recent_probs.end(), 0.0f)
        / recent_probs.size();

    // 如果连续多个 token 概率很低，可能是模型"迷失"了
    if (avg_prob < 0.01f) {
        return true;  // 早停
    }

    return false;
}
```

### 6.3 多样性波束搜索（Diverse Beam Search）

```cpp
struct beam {
    std::vector<llama_token> tokens;
    float score;
};

std::vector<beam> diverse_beam_search(
    llama_context * ctx,
    int n_beams = 5,
    float diversity_penalty = 0.5) {

    std::vector<beam> beams(n_beams);

    // 初始化
    for (int i = 0; i < n_beams; i++) {
        beams[i].score = 0.0f;
    }

    for (int step = 0; step < max_len; step++) {
        // 对每个 beam 生成候选
        for (int i = 0; i < n_beams; i++) {
            auto logits = llama_get_logits(ctx);

            // 惩罚其他 beam 已选择的 token
            for (int j = 0; j < i; j++) {
                llama_token prev_token = beams[j].tokens.back();
                logits[prev_token] -= diversity_penalty;
            }

            // 采样
            llama_token next = sample_top_p(logits, 0.9);
            beams[i].tokens.push_back(next);
            beams[i].score += logf(get_prob(logits, next));
        }
    }

    return beams;
}
```

## 7. 总结

今天我们深入学习了采样策略：

✅ **Temperature**: 控制确定性 vs 随机性
✅ **Top-K/Top-P**: 过滤低质量 token
✅ **Mirostat**: 自适应控制困惑度
✅ **重复惩罚**: 避免无聊的重复
✅ **采样链**: 组合多种策略

### 推荐配置速查

| 场景 | Temperature | Top-P | Top-K | Repeat Penalty |
|------|-------------|-------|-------|----------------|
| **对话** | 0.7 | 0.9 | 40 | 1.1 |
| **代码** | 0.2 | 0.5 | 20 | 1.05 |
| **创作** | 1.3 | 0.95 | 100 | 1.2 |
| **翻译** | 0.3 | 0.7 | 30 | 1.0 |
| **摘要** | 0.5 | 0.8 | 50 | 1.15 |

## 下一步

明天我们将学习 **Day 14: 实战项目与总结**：
- 从零搭建推理服务
- llama-server 源码解析
- OpenAI API 兼容实现
- 课程总结与进阶路线

---

**练习**：
1. 实现一个采样器，结合 Top-K 和 Mirostat
2. 分析不同 temperature 对输出的影响
3. 设计适合你的应用场景的采样配置

📚 [Day 14: 实战项目与总结](day14-tools-practice.md)

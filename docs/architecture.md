# LM Lab 项目文档

## 概览

这是一个从零实现语言模型的教学项目，基于斯坦福 CS336 课程。使用纯 NumPy 实现所有核心组件，深入理解 Transformer 架构、优化算法和训练流程。

## 项目结构

```
lm-lab/
├── src/lm_lab/                    # 核心实现
│   ├── tokenization/              # 分词器（已完成）
│   │   ├── byte_tokenizer.py      # UTF-8 字节分词器
│   │   └── bpe.py                 # BPE 分词算法
│   ├── model/                     # Transformer 模型
│   │   ├── attention.py           # 因果自注意力机制
│   │   ├── layer_norm.py          # 层归一化
│   │   ├── feedforward.py         # 前馈网络
│   │   ├── embedding.py           # 位置编码
│   │   └── transformer.py         # 完整 Transformer 模型
│   ├── optimizer/                 # 优化器
│   │   ├── sgd.py                 # SGD with momentum
│   │   ├── adam.py                # Adam/AdamW
│   │   └── gradient.py            # 梯度裁剪
│   ├── training/                  # 训练流程
│   │   ├── trainer.py             # 训练器
│   │   └── dataloader.py          # 数据加载
│   └── utils/                     # 工具函数
│       ├── numerical.py           # 数值稳定性
│       ├── shapes.py              # 形状断言
│       └── seed.py                # 随机种子
├── tests/                         # 测试
│   ├── tokenization/              # 分词器测试（71 tests passed）
│   └── model/                     # 模型测试
├── docs/                          # 文档
│   ├── roadmap.md                 # 学习路线图
│   ├── tokenizer.md               # 分词器任务说明
│   ├── architecture.md            # 架构设计文档（本文档）
│   ├── api_reference.md           # API 参考
│   └── training_guide.md          # 训练指南
├── scripts/                       # 工具脚本
│   ├── check_init.py              # 初始化检查
│   └── measure_tokenizer.py       # 性能测量
└── runs/                          # 实验记录
```

## 已完成的里程碑

### ✅ B1: BPE Tokenizer（已验证）

完整实现了 5 个阶段：

- **T1**: 字节分词器 - UTF-8 编码/解码
- **T2**: BPE 训练 - 频率统计、确定性合并
- **T3**: 编码解码 - Rank 优先级编码
- **T4**: 预分词 - 无损分割、特殊 token 处理
- **T5**: 持久化 - JSON 序列化、确定性保存

**测试结果**: 71/71 tests passed

### 🚧 B2: Transformer 模型（开发中）

已实现的组件：

#### 1. 因果自注意力（CausalSelfAttention）

- 多头注意力机制
- 因果掩码（防止关注未来 token）
- 缩放点积注意力
- 完整的前向和反向传播

**核心公式**：
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

**关键特性**：
- 自动梯度计算
- 数值稳定的 softmax
- 因果掩码确保 token i 只能关注 ≤ i 的位置

#### 2. 层归一化（LayerNorm）

- 跨特征维度归一化
- 学习的缩放和偏移参数
- 提高训练稳定性

**公式**：
```
LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
```

#### 3. 前馈网络（FeedForward）

- 两层 MLP
- GELU 激活函数
- 典型扩展比例：4x

**公式**：
```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

#### 4. 位置编码（Positional Encoding）

实现两种方式：

**正弦位置编码**（可外推）：
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**学习位置编码**：
- 每个位置的可训练嵌入
- 更灵活但不能外推

#### 5. Transformer Block

Pre-LN 架构：
```python
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

#### 6. 完整语言模型（TransformerLM）

组件：
- Token 嵌入
- 位置编码
- N 个 Transformer blocks
- 最终层归一化
- 输出投影（与输入嵌入绑定）

**特性**：
- 权重绑定（减少参数）
- 前向和反向传播
- 参数/梯度管理

### 🚧 B3: 优化器（开发中）

已实现：

#### 1. SGD with Momentum

```python
v_t = μ * v_{t-1} + lr * g_t
θ_t = θ_{t-1} - v_t
```

支持：
- 标准动量
- Nesterov 动量
- 权重衰减

#### 2. Adam

```python
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
θ_t = θ_{t-1} - α * m̂_t / (sqrt(v̂_t) + ε)
```

**特性**：
- 自适应学习率
- 偏差校正
- 一阶和二阶动量

#### 3. AdamW

解耦权重衰减：
```python
θ_t = θ_{t-1} - α * m̂_t / (sqrt(v̂_t) + ε) - α * λ * θ_{t-1}
```

#### 4. 梯度裁剪

- `clip_grad_norm`: 全局范数裁剪
- `clip_grad_value`: 值裁剪

### ⏳ B4: 训练流程（待实现）

需要实现：
- 数据加载器
- 训练循环
- 检查点管理
- 采样策略
- 评估工具

## 技术亮点

### 1. 纯 NumPy 实现

所有组件使用纯 NumPy 实现，无框架依赖：
- 手动实现前向传播
- 手动推导反向传播
- 完整梯度计算

**教育价值**：深入理解每个操作的数学原理

### 2. 数值稳定性

关键技术：
- 稳定的 softmax（减去最大值）
- 梯度裁剪（防止爆炸）
- 偏差校正（Adam）
- 有限值检查

### 3. 因果注意力掩码

确保自回归特性：
```python
mask = np.tril(np.ones((seq_len, seq_len)))
scores = np.where(mask == 1, scores, -1e10)
```

Token i 只能看到位置 0 到 i 的信息。

### 4. 梯度传播

完整实现所有层的反向传播：
- 注意力机制的梯度
- LayerNorm 的梯度
- 残差连接的梯度
- 嵌入层的梯度

## 维度约定

整个项目使用一致的张量维度：

```python
# 输入
input_ids: (batch_size, seq_len)

# 嵌入
embeddings: (batch_size, seq_len, d_model)

# 注意力
Q, K, V: (batch_size, num_heads, seq_len, d_k)
scores: (batch_size, num_heads, seq_len, seq_len)

# 前馈
hidden: (batch_size, seq_len, d_ff)

# 输出
logits: (batch_size, seq_len, vocab_size)
```

## 设计原则

### 1. 教学优先

- 清晰的代码结构
- 详细的注释
- 逐步构建复杂性

### 2. 正确性第一

- 完整的测试覆盖
- 梯度检查
- 数值稳定性验证

### 3. 模块化设计

- 独立的组件
- 清晰的接口
- 易于测试和调试

## 性能考虑

当前实现优先考虑清晰性而非性能：

- ✅ 正确的算法实现
- ✅ 数值稳定性
- ⚠️ 未优化的 NumPy 操作
- ❌ 无 GPU 加速

**后续优化方向**（S1-S3）：
1. Profiling 识别瓶颈
2. 向量化操作
3. GPU/Triton 实现
4. 分布式训练

## 下一步计划

### 短期（1-2 周）

1. **完成 B3 验证**
   - 优化器单元测试
   - 梯度检查
   - 简单函数优化验证

2. **实现 B4 训练流程**
   - 数据加载器
   - 训练循环
   - 检查点管理
   - 小批量过拟合验证

### 中期（1 个月）

3. **模型训练实验**
   - 小模型（10M 参数）
   - 小数据集（10M tokens）
   - 验证收敛性

4. **文档完善**
   - API 参考
   - 训练指南
   - 故障排查

### 长期（2-3 个月）

5. **性能优化（S1-S3）**
   - GPU 加速
   - Flash Attention
   - 分布式训练

6. **扩展实验（C1-C2）**
   - Scaling laws
   - 模型规模实验

## 学习资源

### 论文

1. **Attention is All You Need** (Vaswani et al., 2017)
   - Transformer 原始论文
   - 位置编码、多头注意力

2. **Language Models are Unsupervised Multitask Learners** (GPT-2)
   - 自回归语言建模
   - BPE 分词

3. **Decoupled Weight Decay Regularization** (Loshchilov & Hutter, 2017)
   - AdamW 优化器

### 课程

- **CS336: Language Modeling from Scratch** (Stanford)
- **CS224N: Natural Language Processing** (Stanford)
- **The Illustrated Transformer** (Jay Alammar)

## 贡献指南

见 [AGENTS.md](../AGENTS.md)

## 许可证

MIT License

---

**最后更新**: 2026-10-07  
**项目状态**: B2 开发中，B1 已完成

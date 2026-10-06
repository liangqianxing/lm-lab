# LM Lab 快速开始指南

本指南帮助您快速上手 LM Lab 项目，从安装到训练第一个语言模型。

## 目录

1. [安装](#安装)
2. [验证安装](#验证安装)
3. [快速示例](#快速示例)
4. [训练第一个模型](#训练第一个模型)
5. [常见问题](#常见问题)

---

## 安装

### 系统要求

- Python 3.11 或更高版本
- 至少 8GB RAM
- 推荐：CUDA GPU（可选，用于后续加速）

### 安装步骤

```bash
# 克隆仓库
cd C:\Users\liangqianxing\Desktop\llm

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境（Windows）
.\.venv\Scripts\activate

# 安装项目（开发模式）
pip install -e ".[dev]"
```

---

## 验证安装

### 1. 运行初始化检查

```bash
python scripts/check_init.py
```

应该看到：
```
✓ 项目结构正确
✓ Python 版本 >= 3.11
✓ 所有模块可导入
```

### 2. 运行分词器测试

```bash
# 测试所有 tokenizer 功能（71 tests）
pytest tests/tokenization/ -v

# 或者按阶段测试
pytest -m t1  # 字节分词器
pytest -m t2  # BPE 训练
pytest -m t3  # 编码解码
pytest -m t4  # 预分词
pytest -m t5  # 持久化
```

**预期结果**：`71 passed`

---

## 快速示例

### 示例 1: 字节分词器

```python
from lm_lab.tokenization import ByteTokenizer

# 创建分词器
tokenizer = ByteTokenizer()

# 编码
text = "Hello 世界! 🌍"
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")
# [72, 101, 108, 108, 111, 32, 228, 184, 150, 231, 149, 140, 33, 32, 240, 159, 140, 141]

# 解码
decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")
# "Hello 世界! 🌍"
```

### 示例 2: BPE 分词器

```python
from lm_lab.tokenization import train_bpe, BPETokenizer

# 准备训练数据
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "这是一个中文句子。",
    "Machine learning is awesome!",
    "Natural language processing 🚀"
]

# 训练 BPE 模型
model = train_bpe(
    texts=texts,
    vocab_size=300,
    pretokenization="simple",
    special_tokens=("<s>", "</s>", "<pad>")
)

# 创建分词器
tokenizer = BPETokenizer(model)

# 使用分词器
text = "<s>Hello world</s>"
token_ids = tokenizer.encode(text, allowed_special={"<s>", "</s>"})
print(f"Tokens: {token_ids}")

decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")

# 保存模型
from pathlib import Path
tokenizer.save(Path("my_tokenizer.json"))

# 加载模型
loaded_tokenizer = BPETokenizer.load(Path("my_tokenizer.json"))
```

### 示例 3: Transformer 模型

```python
import numpy as np
from lm_lab.model import TransformerLM

# 创建模型
model = TransformerLM(
    vocab_size=1000,
    d_model=256,
    num_heads=8,
    num_layers=4,
    d_ff=1024,
    max_seq_len=512
)

# 前向传播
batch_size = 2
seq_len = 10
input_ids = np.random.randint(0, 1000, size=(batch_size, seq_len))

logits = model.forward(input_ids)
print(f"Logits shape: {logits.shape}")  # (2, 10, 1000)

# 计算损失（示例）
targets = np.random.randint(0, 1000, size=(batch_size, seq_len))

# 交叉熵损失
def cross_entropy_loss(logits, targets):
    batch_size, seq_len, vocab_size = logits.shape
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # 负对数似然
    loss = 0.0
    for b in range(batch_size):
        for t in range(seq_len):
            loss -= np.log(probs[b, t, targets[b, t]] + 1e-10)
    
    return loss / (batch_size * seq_len)

loss = cross_entropy_loss(logits, targets)
print(f"Loss: {loss:.4f}")
```

### 示例 4: 优化器

```python
from lm_lab.optimizer import Adam, AdamW, SGD

# Adam 优化器
optimizer = Adam(learning_rate=3e-4, beta1=0.9, beta2=0.999)

# AdamW（推荐用于 Transformer）
optimizer = AdamW(learning_rate=3e-4, weight_decay=0.01)

# SGD with momentum
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# 使用优化器
params = model.parameters()  # 获取所有参数
grads = model.gradients()    # 获取所有梯度（需要先 backward）

# 转换为字典格式
param_dict = {f'param_{i}': p for i, p in enumerate(params)}
grad_dict = {f'param_{i}': g for i, g in enumerate(grads)}

# 执行优化步骤
optimizer.step(param_dict, grad_dict)
```

---

## 训练第一个模型

### 完整训练示例

```python
import numpy as np
from pathlib import Path
from lm_lab.tokenization import train_bpe, BPETokenizer
from lm_lab.model import TransformerLM
from lm_lab.optimizer import AdamW
from lm_lab.utils import set_seed

# 1. 设置随机种子
set_seed(42)

# 2. 准备数据
texts = [
    "The cat sat on the mat.",
    "The dog played in the park.",
    "Machine learning is fun.",
    # ... 更多文本
]

# 3. 训练分词器
print("Training tokenizer...")
model = train_bpe(
    texts=texts,
    vocab_size=500,
    pretokenization="simple",
    special_tokens=("<s>", "</s>", "<pad>")
)
tokenizer = BPETokenizer(model)

# 4. 准备训练数据
print("Preparing training data...")
def prepare_batch(texts, tokenizer, seq_len=128):
    """准备一个批次的数据"""
    batch_inputs = []
    batch_targets = []
    
    for text in texts:
        # 编码文本
        token_ids = tokenizer.encode(text, allowed_special={"<s>", "</s>"})
        
        # 截断或填充到固定长度
        if len(token_ids) > seq_len:
            token_ids = token_ids[:seq_len]
        else:
            pad_id = tokenizer.model.special_tokens["<pad>"]
            token_ids = token_ids + [pad_id] * (seq_len - len(token_ids))
        
        # 输入和目标（自回归：预测下一个 token）
        batch_inputs.append(token_ids[:-1])
        batch_targets.append(token_ids[1:])
    
    return np.array(batch_inputs), np.array(batch_targets)

# 5. 创建模型
print("Creating model...")
model = TransformerLM(
    vocab_size=len(tokenizer.model.vocab),
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=512,
    max_seq_len=128
)

print(f"Model parameters: {sum(p.size for p in model.parameters()):,}")

# 6. 创建优化器
optimizer = AdamW(learning_rate=3e-4, weight_decay=0.01)

# 7. 训练循环
print("\nStarting training...")
num_epochs = 10
batch_size = 8

for epoch in range(num_epochs):
    # 准备批次
    batch_texts = np.random.choice(texts, size=batch_size, replace=True)
    inputs, targets = prepare_batch(batch_texts.tolist(), tokenizer)
    
    # 前向传播
    logits = model.forward(inputs)
    
    # 计算损失
    batch_size, seq_len, vocab_size = logits.shape
    
    # 损失梯度
    grad_logits = np.zeros_like(logits)
    total_loss = 0.0
    
    for b in range(batch_size):
        for t in range(seq_len):
            # Softmax
            exp_logits = np.exp(logits[b, t] - np.max(logits[b, t]))
            probs = exp_logits / np.sum(exp_logits)
            
            # 负对数似然
            target = targets[b, t]
            total_loss -= np.log(probs[target] + 1e-10)
            
            # 梯度
            grad_logits[b, t] = probs
            grad_logits[b, t, target] -= 1.0
    
    loss = total_loss / (batch_size * seq_len)
    grad_logits /= (batch_size * seq_len)
    
    # 反向传播
    model.backward(grad_logits)
    
    # 优化步骤
    params = model.parameters()
    grads = model.gradients()
    
    param_dict = {f'p{i}': p for i, p in enumerate(params)}
    grad_dict = {f'p{i}': g for i, g in enumerate(grads) if g is not None}
    
    optimizer.step(param_dict, grad_dict)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

print("\nTraining complete!")

# 8. 保存模型（简单版本）
print("Saving model...")
np.savez('model_checkpoint.npz', 
         token_embeddings=model.token_embeddings,
         **{f'block_{i}_param_{j}': p 
            for i, block in enumerate(model.blocks) 
            for j, p in enumerate(block.parameters())})

print("Model saved to model_checkpoint.npz")
```

### 预期输出

```
Training tokenizer...
Preparing training data...
Creating model...
Model parameters: 234,752

Starting training...
Epoch 1/10, Loss: 6.2145
Epoch 2/10, Loss: 5.8923
Epoch 3/10, Loss: 5.5678
...
Epoch 10/10, Loss: 3.2341

Training complete!
Saving model...
Model saved to model_checkpoint.npz
```

---

## 常见问题

### Q1: 导入错误 `ModuleNotFoundError: No module named 'lm_lab'`

**解决方法**:
```bash
# 确保在项目根目录
cd C:\Users\liangqianxing\Desktop\llm

# 重新安装
pip install -e .
```

### Q2: BPE 训练时出现 `ValueError: vocab_size too small`

**原因**: `vocab_size` 必须 >= 256 + 特殊 token 数量

**解决方法**:
```python
# 错误
train_bpe(texts, vocab_size=200, special_tokens=("<s>", "</s>"))

# 正确
train_bpe(texts, vocab_size=300, special_tokens=("<s>", "</s>"))
```

### Q3: 训练时出现 NaN 损失

**可能原因**:
1. 学习率过大
2. 梯度爆炸
3. 数值不稳定

**解决方法**:
```python
# 1. 降低学习率
optimizer = AdamW(learning_rate=1e-4)  # 而不是 1e-3

# 2. 使用梯度裁剪
from lm_lab.optimizer import clip_grad_norm

grads = model.gradients()
clip_grad_norm(grads, max_norm=1.0)

# 3. 检查数值
from lm_lab.utils import check_finite

for p in model.parameters():
    check_finite(p, "param")
```

### Q4: 内存不足

**解决方法**:
```python
# 减小批次大小
batch_size = 4  # 而不是 32

# 减小模型大小
model = TransformerLM(
    d_model=128,   # 而不是 512
    num_layers=2,  # 而不是 12
    ...
)

# 减小序列长度
max_seq_len=128  # 而不是 1024
```

### Q5: 测试失败

```bash
# 查看详细错误信息
pytest tests/ -v

# 只运行特定测试
pytest tests/tokenization/test_bpe_training.py::test_simple_training -v

# 查看打印输出
pytest tests/ -v -s
```

---

## 下一步

完成快速开始后，您可以：

1. **阅读架构文档**: [architecture.md](architecture.md)
2. **查看 API 参考**: [api_reference.md](api_reference.md)
3. **学习路线图**: [roadmap.md](roadmap.md)
4. **实验记录**: 在 `runs/` 目录创建实验记录

---

## 获取帮助

- **文档**: `docs/` 目录
- **示例**: `scripts/` 目录
- **测试**: `tests/` 目录作为使用示例

---

**祝学习愉快！** 🚀

最后更新: 2026-10-07

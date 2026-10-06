# API 参考文档

本文档提供 LM Lab 项目所有模块的详细 API 参考。

## 目录

- [Tokenization](#tokenization)
- [Model](#model)
- [Optimizer](#optimizer)
- [Training](#training)
- [Utils](#utils)

---

## Tokenization

### ByteTokenizer

最简单的分词器，直接使用 UTF-8 字节。

```python
from lm_lab.tokenization import ByteTokenizer

tokenizer = ByteTokenizer()
```

#### Methods

**`encode(text: str) -> list[int]`**

将文本编码为字节 ID 列表。

- **参数**:
  - `text`: 输入字符串
- **返回**: 字节 ID 列表 (0-255)
- **示例**:
  ```python
  ids = tokenizer.encode("Hello 世界")
  # [72, 101, 108, 108, 111, 32, 228, 184, 150, 231, 149, 140]
  ```

**`decode(token_ids: Sequence[int], *, errors: Literal["strict", "replace"] = "strict") -> str`**

将字节 ID 解码为文本。

- **参数**:
  - `token_ids`: 字节 ID 序列
  - `errors`: 错误处理模式
    - `"strict"`: 遇到非法字节抛出异常
    - `"replace"`: 用 � 替换非法字节
- **返回**: 解码后的字符串
- **异常**:
  - `ValueError`: token ID 超出 0-255 范围
  - `UnicodeDecodeError`: 非法 UTF-8 序列（strict 模式）

---

### BPE

Byte Pair Encoding 分词器。

```python
from lm_lab.tokenization import train_bpe, BPETokenizer

# 训练
model = train_bpe(
    texts=["Hello world", "你好世界"],
    vocab_size=300,
    pretokenization="simple",
    special_tokens=("<eos>", "<pad>")
)

# 使用
tokenizer = BPETokenizer(model)
```

#### train_bpe

**`train_bpe(texts, vocab_size, *, pretokenization="none", special_tokens=()) -> BPEModel`**

训练 BPE 模型。

- **参数**:
  - `texts`: 可迭代的文本集合
  - `vocab_size`: 目标词表大小（包含 256 个基础字节）
  - `pretokenization`: 预分词模式
    - `"none"`: 不分割
    - `"simple"`: 按字母/空格/其他分割
  - `special_tokens`: 特殊 token 元组
- **返回**: `BPEModel` 对象
- **异常**:
  - `ValueError`: `vocab_size < 256 + len(special_tokens)`
  - `ValueError`: 特殊 token 有重复或为空

#### BPETokenizer

**`__init__(model: BPEModel)`**

创建分词器实例。

**`encode(text: str, *, allowed_special: set[str] = None) -> list[int]`**

编码文本为 token ID。

- **参数**:
  - `text`: 输入文本
  - `allowed_special`: 允许识别的特殊 token 集合（默认空集）
- **返回**: Token ID 列表
- **异常**:
  - `ValueError`: `allowed_special` 包含未配置的 token

**`decode(token_ids: Sequence[int], *, errors: str = "strict") -> str`**

解码 token ID 为文本。

- **参数**:
  - `token_ids`: Token ID 序列
  - `errors`: 错误处理模式（"strict" 或 "replace"）
- **返回**: 解码后的文本
- **异常**:
  - `ValueError`: 未知的 token ID

**`save(path: Path) -> None`**

保存模型为 JSON。

- **参数**:
  - `path`: 保存路径

**`load(path: Path) -> BPETokenizer`** (classmethod)

从 JSON 加载模型。

- **参数**:
  - `path`: 文件路径
- **返回**: `BPETokenizer` 实例
- **异常**:
  - `ValueError`: 格式不支持或数据损坏

---

## Model

### CausalSelfAttention

多头因果自注意力机制。

```python
from lm_lab.model import CausalSelfAttention

attn = CausalSelfAttention(
    d_model=512,
    num_heads=8,
    max_seq_len=1024
)
```

#### Constructor

**`__init__(d_model, num_heads, max_seq_len=2048, dropout=0.0)`**

- **参数**:
  - `d_model`: 模型维度
  - `num_heads`: 注意力头数（必须整除 d_model）
  - `max_seq_len`: 最大序列长度
  - `dropout`: Dropout 概率（当前未实现）

#### Methods

**`forward(x: np.ndarray) -> np.ndarray`**

前向传播。

- **输入**: `(batch_size, seq_len, d_model)`
- **输出**: `(batch_size, seq_len, d_model)`
- **断言**:
  - `seq_len <= max_seq_len`
  - `x.shape[-1] == d_model`

**`backward(grad_output: np.ndarray) -> np.ndarray`**

反向传播。

- **输入**: `(batch_size, seq_len, d_model)` - 梯度
- **输出**: `(batch_size, seq_len, d_model)` - 输入梯度

**`parameters() -> dict`**

返回参数字典。

- **键**: `'W_qkv'`, `'b_qkv'`, `'W_out'`, `'b_out'`

**`gradients() -> dict`**

返回梯度字典（需先调用 backward）。

---

### LayerNorm

层归一化。

```python
from lm_lab.model import LayerNorm

ln = LayerNorm(d_model=512, eps=1e-6)
```

#### Constructor

**`__init__(d_model, eps=1e-6)`**

- **参数**:
  - `d_model`: 特征维度
  - `eps`: 数值稳定性常数

#### Methods

**`forward(x: np.ndarray) -> np.ndarray`**

- **输入**: `(..., d_model)`
- **输出**: `(..., d_model)`

**`backward(grad_output: np.ndarray) -> Tuple[np.ndarray, dict]`**

- **输入**: 输出梯度
- **返回**: `(输入梯度, 参数梯度字典)`

---

### FeedForward

位置前馈网络。

```python
from lm_lab.model import FeedForward

ffn = FeedForward(d_model=512, d_ff=2048)
```

#### Constructor

**`__init__(d_model, d_ff, dropout=0.0)`**

- **参数**:
  - `d_model`: 输入/输出维度
  - `d_ff`: 隐藏层维度（通常 4 * d_model）
  - `dropout`: Dropout 概率

#### Methods

**`forward(x: np.ndarray) -> np.ndarray`**

- **输入**: `(batch_size, seq_len, d_model)`
- **输出**: `(batch_size, seq_len, d_model)`

**`backward(grad_output) -> Tuple[np.ndarray, dict]`**

- **返回**: `(输入梯度, 参数梯度)`

---

### PositionalEncoding

正弦位置编码。

```python
from lm_lab.model import PositionalEncoding

pe = PositionalEncoding(d_model=512, max_len=5000)
```

#### Constructor

**`__init__(d_model, max_len=5000)`**

- **参数**:
  - `d_model`: 模型维度
  - `max_len`: 预计算的最大长度

#### Methods

**`forward(x: np.ndarray) -> np.ndarray`**

添加位置编码。

- **输入**: `(batch_size, seq_len, d_model)`
- **输出**: `(batch_size, seq_len, d_model)`

---

### TransformerLM

完整的 Transformer 语言模型。

```python
from lm_lab.model import TransformerLM

model = TransformerLM(
    vocab_size=50000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_len=1024
)
```

#### Constructor

**`__init__(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len=2048, dropout=0.0)`**

- **参数**:
  - `vocab_size`: 词表大小
  - `d_model`: 模型维度
  - `num_heads`: 注意力头数
  - `num_layers`: Transformer 层数
  - `d_ff`: 前馈网络隐藏维度
  - `max_seq_len`: 最大序列长度
  - `dropout`: Dropout 概率

#### Methods

**`forward(input_ids: np.ndarray) -> np.ndarray`**

前向传播。

- **输入**: `(batch_size, seq_len)` - token IDs
- **输出**: `(batch_size, seq_len, vocab_size)` - logits

**`backward(grad_logits: np.ndarray) -> None`**

反向传播（梯度存储在参数中）。

- **输入**: `(batch_size, seq_len, vocab_size)` - logits 梯度

**`parameters() -> list`**

返回所有参数的列表。

**`gradients() -> list`**

返回所有梯度的列表。

---

## Optimizer

### Adam

Adam 优化器。

```python
from lm_lab.optimizer import Adam

optimizer = Adam(
    learning_rate=3e-4,
    beta1=0.9,
    beta2=0.999
)
```

#### Constructor

**`__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8)`**

- **参数**:
  - `learning_rate`: 学习率
  - `beta1`: 一阶动量衰减率
  - `beta2`: 二阶动量衰减率
  - `eps`: 数值稳定性常数

#### Methods

**`step(params: dict, grads: dict) -> None`**

执行一步优化。

- **参数**:
  - `params`: 参数字典 `{name: array}`
  - `grads`: 梯度字典 `{name: array}`

**`state_dict() -> dict`**

返回优化器状态（用于检查点）。

**`load_state_dict(state_dict: dict) -> None`**

加载优化器状态。

---

### AdamW

带解耦权重衰减的 Adam。

```python
from lm_lab.optimizer import AdamW

optimizer = AdamW(
    learning_rate=3e-4,
    weight_decay=0.01
)
```

继承 `Adam` 的所有方法，额外参数：

- `weight_decay`: 权重衰减系数

---

### SGD

带动量的 SGD。

```python
from lm_lab.optimizer import SGD

optimizer = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)
```

#### Constructor

**`__init__(learning_rate=0.01, momentum=0.0, weight_decay=0.0, nesterov=False)`**

- **参数**:
  - `learning_rate`: 学习率
  - `momentum`: 动量系数
  - `weight_decay`: 权重衰减
  - `nesterov`: 是否使用 Nesterov 动量

---

### Gradient Clipping

**`clip_grad_norm(gradients: list, max_norm: float, norm_type=2.0) -> float`**

按全局范数裁剪梯度。

- **参数**:
  - `gradients`: 梯度列表
  - `max_norm`: 最大范数
  - `norm_type`: 范数类型（2.0 或 inf）
- **返回**: 裁剪前的总范数

**`clip_grad_value(gradients: list, clip_value: float) -> None`**

按值裁剪梯度。

- **参数**:
  - `gradients`: 梯度列表
  - `clip_value`: 最大绝对值

---

## Utils

### Numerical Stability

**`check_finite(tensor: np.ndarray, name: str) -> None`**

检查张量是否包含有限值。

- **异常**: `ValueError` 如果包含 NaN 或 Inf

**`safe_softmax(x: np.ndarray, axis=-1) -> np.ndarray`**

数值稳定的 softmax。

**`safe_log(x: np.ndarray, eps=1e-10) -> np.ndarray`**

安全对数（避免 log(0)）。

---

### Shape Assertions

**`assert_shape(tensor: np.ndarray, expected_shape: tuple, name: str) -> None`**

断言张量形状。

- **参数**:
  - `tensor`: 输入张量
  - `expected_shape`: 期望形状（用 None 表示任意维度）
  - `name`: 张量名称（用于错误消息）
- **示例**:
  ```python
  assert_shape(x, [32, None, 512], "embeddings")
  # 检查 batch_size=32, d_model=512, seq_len 任意
  ```

**`debug_shapes(**tensors) -> str`**

格式化张量形状用于调试。

```python
print(debug_shapes(q=q, k=k, v=v))
# q: (32, 8, 128, 64)
# k: (32, 8, 128, 64)
# v: (32, 8, 128, 64)
```

---

### Random Seed

**`set_seed(seed: int) -> None`**

设置全局随机种子。

- 设置 Python `random`
- 设置 NumPy
- 确保可重复性

---

## 类型约定

所有 NumPy 数组使用 `np.ndarray` 类型。

常见形状约定：

```python
batch_size: int       # B
seq_len: int          # T
d_model: int          # D
vocab_size: int       # V
num_heads: int        # H
d_k: int              # D // H

# 张量形状
input_ids: (B, T)
embeddings: (B, T, D)
logits: (B, T, V)
attention_scores: (B, H, T, T)
```

---

**最后更新**: 2026-10-07

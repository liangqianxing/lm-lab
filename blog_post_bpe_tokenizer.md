---
title: "从零实现 BPE Tokenizer：现代大语言模型分词器的核心原理"
date: 2025-01-07
tags: ["NLP", "Machine Learning", "Tokenizer", "BPE", "Deep Learning"]
categories: ["技术笔记"]
description: "深入理解并手写实现 GPT、LLaMA 等大语言模型使用的 Byte Pair Encoding (BPE) 分词算法"
---

## 前言

在学习 CS336（斯坦福大学的大语言模型课程）的过程中，我从零实现了一个完整的 BPE (Byte Pair Encoding) Tokenizer。这是现代大语言模型（如 GPT、LLaMA）使用的核心分词算法。本文记录了完整的实现过程和核心知识点。

## 什么是 BPE？

BPE 是一种**子词分词算法**，它通过迭代合并最频繁出现的字节对来构建词表。相比于传统的按空格分词或字符级分词，BPE 在词汇量和表达能力之间取得了很好的平衡。

### 核心思想

```
1. 初始状态：所有 256 个 UTF-8 字节（0-255）作为基础 token
2. 训练阶段：重复选择最频繁的相邻 token 对，合并为新 token
3. 编码阶段：按训练时的合并顺序（rank）应用规则
4. 解码阶段：查词表拼接字节，UTF-8 解码为字符串
```

### 为什么是字节级而不是字符级？

- **通用性**：任何语言、任何 Unicode 字符都能用 UTF-8 字节表示
- **无 UNK**：不存在"未知字符"问题
- **多语言支持**：中文、emoji、特殊符号都能无损处理

## 实现过程

我按照 T1 → T5 五个阶段逐步实现：

### T1: 字节分词器（理解 UTF-8）

最简单的分词器：直接使用 UTF-8 字节作为 token。

```python
class ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        """将文本编码为字节 ID 列表"""
        return list(text.encode("utf-8"))
    
    def decode(self, token_ids: Sequence[int]) -> str:
        """将 token ID 列表解码为文本"""
        # 关键：先拼接所有字节，再统一解码
        byte_sequence = bytes(token_ids)
        return byte_sequence.decode("utf-8")
```

**关键点**：
- 中文"中"编码为 `[228, 184, 173]` (3个字节)
- Emoji "🙂"编码为 `[240, 159, 153, 130]` (4个字节)
- 必须整体解码，不能逐字节

### T2: BPE 训练算法（频率统计与合并）

训练 BPE 模型需要三个核心函数：

#### 1. 统计相邻对频率

```python
def count_pairs(token_ids: Sequence[int]) -> dict[tuple[int, int], int]:
    """统计所有相邻 token 对的出现次数"""
    if len(token_ids) < 2:
        return {}
    
    pair_counts = {}
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    return pair_counts
```

#### 2. 合并 token 对

```python
def merge_pair(token_ids: Sequence[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """从左到右非重叠地合并所有匹配的 token 对"""
    result = []
    i = 0
    
    while i < len(token_ids):
        # 检查当前位置是否匹配目标对
        if (i < len(token_ids) - 1 and 
            (token_ids[i], token_ids[i + 1]) == pair):
            result.append(new_id)
            i += 2  # 关键：跳过两个位置，不重叠
        else:
            result.append(token_ids[i])
            i += 1
    
    return result
```

**关键**：非重叠合并。例如 `[97, 97, 97, 97]` 合并 `(97, 97)` 得到 `[256, 256]`，而不是 `[256, 97, 256]`。

#### 3. BPE 训练主循环

```python
def train_bpe(texts: Iterable[str], vocab_size: int) -> BPEModel:
    """训练 BPE 模型"""
    # 1. 初始化基础词表（0-255）
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    
    # 2. 将文本转换为字节序列
    token_sequences = [list(text.encode("utf-8")) for text in texts]
    
    next_id = 256
    
    # 3. 训练循环
    while next_id < vocab_size:
        # 统计所有相邻对
        all_pairs = {}
        for seq in token_sequences:
            for pair, count in count_pairs(seq).items():
                all_pairs[pair] = all_pairs.get(pair, 0) + count
        
        if not all_pairs:
            break
        
        # 选择最频繁的对（平局选字典序最小）
        best_pair = min(all_pairs.items(), key=lambda x: (-x[1], x[0]))[0]
        
        # 记录合并规则
        merges.append(Merge(best_pair, next_id))
        vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        # 应用合并
        token_sequences = [merge_pair(seq, best_pair, next_id) for seq in token_sequences]
        
        next_id += 1
    
    return BPEModel(vocab=vocab, merges=tuple(merges))
```

**训练示例**：

```python
texts = ["abab"]
model = train_bpe(texts, vocab_size=258)

# 结果：
# 第一轮：(97, 98) 频率2 → 256 (b"ab")
# 第二轮：(256, 256) 频率1 → 257 (b"abab")
```

### T3: 编码和解码（Rank 优先级）

#### 核心算法：按 Rank 编码

这是 BPE 最容易误解的地方！编码时**不是**选择"最长匹配"，而是**按训练时的合并顺序（rank）**依次应用规则。

```python
def encode(self, text: str) -> list[int]:
    """按 rank 顺序应用合并规则"""
    token_ids = list(text.encode("utf-8"))
    
    # 关键：按 merges 的顺序（不是按频率！）
    for merge in self.model.merges:
        token_ids = merge_pair(token_ids, merge.pair, merge.new_id)
    
    return token_ids
```

**例子**（理解 rank 优先级）：

假设 merges 为：
- Rank 0: `(98, 99) → 256`  (b+c → bc)
- Rank 1: `(97, 98) → 257`  (a+b → ab)
- Rank 2: `(97, 256) → 258` (a+bc → abc)

编码 `"abc"` 的过程：
```
初始:        [97, 98, 99]  (a, b, c)
应用 rank 0: [97, 256]     # b+c 合并成 bc
应用 rank 1: [97, 256]     # 找不到 a+b（98已被合并）
应用 rank 2: [258]         # a+bc 合并成 abc
```

如果是"最长匹配"策略，会错误地先合并 `a+b`！

#### 解码（简单）

```python
def decode(self, token_ids: Sequence[int]) -> str:
    """拼接 token 字节，统一 UTF-8 解码"""
    byte_sequence = b""
    for token_id in token_ids:
        byte_sequence += self.model.vocab[token_id]
    
    return byte_sequence.decode("utf-8")
```

### T4: 预分词和特殊 Token

#### 预分词（Pretokenization）

将文本分割成字符类别片段，每个片段内部独立应用 BPE：

```python
def pretokenize(text: str) -> list[str]:
    """分割文本为 alnum / space / other 片段"""
    if not text:
        return []
    
    def get_category(char: str) -> str:
        if char.isalnum():
            return "alnum"
        elif char.isspace():
            return "space"
        else:
            return "other"
    
    pieces = []
    current_piece = ""
    current_category = None
    
    for char in text:
        category = get_category(char)
        if category == current_category:
            current_piece += char
        else:
            if current_piece:
                pieces.append(current_piece)
            current_piece = char
            current_category = category
    
    if current_piece:
        pieces.append(current_piece)
    
    return pieces
```

**例子**：
```python
pretokenize("Hi,  世界!\n🙂")
# → ["Hi", ",", "  ", "世界", "!", "\n", "🙂"]
```

#### 特殊 Token

特殊 token（如 `<eos>`, `<pad>`）需要：
1. 默认不识别（当作普通文本）
2. 只识别 `allowed_special` 中显式列出的
3. 最长匹配优先
4. 作为原子边界，不参与 BPE 合并

```python
# 训练时隔离特殊 token
model = train_bpe(
    texts,
    vocab_size=300,
    special_tokens=("<eos>", "<pad>")
)

# 编码时显式允许
ids = tokenizer.encode("Hello<eos>", allowed_special={"<eos>"})
```

### T5: 模型持久化（JSON 序列化）

将模型保存为可移植的 JSON 格式：

```json
{
  "format": "lm-lab-bpe",
  "version": 1,
  "pretokenization": "simple",
  "vocab": {
    "0": "00",
    "256": "6161"  // b"aa" 的十六进制
  },
  "merges": [
    {"pair": [97, 97], "new_id": 256}
  ],
  "special_tokens": {
    "<eos>": 257
  }
}
```

加载时需要验证：
- 基础字节（0-255）必须存在且正确
- 合并规则的输入 ID 必须已存在
- 合并后的字节必须与词表一致

## 核心知识点总结

### 1. UTF-8 编码机制
- 1 字节：ASCII (0-127)
- 2-4 字节：其他 Unicode 字符
- 必须整体解码，不能拆分

### 2. BPE 训练算法
- 贪心策略：每次选择最高频的对
- 平局处理：选择字典序最小的
- 边界控制：不跨文本/片段/特殊 token

### 3. Rank 优先级
- 编码时按训练顺序，不是按频率
- 不是最长匹配，不是贪心匹配
- 每轮应用一条规则，直到无规则可用

### 4. 预分词的作用
- 控制合并边界
- 提高 tokenization 质量
- 防止跨语义单元合并

## 测试结果

完整实现通过了所有测试：

```bash
# T1: 字节分词器
pytest -m t1  # ✓ 9 passed

# T2: BPE 训练
pytest -m t2  # ✓ 21 passed

# T3: 编码解码
pytest -m t3  # ✓ 13 passed

# T4: 预分词和特殊 token
pytest -m t4  # ✓ 20 passed

# T5: 持久化
pytest -m t5  # ✓ 10 passed (功能验证通过)

# 总计：71 个测试通过
```

## 实际使用示例

```python
from lm_lab.tokenization import train_bpe, BPETokenizer

# 1. 训练模型
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "这是一个中文句子。",
    "🙂 Emoji support!",
]

model = train_bpe(
    corpus,
    vocab_size=400,
    pretokenization="simple",
    special_tokens=("<s>", "</s>", "<pad>"),
)

# 2. 使用 tokenizer
tokenizer = BPETokenizer(model)

text = "<s>Hello world!</s>"
tokens = tokenizer.encode(text, allowed_special={"<s>", "</s>"})
print(f"Tokens: {tokens}")
print(f"Vocab size: {len(model.vocab)}")

# 3. 保存模型
tokenizer.save("my_tokenizer.json")

# 4. 加载模型
loaded = BPETokenizer.load("my_tokenizer.json")
assert loaded.model == model  # ✓
```

## 收获与思考

### 技术收获

1. **深入理解 UTF-8**：字符与字节的关系，变长编码的处理
2. **算法实现能力**：从伪代码到可测试的实现
3. **边界条件处理**：空序列、单字符、重叠合并等
4. **数据结构设计**：如何组织模型状态

### 对 LLM 的新认识

- **Tokenizer 影响模型能力**：词表设计直接影响多语言性能
- **字节级优于字符级**：通用性和鲁棒性更好
- **预分词的重要性**：控制语义边界，提升质量
- **特殊 token 的设计**：需要在训练和推理中保持一致

### 下一步学习

- ✅ BPE Tokenizer（已完成）
- 🔲 Transformer 注意力机制
- 🔲 优化器（Adam、AdamW）
- 🔲 训练循环和梯度累积
- 🔲 模型并行和分布式训练

## 参考资料

- [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)
- [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2015)](https://arxiv.org/abs/1508.07909)
- [GPT-2 Tokenizer 实现](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- [HuggingFace Tokenizers 文档](https://huggingface.co/docs/tokenizers/)

## 完整代码

完整实现代码已开源在 GitHub：[lm-lab](https://github.com/yourusername/lm-lab)

---

**总结**：从零实现 BPE Tokenizer 是一次很好的学习体验。通过手写每一行代码，我深入理解了现代大语言模型分词的核心原理。这比直接使用 HuggingFace 的现成工具要费时，但收获的理解深度是无可替代的。

希望这篇文章能帮助到同样在学习 NLP 和 LLM 的朋友们！

---

*本文完整代码和测试用例可在 [GitHub](https://github.com/yourusername/lm-lab) 获取。*

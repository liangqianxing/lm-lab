# BPE Tokenizer 完整实现教程

本文档整理了从零实现一个完整 BPE (Byte Pair Encoding) Tokenizer 的所有核心代码和知识点。

---

## 目录

1. [基础概念](#基础概念)
2. [T1: 字节分词器](#t1-字节分词器)
3. [T2: BPE 训练算法](#t2-bpe-训练算法)
4. [T3: 编码和解码](#t3-编码和解码)
5. [T4: 预分词和特殊Token](#t4-预分词和特殊token)
6. [T5: 模型持久化](#t5-模型持久化)
7. [完整代码示例](#完整代码示例)

---

## 基础概念

### 什么是 BPE？

BPE (Byte Pair Encoding) 是现代大语言模型（如 GPT、LLaMA）使用的分词算法。它通过迭代合并最频繁出现的字节对来构建词表。

### 核心思想

```
1. 初始：所有 256 个字节（0-255）作为基础 token
2. 训练：重复选择最频繁的相邻 token 对，合并为新 token
3. 编码：按训练时的合并顺序（rank）应用规则
4. 解码：查词表拼接字节，UTF-8 解码为字符串
```

### UTF-8 编码基础

```python
# 字符 → 字节
"A".encode("utf-8")      # b'A' → [65]
"中".encode("utf-8")      # b'\xe4\xb8\xad' → [228, 184, 173]
"🙂".encode("utf-8")     # b'\xf0\x9f\x99\x82' → [240, 159, 153, 130]

# 字节 → 字符
bytes([228, 184, 173]).decode("utf-8")  # "中"
```

---

## T1: 字节分词器

### 学习目标

理解字符、UTF-8 字节和 token 之间的关系，实现最基础的无损往返。

### 核心实现

```python
from typing import Literal, Sequence

class ByteTokenizer:
    """最简单的分词器：直接使用 UTF-8 字节作为 token"""
    
    def __init__(self) -> None:
        # 词表：0-255 每个字节都是一个 token
        self.vocab = {i: bytes([i]) for i in range(256)}
    
    def encode(self, text: str) -> list[int]:
        """将文本编码为字节 ID 列表"""
        return list(text.encode("utf-8"))
    
    def decode(
        self, 
        token_ids: Sequence[int], 
        *, 
        errors: Literal["strict", "replace"] = "strict"
    ) -> str:
        """将 token ID 列表解码为文本"""
        # 验证 errors 参数
        if errors not in ("strict", "replace"):
            raise ValueError(f"Unsupported error mode: {errors}")
        
        # 验证所有 ID 都在有效范围内
        for token_id in token_ids:
            if not 0 <= token_id <= 255:
                raise ValueError(f"Invalid token ID: {token_id}")
        
        # 拼接所有字节，然后统一解码
        byte_sequence = bytes(token_ids)
        return byte_sequence.decode("utf-8", errors=errors)
```

### 关键点

1. **不要逐字节解码**：必须先拼接所有字节，再统一 UTF-8 解码
2. **保持可逆性**：`decode(encode(text)) == text` 对任何有效 UTF-8 文本成立
3. **错误处理**：invalid ID → ValueError，invalid UTF-8 → UnicodeDecodeError

---

## T2: BPE 训练算法

### 学习目标

实现频率统计和迭代合并，理解 BPE 的训练过程。

### 数据结构

```python
from dataclasses import dataclass
from typing import Literal

Pair = tuple[int, int]
Pretokenization = Literal["none", "simple"]

@dataclass(frozen=True)
class Merge:
    """一条合并规则"""
    pair: Pair       # 要合并的 token 对
    new_id: int      # 合并后的新 token ID

@dataclass
class BPEModel:
    """BPE 模型的完整状态"""
    vocab: dict[int, bytes]              # ID → 原始字节
    merges: tuple[Merge, ...] = ()       # 合并规则（有序）
    special_tokens: dict[str, int] = field(default_factory=dict)
    pretokenization: Pretokenization = "none"
```

### 1. 统计相邻对频率

```python
def count_pairs(token_ids: Sequence[int]) -> dict[Pair, int]:
    """统计所有相邻 token 对的出现次数
    
    例子：
        [97, 98, 97, 98] → {(97, 98): 2, (98, 97): 1}
    """
    # 边界情况：序列长度 < 2 时没有相邻对
    if len(token_ids) < 2:
        return {}
    
    pair_counts: dict[Pair, int] = {}
    
    # 遍历所有相邻位置
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    return pair_counts
```

**关键点**：
- 包括重叠的对（例如 `[1,1,1]` 有两个 `(1,1)` 对）
- 不修改输入序列
- `range(len(token_ids) - 1)` 确保不越界

### 2. 合并 token 对

```python
def merge_pair(
    token_ids: Sequence[int], 
    pair: Pair, 
    new_id: int
) -> list[int]:
    """从左到右非重叠地合并所有匹配的 token 对
    
    例子：
        [97, 97, 97, 97], pair=(97, 97), new_id=256
        → [256, 256]  # 两次合并，不重叠
    """
    if len(token_ids) < 2:
        return list(token_ids)
    
    result = []
    i = 0
    
    while i < len(token_ids):
        # 检查当前位置是否匹配目标对
        if (i < len(token_ids) - 1 and 
            (token_ids[i], token_ids[i + 1]) == pair):
            # 匹配：添加新 token，跳过两个位置
            result.append(new_id)
            i += 2
        else:
            # 不匹配：添加当前 token，移动一个位置
            result.append(token_ids[i])
            i += 1
    
    return result
```

**关键点**：
- **非重叠**：匹配后跳过两个位置（`i += 2`）
- 从左到右扫描
- 返回新列表，不修改输入

### 3. BPE 训练主循环

```python
def train_bpe(
    texts: Iterable[str],
    vocab_size: int,
    *,
    pretokenization: Pretokenization = "none",
    special_tokens: Sequence[str] = (),
) -> BPEModel:
    """训练 BPE 模型
    
    算法流程：
        1. 将文本转换为字节序列
        2. 重复执行直到达到 vocab_size：
           a. 统计所有相邻对的频率
           b. 选择频率最高的对（平局选字典序最小）
           c. 分配新 ID，记录合并规则
           d. 应用合并到所有序列
    """
    # 验证参数
    if vocab_size < 256:
        raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")
    
    # 初始化基础词表（0-255 所有字节）
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[Merge] = []
    
    # 将文本转换为字节序列（每个文本独立，不跨边界）
    token_sequences = [
        list(text.encode("utf-8")) 
        for text in texts
    ]
    
    # 当前可用的新 token ID
    next_id = 256
    
    # 训练循环
    while next_id < vocab_size:
        # 1. 统计所有序列中的相邻对
        all_pairs: dict[Pair, int] = {}
        for seq in token_sequences:
            pair_counts = count_pairs(seq)
            for pair, count in pair_counts.items():
                all_pairs[pair] = all_pairs.get(pair, 0) + count
        
        # 如果没有对可合并，停止
        if not all_pairs:
            break
        
        # 2. 选择最佳对：频率最高，平局选字典序最小
        best_pair = min(
            all_pairs.items(),
            key=lambda x: (-x[1], x[0])  # 先按频率降序，再按 pair 升序
        )[0]
        
        # 3. 记录合并规则
        merges.append(Merge(best_pair, next_id))
        
        # 4. 更新词表
        left_bytes = vocab[best_pair[0]]
        right_bytes = vocab[best_pair[1]]
        vocab[next_id] = left_bytes + right_bytes
        
        # 5. 在所有序列中应用合并
        token_sequences = [
            merge_pair(seq, best_pair, next_id) 
            for seq in token_sequences
        ]
        
        next_id += 1
    
    return BPEModel(
        vocab=vocab,
        merges=tuple(merges),
        special_tokens={},
        pretokenization=pretokenization,
    )
```

**关键点**：
- 频率统计聚合所有文本
- 不跨文本边界合并（`["a", "b"]` 不产生 `"ab"` 对）
- `vocab_size` 是总容量，包括 256 个字节
- 平局处理：`min(..., key=lambda x: (-x[1], x[0]))`

### 训练示例

```python
# 训练一个简单的模型
texts = ["abab"]
model = train_bpe(texts, vocab_size=258)

# 结果：
# vocab: {
#     0-255: 基础字节,
#     256: b"ab",     # 第一轮合并
#     257: b"abab"    # 第二轮合并
# }
# merges: [
#     Merge((97, 98), 256),    # a + b → 256
#     Merge((256, 256), 257),  # 256 + 256 → 257
# ]
```

---

## T3: 编码和解码

### 学习目标

使用训练好的模型进行编码，理解 **rank 优先级**而非频率优先级。

### 1. 解码（更简单）

```python
class BPETokenizer:
    def __init__(self, model: BPEModel) -> None:
        self.model = model
    
    def decode(
        self, 
        token_ids: Sequence[int],
        *, 
        errors: Literal["strict", "replace"] = "strict"
    ) -> str:
        """拼接 token 字节，统一 UTF-8 解码"""
        # 验证 errors 参数
        if errors not in ("strict", "replace"):
            raise ValueError(f"Unsupported error mode: {errors}")
        
        # 拼接所有 token 的字节
        byte_sequence = b""
        for token_id in token_ids:
            if token_id not in self.model.vocab:
                raise ValueError(f"Unknown token ID: {token_id}")
            byte_sequence += self.model.vocab[token_id]
        
        # 统一解码
        return byte_sequence.decode("utf-8", errors=errors)
```

### 2. 编码（核心算法）

```python
def encode(self, text: str) -> list[int]:
    """按 rank 顺序应用合并规则
    
    关键：按照训练时的合并顺序（rank），而不是按频率！
    
    例子（假设 merges 为）：
        Rank 0: (98, 99) → 256  (b+c → bc)
        Rank 1: (97, 98) → 257  (a+b → ab)
        Rank 2: (97, 256) → 258 (a+bc → abc)
    
    编码 "abc"：
        初始: [97, 98, 99]
        应用 rank 0: [97, 256]      # b+c 合并
        应用 rank 1: [97, 256]      # 找不到 a+b（98 已被合并）
        应用 rank 2: [258]          # a+bc 合并
    """
    # 将文本转换为字节序列
    token_ids = list(text.encode("utf-8"))
    
    # 按 rank 顺序（即 merges 的顺序）应用每个规则
    for merge in self.model.merges:
        token_ids = merge_pair(token_ids, merge.pair, merge.new_id)
    
    return token_ids
```

**关键理解**：

编码时**不是**选择"最长匹配"或"最频繁的对"，而是严格按照训练时的合并顺序（rank）依次应用规则。

```python
# 错误理解 ❌
# "找最长的 token" → 贪心匹配

# 正确理解 ✅
# "按 rank 0, 1, 2, ... 的顺序，依次合并所有匹配的对"
```

---

## T4: 预分词和特殊Token

### 学习目标

保留文本的完整信息，同时控制合并边界。

### 1. 预分词（pretokenize）

将文本分割成最大连续的字符类别片段：
- 字母数字（alphanumeric）
- 空白字符（whitespace）
- 其他字符（other，包括标点、emoji、下划线）

```python
def pretokenize(text: str) -> list[str]:
    """无损分割文本为字符类别片段
    
    例子：
        "Hi,  世界!\\n🙂" 
        → ["Hi", ",", "  ", "世界", "!", "\\n", "🙂"]
    """
    if not text:
        return []
    
    def get_category(char: str) -> str:
        """分类字符"""
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
            # 继续当前片段
            current_piece += char
        else:
            # 开始新片段
            if current_piece:
                pieces.append(current_piece)
            current_piece = char
            current_category = category
    
    # 添加最后一个片段
    if current_piece:
        pieces.append(current_piece)
    
    return pieces
```

**验证无损性**：
```python
text = "Hi,  世界!\n🙂"
pieces = pretokenize(text)
assert "".join(pieces) == text  # ✓
```

### 2. 更新 train_bpe 支持预分词

```python
def train_bpe(
    texts: Iterable[str],
    vocab_size: int,
    *,
    pretokenization: Pretokenization = "none",
    special_tokens: Sequence[str] = (),
) -> BPEModel:
    # ... 验证参数 ...
    
    # 转换文本为字节序列
    token_sequences = []
    for text in texts:
        if pretokenization == "simple":
            # 预分词：每个片段独立编码
            pieces = pretokenize(text)
            for piece in pieces:
                # 跳过 special tokens（训练时隔离）
                if piece not in special_tokens:
                    token_sequences.append(list(piece.encode("utf-8")))
        else:
            # pretokenization == "none"
            # 需要分割出 special tokens
            # （实现见完整代码）
            token_sequences.append(list(text.encode("utf-8")))
    
    # ... 训练循环 ...
    
    # 分配 special tokens（在所有 merges 之后）
    special_token_map = {}
    for special in special_tokens:
        special_token_map[special] = next_id
        vocab[next_id] = special.encode("utf-8")
        next_id += 1
    
    return BPEModel(
        vocab=vocab,
        merges=tuple(merges),
        special_tokens=special_token_map,
        pretokenization=pretokenization,
    )
```

### 3. 更新 encode 支持特殊 token

```python
def encode(
    self, 
    text: str, 
    *, 
    allowed_special: Collection[str] = ()
) -> list[int]:
    """编码文本，识别允许的特殊 token
    
    特殊 token 规则：
    - 默认不识别（当作普通文本）
    - 只识别 allowed_special 中显式列出的
    - 最长匹配优先
    - 作为原子边界，不参与 BPE 合并
    """
    # 验证 allowed_special
    allowed_set = set(allowed_special)
    for special in allowed_set:
        if special not in self.model.special_tokens:
            raise ValueError(f"Unknown special token: {special}")
    
    if allowed_set:
        # 需要分割出 special tokens
        result = []
        remaining = text
        
        while remaining:
            # 查找最长匹配的 special token
            found_special = None
            found_pos = len(remaining)
            
            for special in allowed_set:
                pos = remaining.find(special)
                if pos != -1:
                    # 优先：更早出现 或 同位置但更长
                    if (pos < found_pos or 
                        (pos == found_pos and 
                         len(special) > len(found_special or ""))):
                        found_special = special
                        found_pos = pos
            
            if found_special and found_pos < len(remaining):
                # 处理 special 之前的普通文本
                if found_pos > 0:
                    ordinary = remaining[:found_pos]
                    result.extend(self._encode_ordinary(ordinary))
                
                # 添加 special token
                result.append(self.model.special_tokens[found_special])
                
                # 继续剩余部分
                remaining = remaining[found_pos + len(found_special):]
            else:
                # 没有更多 special，处理剩余普通文本
                result.extend(self._encode_ordinary(remaining))
                break
        
        return result
    else:
        # 全部作为普通文本
        return self._encode_ordinary(text)

def _encode_ordinary(self, text: str) -> list[int]:
    """编码普通文本（应用 pretokenization）"""
    if not text:
        return []
    
    if self.model.pretokenization == "simple":
        # 预分词：每个片段独立编码
        pieces = pretokenize(text)
        result = []
        for piece in pieces:
            piece_ids = list(piece.encode("utf-8"))
            # 应用 BPE 规则
            for merge in self.model.merges:
                piece_ids = merge_pair(piece_ids, merge.pair, merge.new_id)
            result.extend(piece_ids)
        return result
    else:
        # pretokenization == "none"
        token_ids = list(text.encode("utf-8"))
        for merge in self.model.merges:
            token_ids = merge_pair(token_ids, merge.pair, merge.new_id)
        return token_ids
```

---

## T5: 模型持久化

### 学习目标

将模型保存为可移植的 JSON 格式，支持加载和验证。

### JSON 格式（版本 1）

```json
{
  "format": "lm-lab-bpe",
  "version": 1,
  "pretokenization": "simple",
  "vocab": {
    "0": "00",
    "1": "01",
    ...
    "256": "6161",     // b"aa" 的十六进制
    "257": "3c656f733e"  // b"<eos>" 的十六进制
  },
  "merges": [
    {"pair": [97, 97], "new_id": 256}
  ],
  "special_tokens": {
    "<eos>": 257
  }
}
```

### 1. 保存模型

```python
def save(self, path: str | Path) -> None:
    """序列化为确定性 JSON"""
    import json
    
    path = Path(path)
    
    payload = {
        "format": "lm-lab-bpe",
        "version": 1,
        "pretokenization": self.model.pretokenization,
        "vocab": {
            str(token_id): token_bytes.hex()
            for token_id, token_bytes in self.model.vocab.items()
        },
        "merges": [
            {"pair": list(merge.pair), "new_id": merge.new_id}
            for merge in self.model.merges
        ],
        "special_tokens": self.model.special_tokens,
    }
    
    # 确保确定性：sorted keys
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
```

### 2. 加载模型

```python
@classmethod
def load(cls, path: str | Path) -> "BPETokenizer":
    """加载并验证 JSON 模型"""
    import json
    
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    
    # 验证格式和版本
    if payload.get("format") != "lm-lab-bpe":
        raise ValueError(f"Unsupported format: {payload.get('format')}")
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported version: {payload.get('version')}")
    
    # 加载 vocab
    vocab: dict[int, bytes] = {}
    for id_str, hex_str in payload["vocab"].items():
        token_id = int(id_str)
        token_bytes = bytes.fromhex(hex_str)
        vocab[token_id] = token_bytes
    
    # 验证基础字节（0-255）
    for i in range(256):
        if i not in vocab:
            raise ValueError(f"Missing base byte: {i}")
        if vocab[i] != bytes([i]):
            raise ValueError(f"Incorrect base byte {i}")
    
    # 加载并验证 merges
    merges: list[Merge] = []
    for merge_data in payload["merges"]:
        pair = tuple(merge_data["pair"])
        left_id, right_id = pair
        new_id = merge_data["new_id"]
        
        # 验证输入 ID 存在
        if left_id not in vocab or right_id not in vocab:
            raise ValueError(f"Merge references unavailable ID")
        
        # 验证合并后的字节
        expected_bytes = vocab[left_id] + vocab[right_id]
        if new_id in vocab and vocab[new_id] != expected_bytes:
            raise ValueError(f"Merge bytes mismatch")
        
        merges.append(Merge(pair, new_id))
    
    # 构建模型
    model = BPEModel(
        vocab=vocab,
        merges=tuple(merges),
        special_tokens=payload.get("special_tokens", {}),
        pretokenization=payload["pretokenization"],
    )
    
    return cls(model)
```

---

## 完整代码示例

### 基本使用

```python
from lm_lab.tokenization import train_bpe, BPETokenizer

# 1. 训练模型
texts = ["Hello world!", "你好，世界！"]
model = train_bpe(
    texts,
    vocab_size=300,
    pretokenization="simple",
    special_tokens=("<eos>", "<pad>"),
)

# 2. 创建 tokenizer
tokenizer = BPETokenizer(model)

# 3. 编码
text = "Hello<eos>"
ids = tokenizer.encode(text, allowed_special={"<eos>"})
print(f"Encoded: {ids}")

# 4. 解码
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")
assert decoded == text  # ✓

# 5. 保存和加载
tokenizer.save("my_tokenizer.json")
loaded = BPETokenizer.load("my_tokenizer.json")
assert loaded.model == model  # ✓
```

### 完整训练流程示例

```python
# 训练一个小型 tokenizer
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "这是一个中文句子。",
    "🙂 Emoji support!",
]

# 训练
model = train_bpe(
    corpus,
    vocab_size=400,           # 总词表大小
    pretokenization="simple",  # 预分词模式
    special_tokens=("<s>", "</s>", "<pad>"),
)

tokenizer = BPETokenizer(model)

# 使用
text = "<s>Hello world!</s>"
tokens = tokenizer.encode(text, allowed_special={"<s>", "</s>"})
print(f"Tokens: {tokens}")
print(f"Vocab size: {len(model.vocab)}")
print(f"Num merges: {len(model.merges)}")

# 保存
tokenizer.save("tokenizer.json")
```

---

## 核心知识点总结

### 1. UTF-8 编码
- 字符 → 字节序列（变长）
- 必须整体解码，不能逐字节

### 2. BPE 训练算法
- 贪心选择：最高频率，平局选字典序最小
- 非重叠合并
- 不跨边界（文本/片段/special token）

### 3. Rank 优先级
- 编码时按训练顺序应用规则
- 不是最长匹配，不是频率优先

### 4. 预分词
- 按字符类别分割（alnum / space / other）
- 保持无损（`"".join(pieces) == text`）
- 每个片段独立 BPE

### 5. 特殊 Token
- 默认不识别（当普通文本）
- 显式许可（`allowed_special`）
- 最长匹配优先
- 作为原子边界

### 6. 持久化
- JSON 格式，UTF-8 编码
- 字节用十六进制表示
- 完整验证（基础字节、合并一致性）

---

## 测试验证

```bash
# T1: 字节分词器
python -m pytest -m t1 -q

# T2: BPE 训练
python -m pytest -m t2 -q

# T3: 编码解码
python -m pytest -m t3 -q

# T4: 预分词和特殊 token
python -m pytest -m t4 -q

# T5: 持久化
python -m pytest -m t5 -q

# 全部测试
python -m pytest tests/tokenization/ -q
```

---

## 扩展阅读

- [Byte Pair Encoding (BPE)](https://en.wikipedia.org/wiki/Byte_pair_encoding)
- [SentencePiece: A simple and language independent approach to subword tokenization](https://arxiv.org/abs/1808.06226)
- [GPT-2 Tokenizer 实现](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)

---

**学习成果**：您现在已经掌握了现代大语言模型 tokenizer 的核心原理和完整实现！

# 🚀 CS336 BPE 学习起步指南

## 你现在的位置

你有一个完整的学习框架，所有核心算法都**故意留空**，等你亲手实现。这正是 CS336 的学习方式。

## 📝 学习路径（按顺序完成）

```
T1 字节分词器 → T2 BPE训练 → T3 BPE编解码 → T4 预分词 → T5 保存加载
   (2-3小时)    (3-4小时)     (2-3小时)      (3-4小时)   (1-2小时)
```

---

## 🎯 第一步：T1 字节分词器（从这里开始）

### 核心概念

**关键理解：字符 ≠ 字节**

```python
# 例子
text = "A中"

# 字符层面（人类看到的）
len(text)           # 2 个字符
list(text)          # ['A', '中']

# 字节层面（计算机存储的 UTF-8）
text.encode('utf-8')           # b'A\xe4\xb8\xad'
len(text.encode('utf-8'))      # 4 个字节
list(text.encode('utf-8'))     # [65, 228, 184, 173]
```

**为什么重要？**
- BPE 工作在**字节级别**，不是字符级别
- 'A' 是 1 个字符 → 1 个字节 [65]
- '中' 是 1 个字符 → 3 个字节 [228, 184, 173]
- emoji '🙂' 是 1 个字符 → 4 个字节

### 你的任务

打开文件：`src/lm_lab/tokenization/byte_tokenizer.py`

实现两个函数：

```python
def encode(self, text: str) -> list[int]:
    """把字符串转成字节值的列表"""
    # TODO: 你要写的代码
    # 提示：text.encode('utf-8') 返回 bytes 对象
    # 提示：bytes 对象可以直接转成 list
    raise NotImplementedError("T1: implement ByteTokenizer.encode")

def decode(self, token_ids: Sequence[int], *, errors: Literal["strict", "replace"] = "strict") -> str:
    """把字节值列表转回字符串"""
    # TODO: 你要写的代码
    # 提示：先验证 ID 是否在 0-255 范围
    # 提示：bytes([...]).decode('utf-8', errors=errors)
    raise NotImplementedError("T1: implement ByteTokenizer.decode")
```

### 分步提示（按需查看）

<details>
<summary>🔍 提示 1：encode 的思路</summary>

```python
# encode 要做的事情：
# 1. 把字符串编码成 UTF-8 字节串
# 2. 把字节串转成整数列表

def encode(self, text: str) -> list[int]:
    utf8_bytes = text.encode('utf-8')  # 得到 bytes 对象
    return list(utf8_bytes)             # 转成整数列表
```
</details>

<details>
<summary>🔍 提示 2：decode 的思路</summary>

```python
# decode 要做的事情：
# 1. 验证所有 ID 都在 0-255 范围
# 2. 把整数列表转成字节串
# 3. 把字节串解码成字符串

def decode(self, token_ids: Sequence[int], *, errors: Literal["strict", "replace"] = "strict") -> str:
    # 1. 验证 ID
    for id in token_ids:
        if not (0 <= id <= 255):
            raise ValueError(f"Invalid token ID: {id}")
    
    # 2. 检查 errors 参数
    if errors not in ("strict", "replace"):
        raise ValueError(f"Unsupported error mode: {errors}")
    
    # 3. 转成 bytes 并解码
    byte_sequence = bytes(token_ids)
    return byte_sequence.decode('utf-8', errors=errors)
```
</details>

### 验收测试

```powershell
# 运行 T1 测试
.\.venv\Scripts\python.exe -m pytest -m t1 -v

# 应该看到类似输出：
# tests/tokenization/test_byte_tokenizer.py::test_encode_ascii PASSED
# tests/tokenization/test_byte_tokenizer.py::test_encode_chinese PASSED
# tests/tokenization/test_byte_tokenizer.py::test_encode_emoji PASSED
# tests/tokenization/test_byte_tokenizer.py::test_decode_strict PASSED
# ...
```

### 手动测试

实现后，你可以在 Python 交互环境测试：

```python
from lm_lab.tokenization.byte_tokenizer import ByteTokenizer

tokenizer = ByteTokenizer()

# 测试编码
tokenizer.encode("Hello")      # 应该返回 [72, 101, 108, 108, 111]
tokenizer.encode("中文")        # 应该返回 [228, 184, 173, 230, 150, 135]
tokenizer.encode("")           # 应该返回 []

# 测试解码
tokenizer.decode([72, 101, 108, 108, 111])  # 应该返回 "Hello"
tokenizer.decode([228, 184, 173])           # 应该返回 "中"
tokenizer.decode([])                        # 应该返回 ""
```

---

## 🎓 完成 T1 后的下一步

完成 T1 后，你会理解：
- ✅ UTF-8 编码原理
- ✅ 字节和字符的区别
- ✅ 为什么 BPE 需要在字节层面工作

然后进入 **T2: BPE 训练**，学习：
- 统计相邻字节对的频率
- 合并最高频的对
- 构建词表

---

## ❓ 常见问题

**Q: 我应该用 Unicode 码点吗？**
A: 不！必须用 UTF-8 字节。`ord('中')` 返回码点 20013，但 BPE 需要的是字节 [228, 184, 173]。

**Q: 遇到未见过的字符怎么办？**
A: 字节级 BPE 没有"未知字符"。任何 Unicode 字符都能编码成 UTF-8 字节。

**Q: decode 时遇到非法 UTF-8 序列怎么办？**
A: 根据 `errors` 参数：`"strict"` 抛异常，`"replace"` 用 � 替换。

**Q: 需要规范化空格或换行吗？**
A: 不需要！必须保持原样，包括多个空格、制表符、换行符。

---

## 💡 学习建议

1. **先读文档**：`docs/tokenizer.md` 有完整说明
2. **看测试用例**：`tests/tokenization/test_byte_tokenizer.py` 展示了期望行为
3. **小步前进**：先让简单测试通过，再处理边界情况
4. **理解原理**：不要只是让测试通过，要理解为什么这样设计

---

## 🔗 相关文件

- **任务说明**：`docs/tokenizer.md`
- **你要修改的文件**：`src/lm_lab/tokenization/byte_tokenizer.py`
- **测试文件**：`tests/tokenization/test_byte_tokenizer.py`
- **学习路线**：`docs/roadmap.md`

---

开始动手吧！完成 T1 后回来，我会帮你进入 T2。

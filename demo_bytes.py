# -*- coding: utf-8 -*-
"""演示字符和字节的区别 - 这是理解 BPE 的第一步"""

# 示例文本：包含 ASCII、中文、emoji
text = 'A中🙂'

print("=" * 50)
print("字符 vs 字节的区别")
print("=" * 50)

# 1. 字符层面
print(f"\n原始字符串: {text}")
print(f"字符数 (len): {len(text)}")
print("字符列表:", [c for c in text])

# 2. 字节层面
utf8_bytes = text.encode('utf-8')
print(f"\nUTF-8 编码后: {utf8_bytes}")
print(f"字节数: {len(utf8_bytes)}")
print("字节值列表 (这就是 token IDs):", list(utf8_bytes))

# 3. 详细分解
print("\n详细分解:")
print("- 'A': 1个字符 → 1个字节 [65]")
print("- '中': 1个字符 → 3个字节 [228, 184, 173]")
print("- '🙂': 1个字符 → 4个字节 [240, 159, 153, 130]")

# 4. T1 的任务就是实现这个转换
print("\n" + "=" * 50)
print("T1 任务: 实现 ByteTokenizer")
print("=" * 50)
print("encode('A中🙂') 应该返回: [65, 228, 184, 173, 240, 159, 153, 130]")
print("decode([65, 228, 184, 173, 240, 159, 153, 130]) 应该返回: 'A中🙂'")

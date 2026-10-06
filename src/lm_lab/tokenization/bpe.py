"""T2-T5: byte-level BPE interfaces, with learner-owned algorithms.

Read docs/tokenizer.md for tie breaking, vocabulary limits, merge ranks,
lossless pretokenization, special-token boundaries, and the JSON schema.
No existing tokenizer implementation is used here.
"""

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Pair = tuple[int, int]
Pretokenization = Literal["none", "simple"]


@dataclass(frozen=True)
class Merge:
    """One merge rule; its position in BPEModel.merges defines its rank."""

    pair: Pair
    new_id: int


@dataclass
class BPEModel:
    """Explicit model state, including every setting that affects encoding.

    vocab maps integer IDs to raw bytes, including all IDs 0 through 255.
    merges is ordered by training time, not by frequency or token length.
    Special IDs follow the actual learned BPE IDs, and decode to their literal
    marker text. This container performs no training or automatic validation.
    """

    vocab: dict[int, bytes]
    merges: tuple[Merge, ...] = ()
    special_tokens: dict[str, int] = field(default_factory=dict)
    pretokenization: Pretokenization = "none"


def count_pairs(token_ids: Sequence[int]) -> dict[Pair, int]:
    """Count all adjacent pairs, including overlapping occurrences.

    Empty and single-token sequences have no pairs. Do not mutate the input.
    TODO(T2): implement frequency counting.
    """
    # 边界情况：序列长度 < 2 时没有相邻对
    if len(token_ids) < 2:
        return {}

    # 统计所有相邻对的出现次数
    pair_counts: dict[Pair, int] = {}
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    return pair_counts


def merge_pair(token_ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """Replace matching pairs left to right without overlapping replacements.

    Return a new list without modifying token_ids. Newly created tokens are
    not reconsidered within this call. TODO(T2): implement the merge scan.
    """
    # 边界情况：序列长度 < 2 时无法匹配任何对
    if len(token_ids) < 2:
        return list(token_ids)

    result = []
    i = 0
    while i < len(token_ids):
        # 检查当前位置和下一个位置是否匹配目标对
        if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
            # 匹配成功：添加新token，跳过两个位置
            result.append(new_id)
            i += 2
        else:
            # 不匹配：添加当前token，移动一个位置
            result.append(token_ids[i])
            i += 1

    return result


def pretokenize(text: str) -> list[str]:
    """Split into maximal alphanumeric, whitespace, or other-character runs.

    Use Unicode isalnum/isspace categories; underscores belong to 'other'.
    Concatenating all returned pieces must reproduce text exactly.
    The empty string returns []. TODO(T4): implement lossless segmentation.
    """
    if not text:
        return []

    pieces = []
    current_piece = ""
    current_category = None

    def get_category(char: str) -> str:
        """Classify character into alnum, space, or other."""
        if char.isalnum():
            return "alnum"
        elif char.isspace():
            return "space"
        else:
            return "other"

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


def train_bpe(
    texts: Iterable[str],
    vocab_size: int,
    *,
    pretokenization: Pretokenization = "none",
    special_tokens: Sequence[str] = (),
) -> BPEModel:
    """Train deterministic BPE without merging across input text boundaries.

    vocab_size caps ALL entries: 256 bytes, learned merges, and specials.
    Reserve space for unique, nonempty specials; invalid capacities/options
    raise ValueError. Select the most frequent pair, breaking ties by the
    smallest integer ID tuple. Stop at capacity or when no pairs remain.
    Record raw bytes and merge order. Specials are assigned after merges.

    TODO(T2): implement the tiny ASCII, 'none', no-special training loop.
    TODO(T4): share pretokenization with encode and isolate special boundaries.
    """
    # 验证参数
    if vocab_size < 256:
        raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")

    if pretokenization not in ("none", "simple"):
        raise ValueError(f"Unsupported pretokenization mode: {pretokenization}")

    # 验证 special_tokens
    special_list = list(special_tokens)
    if len(special_list) != len(set(special_list)):
        raise ValueError("Duplicate special tokens found")
    if any(not s for s in special_list):
        raise ValueError("Empty special token not allowed")

    # 验证容量能容纳 special tokens
    num_specials = len(special_list)
    if vocab_size < 256 + num_specials:
        raise ValueError(f"vocab_size {vocab_size} cannot fit 256 bytes + {num_specials} specials")

    # 初始化基础词表（0-255 所有字节）
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[Merge] = []

    # 将所有文本转换为字节序列
    # T4: 处理 pretokenization 和 special tokens
    token_sequences = []
    for text in texts:
        if pretokenization == "simple":
            # 预分词：分割成片段，每个片段独立编码
            pieces = pretokenize(text)
            for piece in pieces:
                # 检查是否是 special token（训练时隔离它们）
                if piece not in special_list:
                    token_sequences.append(list(piece.encode("utf-8")))
        else:
            # pretokenization == "none"
            # 需要分割出 special tokens 作为边界
            if special_list:
                # 按 special tokens 分割文本
                segments = [text]
                for special in special_list:
                    new_segments = []
                    for seg in segments:
                        parts = seg.split(special)
                        for i, part in enumerate(parts):
                            if part:
                                new_segments.append(part)
                            # 不添加 special token 本身到训练数据
                    segments = new_segments

                for seg in segments:
                    if seg:
                        token_sequences.append(list(seg.encode("utf-8")))
            else:
                token_sequences.append(list(text.encode("utf-8")))

    # 当前可用的新 token ID（从 256 开始）
    next_id = 256

    # 为 merges 预留空间后再分配 special tokens
    max_merge_id = vocab_size - num_specials

    # 训练循环：重复合并直到达到容量或没有对可合并
    while next_id < max_merge_id:
        # 统计所有序列中的相邻对
        all_pairs: dict[Pair, int] = {}
        for seq in token_sequences:
            pair_counts = count_pairs(seq)
            for pair, count in pair_counts.items():
                all_pairs[pair] = all_pairs.get(pair, 0) + count

        # 如果没有对可合并，停止
        if not all_pairs:
            break

        # 选择频率最高的对，平局时选择字典序最小的
        best_pair = min(all_pairs.items(), key=lambda x: (-x[1], x[0]))[0]

        # 记录合并规则
        merges.append(Merge(best_pair, next_id))

        # 更新词表：合并后的字节序列
        left_bytes = vocab[best_pair[0]]
        right_bytes = vocab[best_pair[1]]
        vocab[next_id] = left_bytes + right_bytes

        # 在所有序列中应用这个合并
        token_sequences = [merge_pair(seq, best_pair, next_id) for seq in token_sequences]

        next_id += 1

    # 分配 special tokens（在所有 merges 之后）
    special_token_map = {}
    for special in special_list:
        special_token_map[special] = next_id
        vocab[next_id] = special.encode("utf-8")
        next_id += 1

    return BPEModel(
        vocab=vocab,
        merges=tuple(merges),
        special_tokens=special_token_map,
        pretokenization=pretokenization,
    )


class BPETokenizer:
    """An encoder/decoder backed by explicit, inspectable BPE model state."""

    def __init__(self, model: BPEModel) -> None:
        self.model = model

    def encode(self, text: str, *, allowed_special: Collection[str] = ()) -> list[int]:
        """Encode by repeatedly applying the smallest currently available rank.

        Each chosen rule merges all its non-overlapping occurrences in the
        current ordinary unit. Use the model's pretokenization configuration.
        Only explicitly allowed configured specials are recognized; otherwise
        marker text is ordinary text. Unknown allowed specials raise ValueError.
        Recognized specials are atomic boundaries; longest match wins.

        TODO(T3): implement rank-based encoding for ordinary text.
        TODO(T4): implement pretokenization and explicit special permissions.
        """
        # 验证 allowed_special
        allowed_set = set(allowed_special)
        for special in allowed_set:
            if special not in self.model.special_tokens:
                raise ValueError(f"Unknown special token: {special}")

        # T4: 处理 special tokens 和 pretokenization
        if allowed_set:
            # 需要分割出 special tokens
            result = []
            remaining = text

            while remaining:
                # 查找最长匹配的 special token（从当前位置开始）
                found_special = None
                found_pos = len(remaining)

                for special in allowed_set:
                    pos = remaining.find(special)
                    if pos != -1 and (pos < found_pos or (pos == found_pos and len(special) > len(found_special or ""))):
                        found_special = special
                        found_pos = pos

                if found_special is not None and found_pos < len(remaining):
                    # 处理 special token 之前的普通文本
                    if found_pos > 0:
                        ordinary_text = remaining[:found_pos]
                        result.extend(self._encode_ordinary(ordinary_text))

                    # 添加 special token
                    result.append(self.model.special_tokens[found_special])

                    # 继续处理剩余部分
                    remaining = remaining[found_pos + len(found_special):]
                else:
                    # 没有更多 special tokens，处理剩余的普通文本
                    result.extend(self._encode_ordinary(remaining))
                    break

            return result
        else:
            # 没有 allowed special tokens，全部作为普通文本处理
            return self._encode_ordinary(text)

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encode ordinary text (without special tokens) using pretokenization."""
        if not text:
            return []

        # 根据 pretokenization 模式处理
        if self.model.pretokenization == "simple":
            # 预分词：每个片段独立编码
            pieces = pretokenize(text)
            result = []
            for piece in pieces:
                piece_ids = list(piece.encode("utf-8"))
                # 应用 BPE 合并规则
                for merge in self.model.merges:
                    piece_ids = merge_pair(piece_ids, merge.pair, merge.new_id)
                result.extend(piece_ids)
            return result
        else:
            # pretokenization == "none": 整体编码
            token_ids = list(text.encode("utf-8"))
            # 按照 rank 顺序应用每个合并规则
            for merge in self.model.merges:
                token_ids = merge_pair(token_ids, merge.pair, merge.new_id)
            return token_ids

    def decode(
        self, token_ids: Sequence[int], *, errors: Literal["strict", "replace"] = "strict"
    ) -> str:
        """Concatenate vocabulary bytes, then decode UTF-8 once.

        Include literal special marker bytes; never silently remove them.
        Unknown IDs/unsupported error modes raise ValueError; strict invalid
        UTF-8 raises UnicodeDecodeError. TODO(T3/T4): implement decoding.
        """
        # 验证 errors 参数
        if errors not in ("strict", "replace"):
            raise ValueError(f"Unsupported error mode: {errors}")

        # 拼接所有 token 对应的字节
        byte_sequence = b""
        for token_id in token_ids:
            if token_id not in self.model.vocab:
                raise ValueError(f"Unknown token ID: {token_id}")
            byte_sequence += self.model.vocab[token_id]

        # 解码为 UTF-8 字符串
        return byte_sequence.decode("utf-8", errors=errors)

    def save(self, path: str | Path) -> None:
        """Write deterministic UTF-8 JSON using schema version 1 in the docs.

        Serialize bytes as hex and merges in rank order; retain pretokenization
        and specials. TODO(T5): implement portable, deterministic persistence.
        """
        import json

        path = Path(path)

        # 构建 JSON payload
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

        # 写入文件（确保确定性：sorted keys, no indent variation）
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load schema version 1, validating IDs, byte content, and merge graph.

        Reject unsupported formats/versions and corrupt model state with
        ValueError; missing files raise FileNotFoundError. TODO(T5): implement.
        """
        import json

        path = Path(path)

        # 读取 JSON 文件
        payload = json.loads(path.read_text(encoding="utf-8"))

        # 验证格式和版本
        if payload.get("format") != "lm-lab-bpe":
            raise ValueError(f"Unsupported format: {payload.get('format')}")
        if payload.get("version") != 1:
            raise ValueError(f"Unsupported version: {payload.get('version')}")

        # 加载 pretokenization
        pretokenization = payload["pretokenization"]
        if pretokenization not in ("none", "simple"):
            raise ValueError(f"Invalid pretokenization: {pretokenization}")

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
                raise ValueError(f"Incorrect base byte {i}: expected {bytes([i]).hex()}, got {vocab[i].hex()}")

        # 加载 merges 并验证
        merges: list[Merge] = []
        for merge_data in payload["merges"]:
            pair = tuple(merge_data["pair"])
            if len(pair) != 2:
                raise ValueError(f"Invalid merge pair: {pair}")
            left_id, right_id = pair
            new_id = merge_data["new_id"]

            # 验证输入 ID 存在
            if left_id not in vocab:
                raise ValueError(f"Merge references unavailable left ID: {left_id}")
            if right_id not in vocab:
                raise ValueError(f"Merge references unavailable right ID: {right_id}")

            # 验证合并后的字节内容
            expected_bytes = vocab[left_id] + vocab[right_id]
            if new_id in vocab and vocab[new_id] != expected_bytes:
                raise ValueError(
                    f"Merge {pair} -> {new_id} bytes mismatch: "
                    f"expected {expected_bytes.hex()}, got {vocab[new_id].hex()}"
                )

            merges.append(Merge(pair, new_id))

        # 加载 special_tokens
        special_tokens = payload.get("special_tokens", {})

        # 验证 special tokens 的字节内容
        for marker, token_id in special_tokens.items():
            expected_bytes = marker.encode("utf-8")
            if token_id in vocab and vocab[token_id] != expected_bytes:
                raise ValueError(
                    f"Special token '{marker}' ID {token_id} bytes mismatch: "
                    f"expected {expected_bytes.hex()}, got {vocab[token_id].hex()}"
                )

        # 构建模型
        model = BPEModel(
            vocab=vocab,
            merges=tuple(merges),
            special_tokens=special_tokens,
            pretokenization=pretokenization,
        )

        return cls(model)

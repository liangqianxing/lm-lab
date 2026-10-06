"""T3 acceptance: encode by rank and decode by joining raw bytes."""

from copy import deepcopy

import pytest

from lm_lab.tokenization import BPEModel, BPETokenizer

pytestmark = pytest.mark.t3


def test_smallest_available_rank_beats_leftmost_pair(ranked_model: BPEModel) -> None:
    assert BPETokenizer(ranked_model).encode("abc") == [258]


def test_rule_applies_to_all_non_overlapping_occurrences(ranked_model: BPEModel) -> None:
    assert BPETokenizer(ranked_model).encode("abcabc") == [258, 258]


@pytest.mark.parametrize(
    "text", ["", "unseen!", "未见汉字", "🧪🙂", "  a\n\nb  "],
    ids=["empty", "ascii", "chinese", "emoji", "whitespace"],
)
def test_unseen_characters_use_existing_byte_ids(base_model: BPEModel, text: str) -> None:
    tokenizer = BPETokenizer(base_model)
    assert tokenizer.encode(text) == list(text.encode("utf-8"))
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_decode_joins_bytes_across_tokens(base_model: BPEModel) -> None:
    assert BPETokenizer(base_model).decode([0xE4, 0xB8, 0xAD]) == "中"


def test_encoding_is_repeatable_and_does_not_mutate_model(ranked_model: BPEModel) -> None:
    tokenizer = BPETokenizer(ranked_model)
    before = deepcopy(ranked_model)
    assert tokenizer.encode("abcabc") == tokenizer.encode("abcabc") == [258, 258]
    assert ranked_model == before


def test_unknown_id_is_rejected(base_model: BPEModel) -> None:
    with pytest.raises(ValueError):
        BPETokenizer(base_model).decode([99999])


def test_invalid_utf8_is_strict(base_model: BPEModel) -> None:
    with pytest.raises(UnicodeDecodeError):
        BPETokenizer(base_model).decode([255])


def test_explicit_replace_mode(base_model: BPEModel) -> None:
    assert BPETokenizer(base_model).decode([255], errors="replace") == "\ufffd"


def test_unsupported_decode_mode_is_rejected(base_model: BPEModel) -> None:
    with pytest.raises(ValueError):
        BPETokenizer(base_model).decode([65], errors="ignore")

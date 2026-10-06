"""T1 acceptance: real byte IDs and lossless UTF-8 behavior."""

import pytest

from lm_lab.tokenization import ByteTokenizer

pytestmark = pytest.mark.t1


@pytest.mark.parametrize(
    "text",
    ["", "Hello, world!", "中文", "🙂🚀", "👩\u200d💻", "a  b", "  \n\n\t\r\n ", "e\u0301"],
    ids=["empty", "english", "chinese", "emoji", "joined-emoji", "spaces", "whitespace", "combining"],
)
def test_exact_bytes_and_round_trip(text: str) -> None:
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode(text)
    assert ids == list(text.encode("utf-8"))
    assert tokenizer.decode(ids) == text


def test_decode_joins_multibyte_character_before_decoding() -> None:
    assert ByteTokenizer().decode([0xE4, 0xB8, 0xAD]) == "中"


@pytest.mark.parametrize("ids", [[-1], [256], [65, 999]])
def test_invalid_byte_ids_raise_value_error(ids: list[int]) -> None:
    with pytest.raises(ValueError):
        ByteTokenizer().decode(ids)


@pytest.mark.parametrize("ids", [[255], [0xC3]])
def test_invalid_utf8_is_strict_by_default(ids: list[int]) -> None:
    with pytest.raises(UnicodeDecodeError):
        ByteTokenizer().decode(ids)


def test_lossy_decoding_requires_explicit_replace_mode() -> None:
    assert ByteTokenizer().decode([255], errors="replace") == "\ufffd"


@pytest.mark.parametrize("ids", [[], [65]], ids=["empty", "nonempty"])
def test_unsupported_decode_mode_is_rejected(ids: list[int]) -> None:
    with pytest.raises(ValueError):
        ByteTokenizer().decode(ids, errors="ignore")

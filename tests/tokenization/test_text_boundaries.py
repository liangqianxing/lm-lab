"""T4 acceptance: lossless pieces and explicit special-token boundaries."""

import pytest

from lm_lab.tokenization import BPEModel, BPETokenizer, Merge, pretokenize, train_bpe

pytestmark = pytest.mark.t4


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("  \n\t", ["  \n\t"]),
        ("Hi,  世界!\n🙂", ["Hi", ",", "  ", "世界", "!", "\n", "🙂"]),
        ("A_1", ["A", "_", "1"]),
        ("...🙂!", ["...🙂!"]),
        ("e\u0301", ["e", "\u0301"]),
    ],
    ids=["empty", "whitespace", "mixed", "underscore", "symbols", "combining"],
)
def test_lossless_maximal_category_runs(text: str, expected: list[str]) -> None:
    pieces = pretokenize(text)
    assert pieces == expected
    assert "".join(pieces) == text


def test_simple_mode_prevents_cross_piece_training_merges() -> None:
    unrestricted = train_bpe(["a b"], 258, pretokenization="none")
    segmented = train_bpe(["a b"], 258, pretokenization="simple")
    assert unrestricted.vocab[257] == b"a b"
    assert segmented.merges == ()
    assert segmented.pretokenization == "simple"


def test_train_and_encode_share_whitespace_boundaries() -> None:
    model = train_bpe(["  aa"], 258, pretokenization="simple")
    assert model.merges == (Merge((32, 32), 256), Merge((97, 97), 257))
    tokenizer = BPETokenizer(model)
    assert tokenizer.encode("  aa") == [256, 257]
    assert tokenizer.decode([256, 257]) == "  aa"


def test_encoding_uses_stored_pretokenization_even_if_rule_crosses_boundary(
    base_vocab: dict[int, bytes],
) -> None:
    model = BPEModel(
        vocab={**base_vocab, 256: b"a "},
        merges=(Merge((97, 32), 256),),
        pretokenization="simple",
    )
    assert BPETokenizer(model).encode("a ") == [97, 32]


def test_special_tokens_count_toward_total_capacity() -> None:
    model = train_bpe(["abab"], 257, special_tokens=("<eos>",))
    assert model.merges == ()
    assert len(model.vocab) == 257
    assert model.special_tokens == {"<eos>": 256}
    assert model.vocab[256] == b"<eos>"


@pytest.mark.parametrize("specials", [("",), ("<e>", "<e>")])
def test_empty_or_duplicate_special_markers_are_rejected(specials: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        train_bpe(["ab"], 260, special_tokens=specials)


def test_capacity_must_reserve_special_ids() -> None:
    with pytest.raises(ValueError):
        train_bpe(["ab"], 256, special_tokens=("<e>",))


def test_unsupported_pretokenization_is_rejected() -> None:
    with pytest.raises(ValueError):
        train_bpe(["ab"], 257, pretokenization="unknown")


@pytest.fixture
def boundary_model(base_vocab: dict[int, bytes]) -> BPEModel:
    # This rule would consume the first byte of a marker without a boundary.
    return BPEModel(
        vocab={**base_vocab, 256: b"a<", 257: b"<e>"},
        merges=(Merge((97, 60), 256),),
        special_tokens={"<e>": 257},
    )


def test_allowed_special_is_atomic_and_blocks_merges(boundary_model: BPEModel) -> None:
    tokenizer = BPETokenizer(boundary_model)
    assert tokenizer.encode("a<e>a", allowed_special={"<e>"}) == [97, 257, 97]


def test_markers_are_ordinary_text_by_default(boundary_model: BPEModel) -> None:
    assert BPETokenizer(boundary_model).encode("a<e>a") == [256, 101, 62, 97]


def test_adjacent_specials_and_literal_decode(boundary_model: BPEModel) -> None:
    tokenizer = BPETokenizer(boundary_model)
    assert tokenizer.encode("<e><e>", allowed_special={"<e>"}) == [257, 257]
    assert tokenizer.decode([257, 257]) == "<e><e>"


def test_longest_allowed_special_wins_at_same_position(base_vocab: dict[int, bytes]) -> None:
    tokenizer = BPETokenizer(BPEModel(
        vocab={**base_vocab, 256: b"<e>", 257: b"<e><e>"},
        special_tokens={"<e>": 256, "<e><e>": 257},
    ))
    assert tokenizer.encode("<e><e>", allowed_special={"<e>", "<e><e>"}) == [257]
    assert tokenizer.encode("<e><e>", allowed_special={"<e>"}) == [256, 256]


def test_unknown_allowed_special_is_rejected(boundary_model: BPEModel) -> None:
    with pytest.raises(ValueError):
        BPETokenizer(boundary_model).encode("abc", allowed_special={"<unknown>"})


def test_training_never_merges_across_special_markers() -> None:
    model = train_bpe(["ab<e>cd"], 260, special_tokens=("<e>",))
    assert model.merges == (Merge((97, 98), 256), Merge((99, 100), 257))
    assert model.special_tokens == {"<e>": 258}
    tokenizer = BPETokenizer(model)
    assert tokenizer.encode("ab<e>cd", allowed_special={"<e>"}) == [256, 258, 257]
    assert tokenizer.decode([256, 258, 257]) == "ab<e>cd"

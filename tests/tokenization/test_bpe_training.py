"""T2 acceptance: manually verifiable counts, merges, and training."""

import pytest

from lm_lab.tokenization import Merge, count_pairs, merge_pair, train_bpe

pytestmark = pytest.mark.t2


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        ([], {}),
        ([1], {}),
        ([1, 1, 1], {(1, 1): 2}),
        ([1, 2, 1, 2], {(1, 2): 2, (2, 1): 1}),
    ],
)
def test_adjacent_pair_counts(ids: list[int], expected: dict[tuple[int, int], int]) -> None:
    original = ids.copy()
    assert count_pairs(ids) == expected
    assert ids == original


@pytest.mark.parametrize(
    ("ids", "pair", "new_id", "expected"),
    [
        ([], (97, 97), 256, []),
        ([97], (97, 97), 256, [97]),
        ([97, 97, 97], (97, 97), 256, [256, 97]),
        ([97, 97, 97, 97], (97, 97), 256, [256, 256]),
        ([97, 98, 97, 98], (97, 98), 256, [256, 256]),
        ([1, 1, 1], (1, 1), 1, [1, 1]),
    ],
)
def test_non_overlapping_merge(
    ids: list[int], pair: tuple[int, int], new_id: int, expected: list[int]
) -> None:
    original = ids.copy()
    assert merge_pair(ids, pair, new_id) == expected
    assert ids == original


def test_hand_calculated_abab_training(base_vocab: dict[int, bytes]) -> None:
    model = train_bpe(["abab"], 258)
    assert model.vocab == {**base_vocab, 256: b"ab", 257: b"abab"}
    assert model.merges == (Merge((97, 98), 256), Merge((256, 256), 257))
    assert model.pretokenization == "none"
    assert model.special_tokens == {}


@pytest.mark.parametrize("texts", [["abac"], ["ba", "ab"]])
def test_frequency_ties_use_smallest_integer_pair(texts: list[str]) -> None:
    model = train_bpe(texts, 257)
    assert model.merges == (Merge((97, 98), 256),)
    assert model.vocab[256] == b"ab"


def test_counts_are_aggregated_across_texts() -> None:
    assert train_bpe(["ab", "ab", "ba"], 257).merges == (Merge((97, 98), 256),)


def test_no_pair_is_created_across_input_boundaries(base_vocab: dict[int, bytes]) -> None:
    model = train_bpe(["a", "b"], 257)
    assert model.merges == ()
    assert model.vocab == base_vocab


def test_empty_corpus_retains_all_byte_tokens(base_vocab: dict[int, bytes]) -> None:
    model = train_bpe([], 300)
    assert model.vocab == base_vocab
    assert model.merges == ()


def test_vocab_size_is_total_capacity_not_number_of_merges(base_vocab: dict[int, bytes]) -> None:
    model = train_bpe(["abab"], 256)
    assert model.vocab == base_vocab
    assert model.merges == ()


@pytest.mark.parametrize("vocab_size", [-1, 0, 255])
def test_capacity_smaller_than_base_vocabulary_is_rejected(vocab_size: int) -> None:
    with pytest.raises(ValueError):
        train_bpe(["ab"], vocab_size)


def test_same_ordered_corpus_and_configuration_are_deterministic() -> None:
    texts = ["abab", "baba", "aa", ""]
    assert train_bpe(texts, 260) == train_bpe(iter(texts), 260)

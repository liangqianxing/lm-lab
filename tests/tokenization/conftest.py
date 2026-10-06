"""Hand-built model fixtures isolate encoding from unfinished training."""

import pytest

from lm_lab.tokenization import BPEModel, Merge


@pytest.fixture
def base_vocab() -> dict[int, bytes]:
    return {token_id: bytes([token_id]) for token_id in range(256)}


@pytest.fixture
def base_model(base_vocab: dict[int, bytes]) -> BPEModel:
    return BPEModel(vocab=base_vocab)


@pytest.fixture
def ranked_model(base_vocab: dict[int, bytes]) -> BPEModel:
    # On "abc", rank 0 must win over the leftmost available pair at rank 1.
    return BPEModel(
        vocab={**base_vocab, 256: b"bc", 257: b"ab", 258: b"abc"},
        merges=(Merge((98, 99), 256), Merge((97, 98), 257), Merge((97, 256), 258)),
    )


@pytest.fixture
def persisted_model(base_vocab: dict[int, bytes]) -> BPEModel:
    return BPEModel(
        vocab={**base_vocab, 256: b"aa", 257: b"<eos>"},
        merges=(Merge((97, 97), 256),),
        special_tokens={"<eos>": 257},
        pretokenization="simple",
    )

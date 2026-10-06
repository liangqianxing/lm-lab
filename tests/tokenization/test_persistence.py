"""T5 acceptance: portable schema, intact ranks, and deterministic behavior."""

import json
from pathlib import Path

import pytest

from lm_lab.tokenization import BPEModel, BPETokenizer, train_bpe

pytestmark = pytest.mark.t5


@pytest.fixture
def schema_payload(persisted_model: BPEModel) -> dict:
    """Independent file fixture, so load can be tested before save is written."""
    return {
        "format": "lm-lab-bpe",
        "version": 1,
        "pretokenization": "simple",
        "vocab": {str(i): value.hex() for i, value in persisted_model.vocab.items()},
        "merges": [{"pair": [97, 97], "new_id": 256}],
        "special_tokens": {"<eos>": 257},
    }


def test_save_schema_preserves_raw_bytes_and_metadata(
    tmp_path: Path, persisted_model: BPEModel, schema_payload: dict,
) -> None:
    path = tmp_path / "model.json"
    BPETokenizer(persisted_model).save(path)
    assert json.loads(path.read_text(encoding="utf-8")) == schema_payload


def test_load_independently_written_fixture(
    tmp_path: Path, persisted_model: BPEModel, schema_payload: dict,
) -> None:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(schema_payload), encoding="utf-8")
    assert BPETokenizer.load(path).model == persisted_model


def test_save_load_keeps_encoding_decoding_and_special_policy(
    tmp_path: Path, persisted_model: BPEModel,
) -> None:
    original = BPETokenizer(persisted_model)
    path = tmp_path / "词表.json"
    original.save(path)
    loaded = BPETokenizer.load(path)
    text = "aa  中文🙂<eos>aa\n"
    assert loaded.model == persisted_model
    for allowed in [set(), {"<eos>"}]:
        before = original.encode(text, allowed_special=allowed)
        after = loaded.encode(text, allowed_special=allowed)
        assert before == after
        assert original.decode(before) == loaded.decode(after) == text


def test_saving_same_model_is_byte_deterministic(
    tmp_path: Path, persisted_model: BPEModel,
) -> None:
    tokenizer = BPETokenizer(persisted_model)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    tokenizer.save(first)
    tokenizer.save(second)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("field,value", [("version", 999), ("format", "unknown")])
def test_unsupported_schema_is_rejected(
    tmp_path: Path, schema_payload: dict, field: str, value: object,
) -> None:
    schema_payload[field] = value
    path = tmp_path / "unsupported.json"
    path.write_text(json.dumps(schema_payload), encoding="utf-8")
    with pytest.raises(ValueError):
        BPETokenizer.load(path)


def test_merge_with_unavailable_input_id_is_rejected(tmp_path: Path, schema_payload: dict) -> None:
    schema_payload["merges"][0]["pair"] = [97, 99999]
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(schema_payload), encoding="utf-8")
    with pytest.raises(ValueError):
        BPETokenizer.load(path)


def test_incorrect_base_byte_is_rejected(tmp_path: Path, schema_payload: dict) -> None:
    schema_payload["vocab"]["65"] = "00"
    path = tmp_path / "corrupt-base.json"
    path.write_text(json.dumps(schema_payload), encoding="utf-8")
    with pytest.raises(ValueError):
        BPETokenizer.load(path)


def test_missing_model_keeps_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BPETokenizer.load(tmp_path / "missing.json")


def test_full_pipeline_is_deterministic(tmp_path: Path) -> None:
    texts = ["abab  中文\n", "aa🙂<eos>aa"]
    first = BPETokenizer(train_bpe(
        texts, 280, pretokenization="simple", special_tokens=("<eos>",),
    ))
    second = BPETokenizer(train_bpe(
        iter(texts), 280, pretokenization="simple", special_tokens=("<eos>",),
    ))
    assert first.model == second.model
    for text in texts:
        ids = first.encode(text, allowed_special={"<eos>"})
        assert ids == second.encode(text, allowed_special={"<eos>"})
        assert first.decode(ids) == text
    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"
    first.save(first_path)
    second.save(second_path)
    assert first_path.read_bytes() == second_path.read_bytes()

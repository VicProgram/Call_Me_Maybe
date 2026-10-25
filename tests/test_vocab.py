"""Tests for the vocabulary index."""

import json

import pytest

from src.vocab import VocabIndex

VOCAB = {"a": 0, "b": 1, "ab": 2, "123": 3}


@pytest.fixture
def vocab(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(VOCAB), encoding="utf-8")
    return VocabIndex(str(path))


def test_builds_bidirectional_mappings(vocab):
    assert vocab.token_to_id == VOCAB
    assert vocab.id_to_token == {0: "a", 1: "b", 2: "ab", 3: "123"}


def test_search_exact_returns_id_or_empty(vocab):
    assert vocab.search_exact("ab") == [2]
    assert vocab.search_exact("missing") == []


def test_search_characters_only_returns_tokens_without_other_chars(vocab):
    assert set(vocab.search_characters("ab")) == {0, 1, 2}
    assert vocab.search_characters("c") == []
    assert vocab.search_characters("") == []


def test_search_any_character_returns_tokens_with_at_least_one(vocab):
    assert set(vocab.search_any_character("a1")) == {0, 2, 3}


def test_missing_file_raises(vocab, tmp_path):
    with pytest.raises(FileNotFoundError, match="Vocab file not found"):
        VocabIndex(str(tmp_path / "missing.json"))


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid vocab JSON"):
        VocabIndex(str(path))


def test_non_object_json_raises(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(TypeError, match="must be a JSON object"):
        VocabIndex(str(path))

"""Tests for the constrained-decoding helpers and the numeric state machine."""

import json
import math

import numpy as np
import pytest

from src.constrained_decoder import (
    JSONGenerator,
    _first_unescaped_quote,
    apply_mask,
    log_softmax,
    select_best_token,
)
from src.function_caller import value_matches
from src.vocab import VocabIndex

TOKENS = {
    "true": 0,
    "false": 1,
    ".": 2,
    "-": 3,
    "1": 4,
    "2": 5,
    "3": 6,
    " ": 7,
    ",": 8,
    "\n": 9,
    "\t": 10,
    '"': 11,
    "a": 12,
    "b": 13,
}
ORDERED = [None] * len(TOKENS)
for token, token_id in TOKENS.items():
    ORDERED[token_id] = token


class FakeModel:
    """Decode token ids using the small vocabulary above."""

    def decode(self, token_ids):
        return "".join(ORDERED[token_id] for token_id in token_ids)


@pytest.fixture
def generator(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(TOKENS), encoding="utf-8")
    return JSONGenerator(FakeModel(), VocabIndex(str(path)))


def test_generator_finds_digits_and_terminators(generator):
    assert set(generator.digit_ids) == {4, 5, 6}
    assert generator.dot_id == [2]
    assert generator.minus_id == [3]
    assert set(generator.bool_valid_ids) == {0, 1}
    assert 12 in generator.string_valid_ids
    assert 11 in generator.string_terminators


def test_numeric_start_allows_sign_and_digits(generator):
    assert set(generator.numeric_allowed("start", integer=False)) == {
        3, 4, 5, 6
    }


def test_integer_mode_never_allows_the_decimal_point(generator):
    allowed = generator.numeric_allowed("int", integer=True)

    assert 2 not in allowed
    assert set(allowed) >= {4, 5, 6}
    assert set(allowed) >= {7, 8, 9, 10}


def test_number_mode_allows_the_decimal_point(generator):
    assert 2 in generator.numeric_allowed("int", integer=False)


def test_numeric_fraction_state_only_allows_digits_and_terminators(generator):
    assert set(generator.numeric_allowed("dot", integer=False)) == {4, 5, 6}
    assert set(generator.numeric_allowed("frac", integer=False)) == {
        4, 5, 6, 7, 8, 9, 10
    }


def test_numeric_state_transitions(generator):
    assert generator.numeric_next("start", 3) == "sign"
    assert generator.numeric_next("start", 4) == "int"
    assert generator.numeric_next("sign", 4) == "int"
    assert generator.numeric_next("int", 2) == "dot"
    assert generator.numeric_next("int", 4) == "int"
    assert generator.numeric_next("dot", 4) == "frac"
    assert generator.numeric_next("frac", 4) == "frac"


def test_apply_mask_keeps_only_valid_logits():
    masked = apply_mask([1.0, 2.0, 3.0], [0, 2])

    assert masked[0] == 1.0
    assert masked[2] == 3.0
    assert np.isneginf(masked[1])


def test_apply_mask_with_no_valid_tokens_is_all_negative_infinity():
    masked = apply_mask([1.0, 2.0], [])

    assert np.all(np.isneginf(masked))


def test_select_best_token_picks_the_highest_logit():
    assert select_best_token([0.1, 5.0, 2.0]) == 1
    assert select_best_token([-np.inf, 0.0, -np.inf]) == 1


def test_select_best_token_rejects_an_empty_mask():
    with pytest.raises(ValueError, match="No valid token"):
        select_best_token([-np.inf, -np.inf])


def test_log_softmax_returns_normalised_log_probabilities():
    result = log_softmax([1.0, 2.0, 3.0])

    assert len(result) == 3
    assert math.isclose(sum(math.exp(value) for value in result), 1.0)
    assert max(result) <= 0.0


def test_first_unescaped_quote_detects_json_string_end():
    assert _first_unescaped_quote("abc") == -1
    assert _first_unescaped_quote('ab"c') == 2
    assert _first_unescaped_quote(r'a\"b') == -1
    assert _first_unescaped_quote(r'\\"') == 2


@pytest.mark.parametrize(
    ("value", "declared", "expected"),
    [
        ("text", "string", True),
        (1, "string", False),
        (True, "boolean", True),
        (1, "boolean", False),
        (3, "integer", True),
        (True, "integer", False),
        (3.0, "integer", False),
        (3, "number", True),
        (3.5, "number", True),
        (False, "number", False),
    ],
)
def test_value_matches_declared_type(value, declared, expected):
    assert value_matches(value, declared) is expected

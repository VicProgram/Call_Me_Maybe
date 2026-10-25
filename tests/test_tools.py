"""Tests for JSON loading, validation and export."""

import json
from pathlib import Path

import pytest

from src.models import FunctionCall
from src.tools import (
    json_exporter,
    json_reader,
    load_function_def,
    load_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "input"

FUNCTIONS = [
    {
        "name": "fn_add",
        "description": "Add two numbers.",
        "parameters": {"a": {"type": "number"}},
        "returns": {"type": "number"},
    }
]


def write(tmp_path, payload, name="data.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_real_function_definitions():
    functions = load_function_def(str(INPUT / "functions_definition.json"))

    assert functions
    assert "fn_add_numbers" in {function.name for function in functions}
    assert all(function.description for function in functions)


def test_loads_real_prompts():
    prompts = load_prompt(str(INPUT / "function_calling_tests.json"))

    assert prompts
    assert "sum" in prompts[0].prompt.lower()


def test_load_function_def_rejects_non_list(tmp_path):
    path = write(tmp_path, {"name": "fn_add"})

    with pytest.raises(TypeError, match="is not a list"):
        load_function_def(str(path))


def test_load_function_def_rejects_invalid_parameter_type(tmp_path):
    broken = [{**FUNCTIONS[0], "parameters": {"a": {"type": "float"}}}]
    path = write(tmp_path, broken)

    with pytest.raises(ValueError, match="Error processing function"):
        load_function_def(str(path))


def test_load_prompt_accepts_strings_and_objects(tmp_path):
    path = write(tmp_path, ["first", {"prompt": "second"}])

    prompts = load_prompt(str(path))

    assert [prompt.prompt for prompt in prompts] == ["first", "second"]


def test_load_prompt_rejects_other_types(tmp_path):
    path = write(tmp_path, [123])

    with pytest.raises(ValueError, match="is not a string or an object"):
        load_prompt(str(path))


def test_load_prompt_rejects_non_list(tmp_path):
    path = write(tmp_path, {"prompt": "hi"})

    with pytest.raises(TypeError, match="is not an array"):
        load_prompt(str(path))


def test_json_reader_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="was not found"):
        json_reader(tmp_path / "missing.json")


def test_json_reader_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        json_reader(path)


def test_json_exporter_creates_directories_and_roundtrips(tmp_path):
    output = tmp_path / "nested" / "results.json"
    calls = [
        FunctionCall(prompt="Add 2 and 3", name="fn_add", parameters={
            "a": 2, "b": 3
        })
    ]

    json_exporter(calls, output)

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == [
        {
            "prompt": "Add 2 and 3",
            "name": "fn_add",
            "parameters": {"a": 2, "b": 3},
        }
    ]


def test_json_exporter_writes_empty_list(tmp_path):
    output = tmp_path / "empty.json"

    json_exporter([], output)

    assert json.loads(output.read_text(encoding="utf-8")) == []


def test_json_exporter_wraps_os_errors(tmp_path):
    with pytest.raises(ValueError, match="Error creating the output"):
        json_exporter([], tmp_path)

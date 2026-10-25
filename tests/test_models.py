"""Tests for the pydantic models and the decoder state enum."""

import pytest
from pydantic import ValidationError

from src.models import (
    FunctionCall,
    FunctionDefinition,
    JSONState,
    ParameterDefinition,
    ReturnDefinition,
)

FUNCTION = {
    "name": "fn_add",
    "description": "Add two numbers.",
    "parameters": {
        "a": {"type": "number"},
        "b": {"type": "integer"},
    },
    "returns": {"type": "number"},
}


def test_function_definition_accepts_supported_types():
    definition = FunctionDefinition(**FUNCTION)

    assert definition.name == "fn_add"
    assert definition.parameters["a"].type == "number"
    assert definition.parameters["b"].type == "integer"
    assert definition.returns.type == "number"


def test_function_definition_rejects_unknown_parameter_type():
    broken = {
        **FUNCTION,
        "parameters": {"a": {"type": "float"}},
    }

    with pytest.raises(ValidationError):
        FunctionDefinition(**broken)


def test_function_definition_requires_all_fields():
    with pytest.raises(ValidationError):
        FunctionDefinition(name="fn_add")  # type: ignore[call-arg]


def test_parameter_and_return_definitions_are_independent():
    parameter = ParameterDefinition(type="string")
    result = ReturnDefinition(type="boolean")

    assert parameter.type == "string"
    assert result.type == "boolean"


def test_function_call_keeps_original_prompt():
    call = FunctionCall(
        prompt="What is 2 + 3?",
        name="fn_add",
        parameters={"a": 2, "b": 3},
    )

    assert call.prompt == "What is 2 + 3?"
    assert call.parameters == {"a": 2, "b": 3}


def test_json_state_members_are_unique():
    values = [state.value for state in JSONState]

    assert len(values) == len(set(values))
    assert JSONState(JSONState.FINISHED.value) is JSONState.FINISHED

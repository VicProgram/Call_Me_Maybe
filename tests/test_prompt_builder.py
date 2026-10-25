"""Tests for the function-selection prompt."""

from src.models import FunctionDefinition
from src.prompt_builder import fun_select_prompt

FUNCTIONS = [
    FunctionDefinition(
        name="fn_add",
        description="Add two numbers.",
        parameters={"a": {"type": "number"}, "b": {"type": "number"}},
        returns={"type": "number"},
    ),
    FunctionDefinition(
        name="fn_greet",
        description="Greet a person.",
        parameters={"name": {"type": "string"}},
        returns={"type": "string"},
    ),
]


def test_prompt_lists_every_function():
    prompt = fun_select_prompt("Add 2 and 3", FUNCTIONS)

    assert "- fn_add: Add two numbers." in prompt
    assert "- fn_greet: Greet a person." in prompt


def test_prompt_keeps_the_user_request_verbatim():
    request = "Reverse the string 'Hello, World!'"

    assert request in fun_select_prompt(request, FUNCTIONS)


def test_prompt_ends_with_the_selection_cue():
    prompt = fun_select_prompt("Add 2 and 3", FUNCTIONS)

    assert prompt.endswith("Function to call: ")


def test_prompt_without_functions_is_still_valid():
    prompt = fun_select_prompt("Hello", [])

    assert "Hello" in prompt
    assert "available functions are:" in prompt

from pydantic import BaseModel
from enum import Enum, auto
from typing import Literal, TypeAlias


ValueType: TypeAlias = Literal["number", "string", "boolean", "integer"]


class ParameterDefinition(BaseModel):
    """Definition of a function parameter.

    Attributes:
        type: The type of the parameter. Only the types supported by the
            constrained decoder are accepted: "number", "string", "boolean"
            or "integer". Any other value raises a validation error.
    """
    type: ValueType


class ReturnDefinition(BaseModel):
    """Definition of a function's return value.

    Attributes:
        type: The type of the returned value. Only the types supported by the
            constrained decoder are accepted: "number", "string", "boolean"
            or "integer". Any other value raises a validation error.
    """
    type: ValueType


class FunctionDefinition(BaseModel):
    """Definition of a callable function.

    Attributes:
        name: The name of the function.
        description: A human-readable description of what the function does.
        parameters: A dictionary mapping parameter names to their definitions.
        returns: The definition of the function's return type.
    """
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class FunctionCall(BaseModel):
    """Result of a function call extraction.

    Attributes:
        prompt: The original user prompt in natural language.
        name: The name of the selected function.
        parameters: A dictionary mapping argument names to their values.
    """
    prompt: str
    name: str
    parameters: dict[str, object]


class TestPrompt(BaseModel):
    """A test prompt to be processed by the function calling system.

    Attributes:
        prompt: The natural language prompt text.
    """
    prompt: str


class JSONState(Enum):
    FN_NAME = auto()
    AFTER_NAME = auto()
    PARAM_KEY = auto()
    PARAM_VALUE = auto()
    NEXT_PARAM = auto()
    END = auto()
    FINISHED = auto()

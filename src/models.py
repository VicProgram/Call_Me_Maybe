from pydantic import BaseModel
from enum import Enum, auto


class ParameterDefinition(BaseModel):
    """Definition of a function parameter.

    Attributes:
        type: The type of the parameter (e.g., "number", "string", "boolean").
    """
    type: str


class ParameterType(str, Enum):
    """Enumeration of supported parameter types.

    Values:
        NUMBER: Represents a numeric parameter.
        STRING: Represents a string parameter.
    """
    NUMBER = "number"
    STRING = "string"


class FunctionDefinition(BaseModel):
    """Definition of a callable function.

    Attributes:
        name: The name of the function.
        description: A human-readable description of what the function does.
        parameters: A dictionary mapping parameter names to their definitions.
        returns: A dictionary describing the return type of the function.
    """
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: dict[str, str]


class FunctionCall(BaseModel):
    """Result of a function call extraction.

    Attributes:
        prompt: The original user prompt in natural language.
        fn_name: The name of the selected function.
        args: A dictionary mapping argument names to their extracted values.
    """
    prompt: str
    fn_name: str
    args: dict[str, object]


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

from typing import Any
from pydantic import BaseModel, Field
from enum import StrEnum, Enum
from typing import Dict


class State(Enum):
    START = 0
    FN_NAME_KEY = 1
    FN_NAME_VALUE = 2
    ARGS_KEY = 3
    ARG_NAME = 4
    ARG_VALUE = 5
    END = 6


class ParameterType(StrEnum):
    NUMBER = "number"
    STRING = "string"


class ParameterDefinition(BaseModel):
    """Define el tipo de un parámetro de función."""

    type: ParameterType
    description: str | None = None


class FunctionDefinition(BaseModel):
    """Define una función que el sistema puede llamar."""
    name: str
    parameters: Dict[str, ParameterDefinition] = Field(..., default_factory=dict)


class FunctionCall(BaseModel):
    """Representa el resultado de una llamada a función."""
    prompt: str
    fn_name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class TestPrompt(BaseModel):
    prompt: str

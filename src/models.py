from typing import Any
from pydantic import BaseModel, field_validator, Field
from enum import Enum


class ParameterType(Enum):
    NUMBER = "number"
    STRING = "string"


class ParameterDefinition(BaseModel):
    """Define el tipo de un parámetro de función."""

    type: ParameterType
    # description: str | None = None


class FunctionDefinition(BaseModel):
    """Define una función que el sistema puede llamar."""
    name: str
    description: str
    parameters: dict[str, ParameterDefinition] = Field(..., default_factory=dict)
    returns: dict[str, Any]

    @field_validator("parameters", mode="before")
    @classmethod
    def parse_parameters(cls, v: Any) -> dict[str, ParameterDefinition]:
        if isinstance(v, dict):
            return {
            key: ParameterDefinition(**val) if isinstance(val, dict) else
            val for key, val in v.items()
        }

        return v


class FunctionCall(BaseModel):
    """Representa el resultado de una llamada a función."""
    prompt: str
    fn_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class TestPrompt(BaseModel):
    prompt: str

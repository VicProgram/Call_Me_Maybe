from typing import Any
from pydantic import BaseModel, Field
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from enum import StrEnum
from typing import Dict


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


my_model=Small_LLM_Model()


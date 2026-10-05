from pydantic import BaseModel, Field
from typing import Any


class ParameterType(BaseModel):
    NUMBER = "number"
    STRING = "string"


class ParameterDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition] = Field(..., default_factory=dict)
    returns: dict[str, Any]

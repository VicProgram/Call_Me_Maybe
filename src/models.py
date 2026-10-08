from pydantic import BaseModel
from enum import Enum

class ParameterDefinition(BaseModel):
    type: str


class ParameterType(str, Enum):
    NUMBER = "number"
    STRING = "string"


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: dict[str, str]


class FunctionCall(BaseModel):
    prompt: str
    fn_name: str
    args: dict[str, object]


class TestPrompt(BaseModel):
    prompt: str

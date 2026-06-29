from typing import Any
from pydantic import BaseModel, field_validator, Field


class ParameterDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition] = Field(
        ..., default_factory=dict
    )
    returns: dict[str, Any]

    @field_validator("parameters", mode="before")
    @classmethod
    def parse_parameters(
        cls, v: Any
    ) -> dict[str, ParameterDefinition]:
        if isinstance(v, dict):
            return {
                key: ParameterDefinition(**val)
                if isinstance(val, dict) else val
                for key, val in v.items()
            }

        return v


class FunctionCall(BaseModel):
    prompt: str
    fn_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class TestPrompt(BaseModel):
    prompt: str

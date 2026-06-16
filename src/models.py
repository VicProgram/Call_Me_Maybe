from typing import Any
from pydantic import BaseModel
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]



class ParameterDefinition(BaseModel):
    """Define el tipo de un parámetro de función."""


class FunctionDefinition(BaseModel):
    """Define una función que el sistema puede llamar."""


class FunctionCall(BaseModel):
    """Representa el resultado de una llamada a función."""



my_model=Small_LLM_Model()


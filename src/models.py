from typing import Any
from pydantic import BaseModel


class ParameterDefinition(BaseModel):
    """Define el tipo de un parámetro de función."""


class FunctionDefinition(BaseModel):
    """Define una función que el sistema puede llamar."""


class FunctionCall(BaseModel):
    """Representa el resultado de una llamada a función."""
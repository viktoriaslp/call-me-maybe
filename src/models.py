"""Pydantic models for prompts, functions, and outputs."""

from enum import Enum
from typing import Any

from pydantic import BaseModel


class VarType(str, Enum):
    """Allowed parameter and return types."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    INTEGER = "integer"


class FieldDefinition(BaseModel):
    """Definition of one parameter or return value."""

    type: VarType


class FunctionDefinition(BaseModel):
    """Definition of one function available to the model."""

    name: str
    description: str
    parameters: dict[str, FieldDefinition]
    returns: FieldDefinition


class InputPrompt(BaseModel):
    """Input prompt to solve."""

    prompt: str


class OutputResult(BaseModel):
    """Final output generated for one prompt."""

    prompt: str
    name: str
    parameters: dict[str, Any]

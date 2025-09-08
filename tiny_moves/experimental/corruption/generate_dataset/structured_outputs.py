from typing import Literal

from pydantic import BaseModel, Field

AllowedOperation = Literal["replace", "insert_before", "insert_after"]
AllowedErrorType = Literal["wrong_entity", "wrong_direction", "add_unsupported_step"]
AllowedDifficulty = Literal[1, 2]


class Corruption(BaseModel):
    """Represents a single corruption applied to a pathway step."""

    anchor_step_index: int
    operation: AllowedOperation
    error_type: AllowedErrorType
    difficulty: AllowedDifficulty
    original_statement: str | None = None
    corrupted_statement: str
    category_rationale: str = Field(default="")


class ListOfCorruptions(BaseModel):
    """Container for a list of corruptions applied to a pathway."""

    corruptions: list[Corruption]

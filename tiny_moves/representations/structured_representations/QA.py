from typing import Literal

from pydantic import BaseModel


class BinaryAnswer(BaseModel):
    """A reference class to a binary QA answer."""

    answer: Literal["Yes", "No"]


class BinaryAnswerWithConfidence(BaseModel):
    """A reference class to a binary QA answer."""

    answer: Literal["Yes", "No"]
    confidence: float


class TernaryAnswerWithEvidence(BaseModel):
    """A reference class to a ternary QA answer with evidence and a rationale."""

    evidence: str | None
    rationale: str
    input_entities: bool
    output_entities: bool
    directionality: bool
    reaction_type: bool
    answer: Literal["Yes", "No", "Partially"]

import warnings
from typing import Any

from pydantic import BaseModel, Field, field_validator


class References(BaseModel):
    """A reference class to hold a url and and the modality of the referenced literature."""

    url: str
    modality: str


class SupportingEvidence(BaseModel):
    """
    A supporting evidence class to hold an individual supporting evidence.

    A supporting evidence has a summary statement and a list
    of references.
    """

    summary: str
    references: list[References]


class BiologicalProcess(BaseModel):
    """
    A biological process class to hold an individual biological process.

    A biological process has a text representation of the biological process
    """

    biological_process: str
    supporting_evidence: list[SupportingEvidence] = Field(default_factory=list)


class Hypothesis(BaseModel):
    """
    A hypothesis class to hold an individual hypothesis.

    A hypothesis contains:
        hypothesis: a text representation of the full hypothesis.
        supporting_evidence: a list of supporting evidence for the overall hypothesis.
        biological_processes: a list of granular biological processes that detail the biology of the hypothesis.
    """

    hypothesis: str
    biological_processes: list[BiologicalProcess]

    def __eq__(self, other: Any) -> bool:
        """Compare equality with another object based on the hypothesis and its biological processes."""
        if not isinstance(other, Hypothesis):
            return NotImplemented
        return (self.hypothesis, *(x.biological_process for x in self.biological_processes)) == (
            other.hypothesis,
            *(x.biological_process for x in other.biological_processes),
        )

    def __hash__(self) -> int:
        """Return a hash based on the hypothesis and its biological processes."""
        return hash((self.hypothesis, *(x.biological_process for x in self.biological_processes)))

    def __str__(self) -> str:
        """Return a human-readable string summary of the hypothesis and its processes."""
        processes = " ".join(x.biological_process for x in self.biological_processes)
        return f"Hypothesis: {self.hypothesis} | Processes: [{processes}]"

    def to_str(self) -> str:
        """Convert the hypothesis to a string representation."""
        return self.__str__()


class Hypotheses(BaseModel):
    """A Hypotheses class to hold a list of hypotheses."""

    hypotheses: list[Hypothesis]


class HypothesisMove(BaseModel):
    """
    ❌ Deprecated: use `Move[Hypothesis]` instead.

    This class will be removed in a future version.
    """

    move_type: str
    move_index: int
    move_entities: set[str] = set()
    hypothesis: Hypothesis

    @field_validator("*", mode="before")
    @classmethod
    def _warn_deprecated(cls, v: Any) -> Any:
        warnings.warn(
            "`HypothesisMove` is deprecated and will be removed in a future version. "
            "Use `Move[Hypothesis]` from `tiny_moves.trajectory.core` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return v


class HypothesisTrajectory(BaseModel):
    """
    ❌ Deprecated: use `Trajectory` instead.

    This class will be removed in a future version.
    """

    moves: list[HypothesisMove]

    @field_validator("*", mode="before")
    @classmethod
    def _warn_deprecated(cls, v: Any) -> Any:
        warnings.warn(
            "`HypothesisTrajectory` is deprecated and will be removed in a future version. "
            "Use `Trajectory` from `tiny_moves.trajectory.core` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return v

from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel, Field

DIRECTIONAL_PREDICATES = {
    "post transcriptionally modifies",
    "splices",
    "activates",
    "inhibits",
    "modulates",
    "upregulates",
    "downregulates",
    "regulates",
}
SIGNED_PREDICATES = {"activates": "UP", "upregulates": "UP", "inhibits": "DOWN", "downregulates": "DOWN"}

PREDICATES = Literal[
    "binds to",
    "interacts with",
    "post transcriptionally modifies",
    "splices",
    "activates",
    "inhibits",
    "modulates",
    "upregulates",
    "downregulates",
    "regulates",
]


class Triple(BaseModel):
    """A simple representation of a triple in the form of subject, predicate, and object."""

    subject: str
    predicate: PREDICATES
    object: str


class SignedTriple(BaseModel):
    """A triple with additional attributes for directionality and sign of the predicate."""

    triple: Triple
    directional_predicate: bool
    predicate_sign: str


def assign_directionality_and_sign(triple: Triple) -> SignedTriple:
    """
    Assign directionality and sign to a triple.

    Args:
        triple: A Triple object to which directionality and sign will be assigned.

    Returns:
        A SignedTriple object with the original triple and additional attributes.

    """
    # Placeholder logic for assigning directionality and sign
    directional_predicate = triple.predicate in DIRECTIONAL_PREDICATES
    predicate_sign = SIGNED_PREDICATES.get(triple.predicate, "NONE")

    return SignedTriple(triple=triple, directional_predicate=directional_predicate, predicate_sign=predicate_sign)


class ListOfTriples(BaseModel):
    """A list of triples, used to represent a collection of relationships."""

    triples: list[Triple]


class EvaluationResult(BaseModel):
    """A structured output for evaluation results of a candidate hypothesis."""

    reference_triple: SignedTriple
    entities_linked: bool
    predicate_direction: bool | None
    predicate_sign: bool | None
    evidence: list[str] = Field(default_factory=list)


class ListOfEvaluationResults(BaseModel):
    """A list of evaluation results, used to represent multiple evaluations of hypotheses."""

    results: list[EvaluationResult]

    def __len__(self) -> int:
        """Return the number of evaluation results."""
        return len(self.results)

    def __iter__(self) -> Iterator[EvaluationResult]:  # type: ignore[override]
        """Return an iterator over the evaluation results."""
        return iter(self.results)

    def __getitem__(self, index: int) -> EvaluationResult:
        """Get an evaluation result by index."""
        return self.results[index]

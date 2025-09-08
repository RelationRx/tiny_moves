from collections.abc import Iterator

from pydantic import BaseModel


class StatementPair(BaseModel):
    """A pair of statements, one correct and one corrupted, used for evaluation."""

    correct: str
    corrupted: str


class ListOfStatementPairs(BaseModel):
    """A list of statement pairs, used to represent multiple correct-corrupted pairs."""

    pairs: list[StatementPair]

    def __len__(self) -> int:
        """Return the number of statement pairs."""
        return len(self.pairs)

    def __iter__(self) -> Iterator[StatementPair]:  # type: ignore[override]
        """Return an iterator over the statement pairs."""
        return iter(self.pairs)

    def __getitem__(self, index: int) -> StatementPair:
        """Get a statement pair by index."""
        return self.pairs[index]


class EvaluationResult(BaseModel):
    """A structured output for evaluation results of a candidate hypothesis."""

    correct: str
    corrupted: str
    relevant_fragment_from_candidate: str
    score: float
    reason_for_score: str


class ListOfStatementPairEvaluations(BaseModel):
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

from typing import Any

from tiny_moves.metrics.pairwise_hypothesis_metrics.score_errors.find_errors import (
    evaluate_presence_of_errors_in_candidate,
)
from tiny_moves.metrics.pairwise_hypothesis_metrics.score_errors.structured_outputs import (
    ListOfStatementPairEvaluations,
    ListOfStatementPairs,
)
from tiny_moves.metrics.pairwise_hypothesis_metrics.template import (
    PairwiseHypothesisMetric,
)
from tiny_moves.representations.template import ProtocolStr

AMBIGUOUS_SCORE = 0.5


def mean_error_score(results: ListOfStatementPairEvaluations) -> float:
    """
    Mean error score: average across all scores.

    1 = error fully present, 0.5 = ambiguous, 0 = error removed.

    """
    return sum(r.score for r in results) / len(results) if results else 0.0


def clean_rate(results: ListOfStatementPairEvaluations) -> float:
    """Fraction of errors fully removed (score == 0)."""
    return sum(r.score == 0 for r in results) / len(results) if results else 0.0


def partial_error_rate(results: ListOfStatementPairEvaluations) -> float:
    """Fraction of ambiguous / partial errors (score == 0.5)."""
    return sum(r.score == AMBIGUOUS_SCORE for r in results) / len(results) if results else 0.0


def full_error_persistence_rate(results: ListOfStatementPairEvaluations) -> float:
    """Fraction of errors fully persisted (score == 1)."""
    return sum(r.score == 1 for r in results) / len(results) if results else 0.0


def compute_statistics(results: ListOfStatementPairEvaluations) -> dict[str, float]:
    """
    Compute cleaned, partial, full, and average error persistence metrics.

    Returns:
        A dictionary with all relevant metrics.

    """
    return {
        "mean_error_score": mean_error_score(results),
        "clean_rate": clean_rate(results),
        "partial_error_rate": partial_error_rate(results),
        "full_error_persistence_rate": full_error_persistence_rate(results),
    }


class CorruptionMetric(PairwiseHypothesisMetric[ProtocolStr, tuple[Any, Any]]):
    """Evaluates how many statement-level corruptions persist in a candidate hypothesis."""

    def compute(
        self,
        reference_hypothesis: ProtocolStr,
        candidate_hypothesis: ProtocolStr,
        statement_pairs: ListOfStatementPairs,
        seed: int = 42,
    ) -> tuple[ListOfStatementPairEvaluations, dict[str, float]]:
        """
        Evaluate whether corruptions in the hypothesis have been removed.

        Returns:
            - Raw evaluation results per corrupted pair.
            - Summary statistics: corruption rate, clean rate, ambiguity rate.

        """
        list_of_evaluations = evaluate_presence_of_errors_in_candidate(
            seed=seed,
            list_of_statement_pairs=statement_pairs,
            candidate_text=candidate_hypothesis.to_str(),
        )
        stats = compute_statistics(list_of_evaluations)
        return list_of_evaluations, stats

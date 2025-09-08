from typing import Any

from tiny_moves.metrics.compare_against_reference.template import (
    CompareAgainstReferenceMetric,
)
from tiny_moves.metrics.core import HypothesisRepresentation
from tiny_moves.representations.structured_representations.triples import EvaluationResult
from tiny_moves.representations.template import ProtocolStr

from .extract_moa_references_agent import extract_pathway_triples
from .score_candidate_against_reference_agent import (
    evaluate_candidate_against_reference,
)


def score_multiple_candidates_against_reference(
    reference_hypothesis: HypothesisRepresentation,
    candidate_hypotheses: dict[str, HypothesisRepresentation],
    seed: int,
) -> dict[str, list[EvaluationResult]]:
    """
    Score multiple candidate hypotheses against a reference hypothesis.

    Args:
        reference_hypothesis: The reference hypothesis containing ground-truth biological mechanisms.
        candidate_hypotheses: A dictionary of candidate hypotheses to be evaluated against the reference.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A dictionary where keys are candidate names and values are their evaluation results.

    """
    results_dict = {}

    reference_triples = extract_pathway_triples(reference_text=reference_hypothesis.to_str(), seed=seed)

    for candidate_name, candidate in candidate_hypotheses.items():
        candidate_text = candidate.to_str()
        results_dict[candidate_name] = evaluate_candidate_against_reference(
            reference_triples=reference_triples, candidate_text=candidate_text, seed=seed
        ).results

    return results_dict


def count_exists(results: list[EvaluationResult]) -> int:
    """
    Count how many reference triples were identified (exist = 1) in the candidate text.

    Args:
        results: list of evaluation results

    Returns:
        Count of existing triples.

    """
    return sum(r.entities_linked for r in results)


def count_predicate_direction_predictions(results: list[EvaluationResult]) -> int:
    """
    Count how many attempts at predicate direction were made.

    :param results: List of evaluation results.
    :return: Count of directional predicates.
    """
    return sum(isinstance(r.predicate_direction, bool) for r in results)


def count_predicate_sign_predictions(results: list[EvaluationResult]) -> int:
    """
    Count how many attempts at predicate direction were made.

    :param results: List of evaluation results.
    :return: Count of directional predicates.
    """
    return sum(isinstance(r.predicate_sign, bool) for r in results)


def existence_recall(results: list[EvaluationResult]) -> float:
    """
    Compute the recall of existence detection.

    Args:
        results: list of evaluation results

    Returns:
        Existence recall as a float (0.0 - 1.0), representing the proportion of correctly identified triples.

    """
    return count_exists(results) / len(results) if results else 0.0


def predicate_direction_accuracy(results: list[EvaluationResult], relative: bool = False) -> float:
    """
    Compute the f1 score for predicate direction predictions.

    Args:
        results: list of evaluation results
        relative: whether the denominator is all attempted predicates (True) or all existing triples (False).

    Returns:
        Predicate direction f1 across all triples.

    """
    correct = sum(bool(r.predicate_direction) for r in results)

    if relative:
        attempted = count_predicate_direction_predictions(results)
        return correct / attempted if attempted > 0 else 0.0
    else:
        return correct / len(results) if results else 0.0


def predicate_sign_accuracy(results: list[EvaluationResult], relative: bool = False) -> float:
    """
    Compute the f1 score for predicate sign predictions.

    Args:
        results: list of evaluation results
        relative: whether the denominator is all attempted predicates (True) or all existing triples (False).

    Returns:
        Predicate sign f1 across all triples.

    """
    correct = sum(bool(r.predicate_sign) for r in results)
    if relative:
        attempted = count_predicate_sign_predictions(results)
        return correct / attempted if attempted > 0 else 0.0
    else:
        return correct / len(results) if results else 0.0


def mean_total_score_per_triple(results: list[EvaluationResult]) -> float:
    """
    Compute the mean score per triple by averaging existence, predicate, and direction when available.

    Args:
        results: list of evaluation results

    Returns:
        Mean score per triple (0.0 - 1.0) based on existence and predicate score.

    """
    per_triple_scores: list[float] = []

    for r in results:
        scores: list[bool] = [r.entities_linked]

        scores.append(bool(r.predicate_direction))
        scores.append(bool(r.predicate_sign))

        per_triple_score = sum(scores) / len(scores)
        per_triple_scores.append(per_triple_score)

    return sum(per_triple_scores) / len(per_triple_scores) if per_triple_scores else 0.0


def compute_statistics(results: list[EvaluationResult]) -> dict[str, float | None]:
    """
    Compute all core metrics over a set of evaluation results.

    Args:
        results: List of evaluation results.

    Returns:
        A dictionary containing computed statistics for each evaluation method.

    """
    return {
        "mean_total_score_per_triple": mean_total_score_per_triple(results),
        "existence_recall": existence_recall(results),
        "predicate_direction_accuracy_relative": predicate_direction_accuracy(results, relative=True),
        "predicate_direction_accuracy_absolute": predicate_direction_accuracy(results),
        "predicate_sign_accuracy_relative": predicate_sign_accuracy(results, relative=True),
        "predicate_sign_accuracy_absolute": predicate_sign_accuracy(results),
    }


class EvaluateMOAMetric(CompareAgainstReferenceMetric[ProtocolStr, tuple[Any, Any]]):
    """
    A metric to evaluate the performance of a hypothesis discovery system against a reference of biological mechanisms.

    This metric computes various scores based on the existence and direction of biological mechanisms in the candidate
    hypotheses compared to a reference hypothesis.
    """

    def compute(
        self,
        reference_hypothesis: HypothesisRepresentation,
        candidate_hypothesis: dict[str, HypothesisRepresentation],
        seed: int = 42,
    ) -> tuple[Any, Any]:
        """
        Compute the evaluation metrics by comparing candidate hypotheses against a reference hypothesis.

        Args:
            reference_hypothesis: The reference hypothesis containing ground-truth biological mechanisms.
            candidate_hypothesis: A dictionary of candidate hypotheses to be evaluated against the reference.
            seed: Random seed for reproducibility. Defaults to 42.

        Returns:
            A tuple containing:
            - A dictionary of raw counts for each evaluation method.
            - A dictionary of computed statistics for each method.

        """
        raw_counts = score_multiple_candidates_against_reference(
            reference_hypothesis=reference_hypothesis, candidate_hypotheses=candidate_hypothesis, seed=seed
        )

        statistics = {}
        for method in raw_counts:
            statistics[method] = compute_statistics(raw_counts[method])

        return raw_counts, statistics

from typing import Any

from tiny_moves.metrics.compare_against_reference.moa_evaluation.reaction_in_hypothesis_agent import (
    evaluate_candidate_against_reference,
)
from tiny_moves.metrics.compare_against_reference.template import (
    CompareAgainstReferenceMetric,
)
from tiny_moves.metrics.core import HypothesisRepresentation
from tiny_moves.representations.structured_representations.QA import TernaryAnswerWithEvidence
from tiny_moves.representations.template import ProtocolStr


def compute_metrics(results: dict[str, dict[str, TernaryAnswerWithEvidence]]) -> dict[str, dict[str, float]]:
    """
    Compute full and partial recall over a set of statements and answers.

    Args:
        results: a dictionary of method: answers, where the answers are a dict of statement: TernaryAnswers.

    Returns:
        A dictionary containing computed full and partial recall.

    """
    metrics = {}
    for method, answers in results.items():
        total = len(answers)
        if total == 0:
            raise ValueError("no reference statements in evaluation, no metric to compute.")

        yes_count = sum(1 for reference, answer in answers.items() if answer.answer == "Yes")
        partially_count = sum(1 for answer in answers.values() if answer.answer == "Partially")
        input_entity_count = sum(answer.input_entities for answer in answers.values())
        output_entity_count = sum(answer.output_entities for answer in answers.values())
        directionality_count = sum(answer.directionality for answer in answers.values())
        reaction_type_count = sum(answer.reaction_type for answer in answers.values())

        metrics[method] = {
            "full_recall": yes_count / total,
            "partial_recall": (partially_count + yes_count) / total,
            "input_recall": input_entity_count / total,
            "output_recall": output_entity_count / total,
            "directionality_recall": directionality_count / total,
            "reaction_type_recall": reaction_type_count / total,
            "recall_score": (input_entity_count + output_entity_count + directionality_count + reaction_type_count)
            / (4 * total),
        }

    return metrics


def score_candidate_against_reference_statements(
    reference_hypothesis: HypothesisRepresentation,
    candidate_hypothesis: HypothesisRepresentation,
    seed: int,
    reference_splitter: str = "|",
) -> dict[str, TernaryAnswerWithEvidence]:
    """
    Evaluate the candidate hypotheses against the reference hypothesis.

    Args:
        reference_hypothesis: The reference hypothesis containing ground-truth biological mechanisms.
        candidate_hypothesis: A candidate hypotheses to be evaluated against the reference.
        seed: Random seed for reproducibility. Defaults to 42.
        reference_splitter: A string used to split the reference hypothesis into multiple statements. Defaults to '|'.

    Returns:
        A dict of reference statement to answer, as to whether each reference hypothesis statement
        can be found in the hypothesis.

    """
    return {
        reference_statement: evaluate_candidate_against_reference(
            reference_statement, candidate_hypothesis.to_str(), seed
        )
        for reference_statement in reference_hypothesis.to_str().split(reference_splitter)
    }


class HypothesisContainsReferenceStatements(CompareAgainstReferenceMetric[ProtocolStr, tuple[Any, Any]]):
    """Metric to evaluate whether reference statements can be found in a candidate hypothesis."""

    def compute(
        self,
        reference_hypothesis: HypothesisRepresentation,
        candidate_hypothesis: dict[str, HypothesisRepresentation],
        reference_splitter: str = "|",
        seed: int = 42,
    ) -> tuple[Any, Any]:
        """
        Compute the evaluation metrics by comparing candidate hypotheses against a reference hypothesis.

        Reference hypothesis will be split into multiple statements using the provided splitter.

        Args:
            reference_hypothesis: The reference hypothesis containing ground-truth biological mechanisms.
            candidate_hypothesis: A dictionary of candidate hypotheses to be evaluated against the reference.
            reference_splitter: A string used to split the reference hypothesis into multiple statements.
            seed: Random seed for reproducibility. Defaults to 42.

        Returns:
            A tuple containing:
            - A dictionary of raw counts for each evaluation method.
            - A dictionary of computed statistics for each method.

        """
        raw_counts = {
            name: score_candidate_against_reference_statements(
                reference_hypothesis=reference_hypothesis,
                candidate_hypothesis=candidate_hypothesis,
                reference_splitter=reference_splitter,
                seed=seed,
            )
            for name, candidate_hypothesis in candidate_hypothesis.items()
        }

        metrics = compute_metrics(raw_counts)

        return raw_counts, metrics

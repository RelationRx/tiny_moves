from typing import Literal

from tiny_moves.metrics.pairwise_hypothesis_metrics.template import (
    PairwiseHypothesisMetric,
)
from tiny_moves.metrics.utils.named_entity_recognition import (
    extract_and_filter_biomedical_entities,
)
from tiny_moves.representations.template import ProtocolStr


class AddedEntities(PairwiseHypothesisMetric[ProtocolStr, int]):
    """Counts the number of added unique biomedical entities."""

    def compute(
        self,
        reference_hypothesis: ProtocolStr,
        candidate_hypothesis: ProtocolStr,
        type_subsets: list[Literal["HGNC", "MESH", "CHEBI", "GO", "EFO"]] | None = None,
    ) -> int:
        """
        Count the number of added unique biomedical entities.

        Args:
            reference_hypothesis: The reference hypothesis.
            candidate_hypothesis: The candidate hypothesis.
            type_subsets: Optional list of entity type subsets to filter the entities.

        Returns:
            int: The number of unique biomedical entities added in the candidate hypothesis.

        """
        reference_entities = extract_and_filter_biomedical_entities(
            reference_hypothesis.to_str(), type_subsets=type_subsets
        )
        candidate_entities = extract_and_filter_biomedical_entities(
            candidate_hypothesis.to_str(), type_subsets=type_subsets
        )
        added_entities = set(candidate_entities) - set(reference_entities)
        return len(added_entities)


class RemovedEntities(PairwiseHypothesisMetric[ProtocolStr, int]):
    """Counts the number of removed unique biomedical entities."""

    def compute(
        self,
        reference_hypothesis: ProtocolStr,
        candidate_hypothesis: ProtocolStr,
        type_subsets: list[Literal["HGNC", "MESH", "CHEBI", "GO", "EFO"]] | None = None,
    ) -> int:
        """
        Count the number of removed unique biomedical entities.

        Args:
            reference_hypothesis: The reference hypothesis.
            candidate_hypothesis: The candidate hypothesis.
            type_subsets: Optional list of entity type subsets to filter the entities.

        Returns:
            int: The number of unique biomedical entities removed in the candidate hypothesis.

        """
        reference_entities = extract_and_filter_biomedical_entities(
            reference_hypothesis.to_str(), type_subsets=type_subsets
        )
        candidate_entities = extract_and_filter_biomedical_entities(
            candidate_hypothesis.to_str(), type_subsets=type_subsets
        )
        removed_entities = set(reference_entities) - set(candidate_entities)
        return len(removed_entities)


class RecallEntities(PairwiseHypothesisMetric[ProtocolStr, float]):
    """Computes the recall of biomedical entities in the candidate hypothesis compared to the reference hypothesis."""

    def compute(
        self,
        reference_hypothesis: ProtocolStr,
        candidate_hypothesis: ProtocolStr,
        type_subsets: list[Literal["HGNC", "MESH", "CHEBI", "GO", "EFO"]] | None = None,
    ) -> float:
        """
        Compute the recall of biomedical entities in the candidate hypothesis compared to the reference hypothesis.

        Args:
            reference_hypothesis: The reference hypothesis.
            candidate_hypothesis: The candidate hypothesis.
            type_subsets: Optional list of entity type subsets to filter the entities.

        Returns:
            float: The recall score of biomedical entities in the candidate hypothesis.

        """
        reference_entities = extract_and_filter_biomedical_entities(
            reference_hypothesis.to_str(), type_subsets=type_subsets
        )
        candidate_entities = extract_and_filter_biomedical_entities(
            candidate_hypothesis.to_str(), type_subsets=type_subsets
        )

        if not reference_entities:
            return 0.0

        recall_score = len(set(candidate_entities) & set(reference_entities)) / len(set(reference_entities))
        return recall_score


class PrecisionEntities(PairwiseHypothesisMetric[ProtocolStr, float]):
    """Computes the precision of entities in the candidate hypothesis compared to the reference hypothesis."""

    def compute(
        self,
        reference_hypothesis: ProtocolStr,
        candidate_hypothesis: ProtocolStr,
        type_subsets: list[Literal["HGNC", "MESH", "CHEBI", "GO", "EFO"]] | None = None,
    ) -> float:
        """
        Compute the recall of biomedical entities in the candidate hypothesis compared to the reference hypothesis.

        Args:
            reference_hypothesis: The reference hypothesis.
            candidate_hypothesis: The candidate hypothesis.
            type_subsets: Optional list of entity type subsets to filter the entities.

        Returns:
            float: The precision score of biomedical entities in the candidate hypothesis.

        """
        reference_entities = extract_and_filter_biomedical_entities(
            reference_hypothesis.to_str(), type_subsets=type_subsets
        )
        candidate_entities = extract_and_filter_biomedical_entities(
            candidate_hypothesis.to_str(), type_subsets=type_subsets
        )

        if not candidate_entities:
            return 0.0

        precision_score = len(set(candidate_entities) & set(reference_entities)) / len(set(candidate_entities))
        return precision_score


class F1Entities(PairwiseHypothesisMetric[ProtocolStr, float]):
    """Computes the F1 score of biomedical entities in the candidate hypothesis compared to the reference hypothesis."""

    def compute(
        self,
        reference_hypothesis: ProtocolStr,
        candidate_hypothesis: ProtocolStr,
        type_subsets: list[Literal["HGNC", "MESH", "CHEBI", "GO", "EFO"]] | None = None,
    ) -> float:
        """
        Compute the F1 score of biomedical entities in the candidate hypothesis compared to the reference hypothesis.

        Args:
            reference_hypothesis: The reference hypothesis.
            candidate_hypothesis: The candidate hypothesis.
            type_subsets: Optional list of entity type subsets to filter the entities.

        Returns:
            float: The F1 score of biomedical entities.

        """
        reference_entities = extract_and_filter_biomedical_entities(
            reference_hypothesis.to_str(), type_subsets=type_subsets
        )
        candidate_entities = extract_and_filter_biomedical_entities(
            candidate_hypothesis.to_str(), type_subsets=type_subsets
        )

        ref_set = set(reference_entities)
        cand_set = set(candidate_entities)

        if not ref_set and not cand_set:
            return 1.0  # perfect match: both empty

        if not ref_set or not cand_set:
            return 0.0  # one empty, one not

        intersection = ref_set & cand_set
        precision = len(intersection) / len(cand_set)
        recall = len(intersection) / len(ref_set)

        if precision + recall == 0:
            return 0.0

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

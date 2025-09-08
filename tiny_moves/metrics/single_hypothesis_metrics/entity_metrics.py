from typing import Literal

from tiny_moves.metrics.single_hypothesis_metrics.template import (
    SingleHypothesisMetric,
)
from tiny_moves.metrics.utils.named_entity_recognition import (
    extract_and_filter_biomedical_entities,
)
from tiny_moves.representations.template import ProtocolStr


class NumberUniqueEntities(SingleHypothesisMetric[ProtocolStr, int]):
    """A metric that counts the number of unique biomedical entities in a  hypothesis."""

    def compute(
        self, hypothesis: ProtocolStr, type_subsets: list[Literal["HGNC", "MESH", "CHEBI", "GO", "EFO"]] | None = None
    ) -> int:
        """
        Count the number of unique biomedical entities in text.

        Args:
            hypothesis: The hypothesis to analyze.
            type_subsets: Optional list of entity type subsets to filter the entities.

        Returns:
            int: The number of unique biomedical entities found in the hypothesis.

        """
        entities = extract_and_filter_biomedical_entities(hypothesis.to_str(), type_subsets=type_subsets)
        return len(set(entities))

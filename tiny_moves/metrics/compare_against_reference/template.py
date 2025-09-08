from abc import ABC, abstractmethod
from typing import Generic

from tiny_moves.metrics.core import HypothesisRepresentation, MetricResult


class CompareAgainstReferenceMetric(ABC, Generic[HypothesisRepresentation, MetricResult]):
    """Base class for metrics that compare hypothesis against a reference."""

    @abstractmethod
    def compute(
        self,
        reference_hypothesis: HypothesisRepresentation,
        candidate_hypothesis: dict[str, HypothesisRepresentation],
    ) -> MetricResult:
        """Compute the metric by comparing a hypothesis against a reference."""
        pass

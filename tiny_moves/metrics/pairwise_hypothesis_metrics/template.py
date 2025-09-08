from abc import ABC, abstractmethod
from typing import Any, Generic

from tiny_moves.metrics.core import HypothesisRepresentation, MetricResult


class PairwiseHypothesisMetric(ABC, Generic[HypothesisRepresentation, MetricResult]):
    """Base class for comparing two hypothesis to each other."""

    @abstractmethod
    def compute(
        self,
        reference_hypothesis: HypothesisRepresentation,
        candidate_hypothesis: HypothesisRepresentation,
        *args: Any,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute the metric by comparing two hypothesis to each other."""
        pass

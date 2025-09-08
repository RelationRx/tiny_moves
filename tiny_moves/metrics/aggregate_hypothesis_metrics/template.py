from abc import ABC, abstractmethod
from typing import Any, Generic

from tiny_moves.metrics.core import HypothesisRepresentation, MetricResult


class AggregateHypothesisMetric(ABC, Generic[HypothesisRepresentation, MetricResult]):
    """Base class for aggregate hypothesis metrics."""

    @abstractmethod
    def compute(self, hypothesis: list[HypothesisRepresentation], *args: Any, **kwargs: Any) -> MetricResult:
        """Compute the metric by comparing N hypothesis to each other."""
        pass

from abc import ABC, abstractmethod
from typing import Any, Generic

from tiny_moves.metrics.core import HypothesisRepresentation, MetricResult


class SingleHypothesisMetric(ABC, Generic[HypothesisRepresentation, MetricResult]):
    """
    Base class for single hypothesis metrics.

    This class should be inherited by any specific single hypothesis metric implementation.
    """

    @abstractmethod
    def compute(self, hypothesis: HypothesisRepresentation, *args: Any, **kwargs: Any) -> MetricResult:
        """
        Compute the metric based on the provided arguments.

        This method should be implemented by subclasses.
        """
        pass

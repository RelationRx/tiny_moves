from typing import Any, TypeVar

from tiny_moves.representations.template import ProtocolStr

HypothesisRepresentation = TypeVar("HypothesisRepresentation", bound=ProtocolStr)
MetricResult = TypeVar("MetricResult", float, int, str, bool, dict[Any, Any], tuple[Any, Any])

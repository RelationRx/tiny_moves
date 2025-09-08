from tiny_moves.metrics.single_hypothesis_metrics.template import (
    SingleHypothesisMetric,
)
from tiny_moves.representations.structured_representations.hypothesis import (
    Hypothesis,
)


class NumberBiologicalProcesses(SingleHypothesisMetric[Hypothesis, int]):
    """
    Counts the number of biological processes in a structured hypothesis.
    """

    def compute(self, hypothesis: Hypothesis) -> int:
        """Count the number of biological processes in the structured hypothesis."""
        return len(hypothesis.biological_processes)


class NumberUniqueSupportingEvidence(SingleHypothesisMetric[Hypothesis, int]):
    """
    Counts the number of unique supporting evidence entries in a hypothesis.

    This metric returns the deduplicated count of supporting evidence across all biological processes.
    """

    def compute(self, hypothesis: Hypothesis) -> int:
        """Count the unique number of supporting evidence entries in the structured hypothesis."""
        seen = set()
        for bp in hypothesis.biological_processes:
            for ev in bp.supporting_evidence:
                # Deduplicate by (summary, set of reference URLs)
                urls = frozenset(ref.url for ref in ev.references)
                seen.add((ev.summary, urls))
        return len(seen)


class UnsupportedRatio(SingleHypothesisMetric[Hypothesis, float]):
    """Computes ratio of biological processes that are not supported by any evidence."""

    def compute(self, hypothesis: Hypothesis) -> float:
        """Compute the ratio of unsupported biological processes in the hypothesis."""
        if not hypothesis.biological_processes:
            return 0.0
        unsupported_count = sum(1 for bp in hypothesis.biological_processes if not bp.supporting_evidence)
        return unsupported_count / len(hypothesis.biological_processes) if hypothesis.biological_processes else 0.0


class StructuredHypothesisLength(SingleHypothesisMetric[Hypothesis, int]):
    """Computes the length of a structured hypothesis in terms of the number of biological processes."""

    def compute(self, hypothesis: Hypothesis) -> int:
        """Return the number of biological processes in the structured hypothesis."""
        full_hypothesis_str = hypothesis.hypothesis
        words = full_hypothesis_str.split()
        return len(words)

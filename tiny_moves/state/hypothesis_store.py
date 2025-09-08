from tiny_moves.representations.structured_representations.hypothesis import Hypothesis

HypothesisType = Hypothesis | str


class HypothesisStore:
    """A class to manage hypotheses, allowing setting, getting, and tracking changes over time."""

    def __init__(self) -> None:
        """Initialize the HypothesisStore with an empty history."""
        self._history: list[HypothesisType] = list()

    def set(self, new_hypothesis: HypothesisType) -> None:
        """
        Set a new hypothesis and store it in the history.

        Args:
            new_hypothesis: The new hypothesis to set. It can be a Hypothesis object or a string.

        """
        self._history.append(new_hypothesis)

    def get(self) -> HypothesisType:
        """Return the most recent hypothesis."""
        if self._history:
            return self._history[-1]
        return "No hypotheses have been set yet."

    def history(self) -> list[HypothesisType]:
        """Return a copy of the history of hypotheses."""
        return self._history.copy()

    def clear(self) -> None:
        """Clear the current hypothesis and history."""
        self._history = list()

    def __repr__(self) -> str:
        """Get string representation of the HypothesisStore."""
        return f"<HypothesisStore current={'set' if self.get() else 'empty'} versions={len(self._history)}>"


# Create a global shared instance
global_hypothesis_store = HypothesisStore()

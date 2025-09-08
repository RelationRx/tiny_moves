from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolStr(Protocol):
    """Protocol for objects that can be converted to a string representation."""

    def to_str(self) -> str:
        """Convert the object to a string representation."""
        raise NotImplementedError("Subclasses must implement this method.")

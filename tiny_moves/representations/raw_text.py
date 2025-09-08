from pydantic import BaseModel


class RawText(BaseModel):
    """
    An unstructured text class to hold a piece of unstructured text.

    This class is used to represent unstructured text that can be converted to a string.
    """

    text: str

    def to_str(self) -> str:
        """Convert the unstructured text to a string representation."""
        return self.text

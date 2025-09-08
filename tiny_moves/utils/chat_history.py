import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Message:
    """Single chat message with role, content and optional name."""

    content: str
    role: str
    name: str
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def __init__(self, content: str, role: str, name: str = "", **kwargs: dict[str, Any]) -> None:
        """
        Initialize a Message instance.

        Args:
            content: The text content of the message.
            role: The role of the message sender (e.g., "user", "assistant").
            name: Optional name of the sender.
            **kwargs: Additional fields to store in the message.

        """
        self.content = content
        self.role = role
        self.name = name
        self.extra_fields = kwargs


@dataclass
class MessageList:
    """Container for a list of messages, with helper methods."""

    messages: list[Message]

    def __iter__(self) -> Iterator[Message]:
        """Allow iteration over the contained messages."""
        return iter(self.messages)

    @classmethod
    def from_json_file(cls, path: Path) -> "MessageList":
        """Load messages from a JSON file."""
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls([Message(**item) for item in raw])

    @classmethod
    def from_json_files(cls, path: Path) -> "MessageList":
        """Load messages from a directory of JSON files."""
        result = []
        for file in list(path.iterdir()):
            with open(file, encoding="utf-8") as f:
                result.append(json.load(f))
        return cls([Message(**item) for sublist in result for item in sublist])

    def subset(self, predicate: Callable[[Message], bool]) -> "MessageList":
        """Return a new MessageList filtered by a predicate."""
        return MessageList([msg for msg in self.messages if predicate(msg)])

    def to_dict_list(self) -> list[dict[str, Any]]:
        """
        Return a list of dicts representing each Message.

        Values are heterogeneous: core fields are strings, and
        "extra_fields" is a dict[str, Any].
        """
        return [asdict(message) for message in self.messages]

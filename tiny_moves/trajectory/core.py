from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Move(BaseModel, Generic[T]):
    """A single move in a trajectory."""

    move_type: str
    move_index: int
    output: T


class Trajectory(BaseModel, Generic[T]):
    """
    A generic trajectory representation.

    A trajectory is a sequence of moves, where each move can have a output
    of type T. The output can be a Pydantic model or a simple string.
    """

    moves: list[Move[T]]

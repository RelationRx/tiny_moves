from collections.abc import Callable
from typing import Any, TypeVar, cast

from tiny_moves.utils.registry import Registry

ToolFunctionType = TypeVar("ToolFunctionType", bound=Callable[..., Any])


class ToolRegistry(Registry[str, list[ToolFunctionType]]):
    """
    The registry maps tool names to lists of functions specific to that
    tool type. For example, the "literature" tool might have functions
    for searching, summarizing, and analyzing literature.
    """

    pass


def register_function_for_tool(tool: str) -> Callable[[ToolFunctionType], ToolFunctionType]:
    """
    Return a decorator to register a function with the tool registry.

    Args:
        tool: The tool to register the function with.

    Returns:
        A decorator that registers the function with the tool registry.

    """

    def decorator(func: ToolFunctionType) -> ToolFunctionType:
        current_list: list[ToolFunctionType]
        try:
            current_list = cast(list[ToolFunctionType], ToolRegistry.get(tool))
        except ValueError:
            current_list = list()

        current_list.append(func)
        ToolRegistry.register(tool, current_list, overwrite=True)
        return func

    return decorator

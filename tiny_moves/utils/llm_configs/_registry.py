from collections.abc import Callable

from autogen.runtime_logging import LLMConfig
from tiny_moves.utils.registry import Registry

FunctionType = Callable[..., LLMConfig]


class LLMConfigRegistry(Registry[str, FunctionType]):
    """A registry for AutoGen-compatibler LLM configs."""

    pass


# Decorators
def register_llm_config() -> Callable[[FunctionType], FunctionType]:
    """
    Return a decorator to register a file handler function with the handler registry.

    Args:
        handler_registry: The handler registry to register the handler function
            with.

    Returns:
        A decorator that registers the llm config function output with the function name.

    """

    def decorator(llm_config_func: FunctionType) -> FunctionType:
        LLMConfigRegistry.register(llm_config_func.__name__, llm_config_func)
        return llm_config_func

    return decorator

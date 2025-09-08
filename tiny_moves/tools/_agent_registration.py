from collections.abc import Callable
from typing import Any

import autogen
from autogen import ConversableAgent


def register_function_with_doc(func: Callable[..., Any], caller: ConversableAgent, executor: ConversableAgent) -> None:
    """
    Register a function using its docstring as description.

    If the function does not have a docstring, the function name is used instead.

    Args:
        func: The function to register.
        caller: The agent that will call the function.
        executor: The agent that will execute.

    """
    func_description = func.__doc__ if func.__doc__ else func.__name__
    autogen.register_function(func, caller=caller, executor=executor, name=None, description=func_description)


def register_functions_with_docs(
    functions: list[Callable[..., Any]], caller: ConversableAgent, executor: ConversableAgent
) -> None:
    """
    Register a list of functions using their docstrings as descriptions.

    If a function does not have a docstring, the function name is used instead.

    Args:
        functions: The list of functions to register.
        caller: The agent that will call the function.
        executor: The agent that will execute.

    """
    for func in functions:
        register_function_with_doc(func, caller, executor)

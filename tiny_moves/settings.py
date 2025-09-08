
import os
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)

def get_setting(
    env_var: str,
) -> str | None:
    """
    Resolve a setting by preferring an environment variable

    Args:
        env_var: Name of the environment variable to check.
    Returns:
        The resolved setting value or default if not found.

    """
    val = os.getenv(env_var)
    if val:
        return val
    raise ValueError(f"Environment variable {env_var} is not set")


OPENAI_API_KEY = get_setting("OPENAI_API_KEY")


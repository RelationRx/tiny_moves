from enum import Enum
from pathlib import Path


class ConfigSubdir(Enum):
    """Predefined subdirectories for the configuration directory."""

    USER_STORIES = "user_stories"
    AGENTS = "agents"
    REPORTS = "reports"


def get_project_root() -> Path:
    """
    Get the project's root directory.

    Returns:
        The project's root directory.

    """
    return Path(__file__).resolve().parents[2]


def get_configs_dir(subdir: ConfigSubdir | str | None = None) -> Path:
    """
    Get the project's configuration directory.

    Usage:
    ```python
    # Get the root configuration directory
    config_dir = get_configs_dir()

    # Get the configuration directory for user stories
    user_stories_dir = get_configs_dir(ConfigSubdir.USER_STORIES)
    ```

    Args:
        subdir: Optional subdirectory within the configuration
            directory. If a string is provided, it will be appended
            to the configuration directory path. If a ConfigSubdir
            is provided, the corresponding value will be used.

    Returns:
        The path to the project's configuration directory.

    """
    path = get_project_root() / "configs"

    if isinstance(subdir, str):
        path /= subdir
    elif isinstance(subdir, ConfigSubdir):
        path /= subdir.value

    return path


def get_tmp_dir(subdir: str | None = None) -> Path:
    """
    Get the project's tmp directory for temporary data and outputs.

    Args:
        subdir: Optional subdirectory within tmp.

    Returns:
        The absolute path to the temporary directory.

    """
    path = get_project_root() / "tmp"

    if subdir:
        path /= subdir
        path.mkdir(parents=True, exist_ok=True)

    return path

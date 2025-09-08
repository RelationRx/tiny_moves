import argparse
from collections.abc import Callable
from pathlib import Path


def valid_directory(path_str: str) -> Path:
    """
    Validate that the provided path is a directory and return it as a Path object.

    Args:
        path_str (str): The path to validate.

    Returns:
        The validated Path object.

    """
    path = Path(path_str).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
    return path


def valid_file(ext: str | None = None, exists: bool = False) -> Callable[[str], Path]:
    """
    Create a callable to validate that a file has the specified extension.

    Args:
        ext: The file extension to check for.
        exists: Whether the file must exist.

    Returns:
        A callable that validates the file extension and existence.

    """

    def _valid_file(path_str: str) -> Path:
        path = Path(path_str).expanduser().resolve()
        if ext and not path.name.endswith(ext):
            raise argparse.ArgumentTypeError(f"File must have extension '{ext}': {path}")
        if exists and not path.is_file():
            raise argparse.ArgumentTypeError(f"File does not exist: {path}")
        return path

    return _valid_file

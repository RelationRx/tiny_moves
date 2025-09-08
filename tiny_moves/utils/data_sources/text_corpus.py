from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Literal
from zipfile import ZipFile


def read_zip_from_disk(path: str) -> ZipFile:
    """
    Load a ZIP file from disk into memory and return a ZipFile object.

    Args:
        path (str): Path to the ZIP file on disk.

    Returns:
        ZipFile: A ZipFile object representing the contents of the ZIP file.

    """
    with open(path, "rb") as f:
        zip_bytes = BytesIO(f.read())
    return ZipFile(zip_bytes)


def article_contains_string(file_path: Path, query_strings: list[str]) -> bool:
    """
    Check if the article contains a string of interest.

    Args:
        file_path (Path): Path to the file.
        query_strings (list[str]): List of query strings to search for in the article.

    Returns:
        True if the article contains the query string, False otherwise.

    """
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    return any(query in content for query in query_strings)


def multi_threaded_filter_valid_articles(
    file_paths: list[Path], predicate: Callable[[Path], bool], max_workers: int = 20
) -> list[Path]:
    """
    Filter a list of file paths using a predicate function in a multithreaded manner.

    Args:
        file_paths (list[Path]): List of file paths to filter.
        predicate (Callable[[Path], bool]): Predicate function that returns True for valid files.
        max_workers (int): Maximum number of threads to use for filtering.

    Returns:
        list[Path]: List of file paths that do not satisfy the predicate function.

    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda p: not predicate(p), file_paths))

    return [p for p, valid in zip(file_paths, results) if valid]


def write_to_zip(files: list[Path], write_dir: Path, write_mode: Literal["w", "a"] = "w") -> None:
    """
    Write a list of files to a ZIP archive.

    Args:
        files (list[Path]): List of file paths to include in the ZIP archive.
        write_dir (Path): Directory where the ZIP file will be written.
        write_mode (str): Mode for writing the ZIP file, default is 'w' for write.
                          Use 'a' to update an existing archive.

    """
    zip_path = write_dir / "corpus.zip"

    with ZipFile(zip_path, write_mode) as zipf:
        for file_path in files:
            zipf.write(file_path, arcname=file_path.name)

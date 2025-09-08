import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def get_all_file_paths(directory: Path, latest_run_only: bool = False) -> list[Path]:
    """
    Get all file paths in the given directory.

    Args:
        directory: The directory to search for files.
        latest_run_only: If True, return only the latest file for each prefix group.

    Returns:
        List of Path objects representing the files in the directory.

    """
    paths = [p for p in directory.iterdir() if p.is_file()]
    if latest_run_only:
        return get_latest_files_by_prefix(paths)
    return paths


def _parse_timestamp_from_filename(filename: str) -> datetime | None:
    """
    Extract timestamp from a filename in the format PREFIX_YYYY_M_D_H:M:S.json.

    Args:
        filename: The filename string to parse.

    Returns:
        datetime object if timestamp is found, otherwise None.

    """
    try:
        # Match timestamp: e.g. 2025_6_19_10:3:40
        match = re.search(r"(\d{4}_\d{1,2}_\d{1,2}_\d{1,2}:\d{1,2}:\d{1,2})", filename)
        if not match:
            return None
        ts_str = match.group(1)
        return datetime.strptime(ts_str, "%Y_%m_%d_%H:%M:%S")
    except Exception:
        return None


def get_latest_files_by_prefix(paths: list[Path]) -> list[Path]:
    """
    From a list of file paths, return only the latest for each prefix group.

    :param paths: list of Path objects
    :return: Filtered list with only latest files by prefix
    """
    groups = defaultdict(list)

    for path in paths:
        stem = path.stem
        ts = _parse_timestamp_from_filename(stem)
        if ts is None:
            continue
        # Assume everything before the timestamp is the prefix
        prefix = stem.split("_" + ts.strftime("%Y_%m_%d_%H:%M:%S"))[0]
        groups[prefix].append((ts, path))

    latest_files = []
    for file_group in groups.values():
        file_group.sort(reverse=True)
        latest_files.append(file_group[0][1])  # file with latest timestamp

    return latest_files

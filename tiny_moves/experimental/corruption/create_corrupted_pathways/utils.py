from pathlib import Path

import pandas as pd


def load_pathway_file(tsv_path: Path) -> tuple[str, str, list[str]]:
    """
    Read pathway .tsv and extract ID, title, and steps.

    TODO: this is currently a duplication of what is in generate_dataset dir.
    We should create common utils for this.

    """
    df = pd.read_csv(tsv_path, sep="\t")

    if "name" not in df.columns:
        raise ValueError(f"Missing 'name' column in {tsv_path}")

    title = str(df.iloc[0]["name"]).lower().replace(" ", "_")
    steps = [str(x) for x in df["name"].tolist()[1:]]

    if not steps:
        raise ValueError(f"No steps found in {tsv_path} (only a title row present).")

    return tsv_path.stem, title, steps


def write_pathway_file(
    output_path: Path,
    title: str,
    steps: list[str],
) -> pd.DataFrame:
    """
    Reconstruct and write the pathway TSV file from title and steps.

    :param output_path: Path to save the .tsv file
    :param title: Title of the pathway (underscores will be replaced with spaces)
    :param steps: List of pathway step strings (one per row)
    """
    readable_title = title.replace("_", " ")

    all_rows = [readable_title, *steps]
    df = pd.DataFrame({"name": all_rows})

    df.to_csv(output_path, sep="\t", index=False)
    return df


def compute_number_of_errors_per_category(
    fraction_of_errors: float,
    length_of_pathway: int,
    number_of_error_types: int,
    minimum_errors_per_category: int = 1,
) -> int:
    """
    Compute how many errors of *each* category should be injected.

    Args:
        fraction_of_errors: Fraction (0-1] of the pathway length that should become erroneous overall.
        length_of_pathway: Number of steps in the reference pathway.
        number_of_error_types: How many distinct error categories you plan to introduce.
        minimum_errors_per_category: Ensures every category gets at least this many errors
            (useful when `fraction_of_errors * length_of_pathway` would otherwise round down to zero).

    Returns:
        The number of errors to introduce per category.
        Example: If the return value is 3 and you have two categories,
        you will introduce 3 x 2 = 6 errors in total.

    """
    if not (0 < fraction_of_errors <= 1):
        raise ValueError("fraction_of_errors must be in the interval (0, 1].")
    if length_of_pathway <= 0:
        raise ValueError("length_of_pathway must be a positive integer.")
    if number_of_error_types <= 0:
        raise ValueError("number_of_error_types must be a positive integer.")

    requested_total_errors = round(fraction_of_errors * length_of_pathway)
    minimum_total = number_of_error_types * minimum_errors_per_category
    total_errors = max(requested_total_errors, minimum_total)

    errors_per_category = total_errors // number_of_error_types
    return max(errors_per_category, minimum_errors_per_category)

import random
from pathlib import Path

import pandas as pd

from tiny_moves.experimental.corruption.create_corrupted_pathways.utils import (
    compute_number_of_errors_per_category,
    load_pathway_file,
    write_pathway_file,
)


def create_corrupted_pathway_with_metadata(
    reference_pathway: Path,
    corruptions_bank: Path,
    errors_to_introduce: list[str],
    difficulty_level: int,
    fraction_of_errors: float,
    base_directory: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a corrupted pathway and metadata about the applied corruptions.

    Args:
        reference_pathway: Path to the original pathway .tsv file.
        corruptions_bank: Path to the corruption bank .tsv file.
        errors_to_introduce: List of error types to introduce.
        difficulty_level: Difficulty level of the corruptions (1-2).
        fraction_of_errors: Fraction of the pathway that should become erroneous (0-1].
        base_directory: Base directory to save the corrupted pathway and metadata.
        seed: Random seed for reproducibility.

    Returns:
        - DataFrame with the corrupted pathway steps.
        - DataFrame with metadata about applied corruptions.

    """
    save_folder = f"{'_'.join(errors_to_introduce)}_difficulty_{difficulty_level}_fraction_{fraction_of_errors}"
    save_dir = base_directory / save_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    corrupted_pathway_filename = f"{reference_pathway.stem}.tsv"
    corrupted_pathway_metadata = f"{reference_pathway.stem}.metadata.tsv"

    pathway_id, title, steps = load_pathway_file(reference_pathway)
    corruption_bank: pd.DataFrame = pd.read_csv(corruptions_bank, sep="\t")
    corruption_df: pd.DataFrame = corruption_bank.loc[corruption_bank["pathway_id"] == pathway_id]

    number_of_errors_per_category = compute_number_of_errors_per_category(
        fraction_of_errors=fraction_of_errors,
        length_of_pathway=len(steps),
        number_of_error_types=len(errors_to_introduce),
    )
    corrupted_pathway_steps, applied_corruptions_metadata = corrupt_pathway(
        reference_pathway=steps,
        corruption_df=corruption_df,
        errors_to_introduce=errors_to_introduce,
        difficulty_level=difficulty_level,
        number_of_errors_per_category=number_of_errors_per_category,
        seed=seed,
        verbose=True,
    )

    corrupted_pathway = write_pathway_file(
        output_path=save_dir / corrupted_pathway_filename,
        title=title,
        steps=corrupted_pathway_steps,
    )

    applied_corruptions_metadata.to_csv(
        save_dir / corrupted_pathway_metadata,
        sep="\t",
        index=False,
    )

    return corrupted_pathway, applied_corruptions_metadata


def build_corruption_plan(
    errors_to_introduce: list[str],
    difficulty_level: int,
    number_of_errors_per_category: int,
    num_steps: int,
    seed: int = 42,
) -> list[tuple[int, str, int]]:
    """
    Build a corruption plan specifying which steps to corrupt with what errors.

    Returns list of (step_idx, error_type, difficulty)
    """
    total_errors = len(errors_to_introduce) * number_of_errors_per_category

    if total_errors > num_steps:
        raise ValueError(f"Requested {total_errors} corruptions but pathway only has {num_steps} steps.")

    # Create request pool: list of (error_type, difficulty)
    requests = [
        (etype, difficulty_level) for etype in errors_to_introduce for _ in range(number_of_errors_per_category)
    ]

    rng = random.Random(seed)
    rng.shuffle(requests)

    # Sample unique step indices to corrupt (relative to original reference)
    chosen_steps = random.sample(range(num_steps), total_errors)
    plan = list(zip(chosen_steps, requests))  # list of (step_idx, (error_type, difficulty))

    return [(idx, etype, diff) for idx, (etype, diff) in plan]


def apply_corruption_plan(
    reference_pathway: list[str],
    corruption_df: pd.DataFrame,
    plan: list[tuple[int, str, int]],
    seed: int,
    verbose: bool = False,
) -> tuple[list[str], pd.DataFrame]:
    """
    Apply corruptions from the plan to the reference pathway.

    Args:
        reference_pathway: Original pathway steps (list[str]).
        corruption_df: DataFrame containing corruption metadata.
        plan: List of tuples (step_idx, error_type, difficulty) specifying the corruption plan.
        seed: Random seed for reproducibility.
        verbose: If True, print detailed information about each corruption applied.

    Returns:
        - Modified pathway (list[str])
        - Subset of corruption_df with all original columns + applied metadata

    """
    modified = reference_pathway.copy()
    applied_rows = []

    # Sort plan in ascending order by reference step index
    plan_sorted = sorted(plan, key=lambda x: x[0])
    insertion_offset = 0

    for ref_idx, error_type, difficulty in plan_sorted:
        match = corruption_df[
            (corruption_df["anchor_step_index"] == ref_idx)
            & (corruption_df["error_type"] == error_type)
            & (corruption_df["difficulty"] == difficulty)
        ]

        if match.empty:
            raise ValueError(f"No corruption found for step={ref_idx}, type={error_type}, difficulty={difficulty}")

        selected_row = match.iloc[0]
        op = selected_row["operation"]
        original = selected_row["original_statement"]
        corrupted = selected_row["corrupted_statement"]

        mod_idx = ref_idx + insertion_offset

        # Validate replacement against reference (not modified)
        if op == "replace":
            if reference_pathway[ref_idx].strip() != original.strip():
                raise ValueError(
                    f"Validation mismatch at reference step {ref_idx}: "
                    f"Expected {original!r}, got {reference_pathway[ref_idx]!r}"
                )
            if verbose:
                print(f"🔁 REPLACE @ ref={ref_idx}, mod={mod_idx}:")
                print(f"    {original} → {corrupted}")

            modified[mod_idx] = corrupted
            final_idx = mod_idx

        elif op == "insert_before":
            if verbose:
                print(f"⏮ INSERT BEFORE @ ref={ref_idx}, mod={mod_idx}: {corrupted}")
            modified.insert(mod_idx, corrupted)
            insertion_offset += 1
            final_idx = mod_idx

        elif op == "insert_after":
            if verbose:
                print(f"⏭ INSERT AFTER @ ref={ref_idx}, mod={mod_idx + 1}: {corrupted}")
            modified.insert(mod_idx + 1, corrupted)
            insertion_offset += 1
            final_idx = mod_idx + 1

        else:
            raise ValueError(f"Unknown operation: {op}")

        # Convert selected row to dict and enrich it
        row_dict = selected_row.to_dict()

        if op == "replace":
            orig_idx = ref_idx
            orig_txt = reference_pathway[ref_idx]

        else:  # insert_before / insert_after
            orig_idx = None
            orig_txt = None

        row_dict.update(
            {
                "corrupted_step_index": final_idx,
                "original_ref_step_index": orig_idx,
                "original_ref_step_text": orig_txt,
            }
        )
        applied_rows.append(row_dict)

    applied_df = pd.DataFrame(applied_rows).sort_values("corrupted_step_index").reset_index(drop=True)
    applied_df["sampling_seed"] = seed
    return modified, applied_df


def corrupt_pathway(
    reference_pathway: list[str],
    corruption_df: pd.DataFrame,
    errors_to_introduce: list[str],
    difficulty_level: int,
    number_of_errors_per_category: int,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[list[str], pd.DataFrame]:
    """
    Create corrupted pathway according to corruption policy.

    Args:
        reference_pathway: Original pathway steps (list[str]).
        corruption_df: DataFrame containing corruption metadata.
        errors_to_introduce: List of error types to introduce.
        difficulty_level: Difficulty level of the corruptions (1-2).
        number_of_errors_per_category: How many errors of each type to introduce.
        seed: Random seed for reproducibility.
        verbose: If True, print detailed information about each corruption applied.

    Returns:
        - Modified pathway (list[str])
        - Subset of corruption_df with full metadata and modified index references

    """
    plan = build_corruption_plan(
        errors_to_introduce=errors_to_introduce,
        difficulty_level=difficulty_level,
        number_of_errors_per_category=number_of_errors_per_category,
        num_steps=len(reference_pathway),
        seed=seed,
    )

    return apply_corruption_plan(reference_pathway, corruption_df, plan, seed, verbose=verbose)

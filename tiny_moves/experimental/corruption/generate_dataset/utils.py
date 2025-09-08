import itertools
import json
import re
from pathlib import Path
from typing import Any

import json5
import pandas as pd
from json_repair import repair_json

from tiny_moves.experimental.corruption.generate_dataset.structured_outputs import (
    Corruption,
)


def strip_code_fences(content: str) -> str:
    """Remove backtick-wrapped code fences, if present."""
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", content).strip()


def escape_ctrl_chars_in_json_strings(s: str) -> str:
    """Escape control characters inside JSON string values."""
    out = []
    in_str = False
    escape = False
    for ch in s:
        if in_str:
            if escape:
                out.append(ch)
                escape = False
            elif ch == "\\":
                out.append(ch)
                escape = True
            elif ch == '"':
                out.append(ch)
                in_str = False
            elif ord(ch) < 0x20 or ord(ch) == 0x7F:
                if ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                else:
                    out.append(f"\\u{ord(ch):04x}")
            else:
                out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_str = True
    return "".join(out)


def safe_json_loads(s: str) -> Any | None:
    """Safely parse JSON from a string, handling various formats and errors."""
    raw = strip_code_fences(s)  # only once

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    escaped = escape_ctrl_chars_in_json_strings(raw)
    try:
        return json.loads(escaped)
    except json.JSONDecodeError:
        pass

    try:
        return json5.loads(raw)
    except Exception:
        pass

    try:
        repaired = repair_json(raw)
        return json.loads(repaired)
    except Exception:
        pass

    # Debug output
    print("==== BAD JSON ====")
    print(raw)
    print("==================")
    raise ValueError("Failed to parse JSON after multiple repair attempts.")


def render_prompt(title: str, steps: list[str], error_type: str) -> str:
    """Build the user prompt for the LLM to generate all difficulties for all steps for a given error type."""
    step_lines = [f"- [{i}] {step}" for i, step in enumerate(steps)]

    return "\n".join(
        [
            f"Pathway Title: {title}",
            "Pathway Steps:",
            *step_lines,
            "",
            "For the above pathway, generate corruptions for EVERY step (0-based index) "
            f"for the error type: {error_type}.",
            "For each step, produce exactly TWO corruptions:",
            "- One with difficulty = 1",
            "- One with difficulty = 2",
            "",
            "Output them as a JSON object matching the schema given to you in the system message.",
        ]
    )


def validate_and_fix_corruptions(corruptions: list[Corruption], steps: list[str]) -> list[Corruption]:
    """
    Validate generated corruptions and auto-fix mismatches in original_statement for 'replace' operations.

    For 'replace', if original_statement does not match the expected step text, it is replaced with the correct one.
    For 'insert_before' and 'insert_after', original_statement must be None.
    All indices are validated against the provided steps list.

    Args:
        corruptions: List of Corruption objects to validate and fix.
        steps: List of pathway steps corresponding to the corruptions.

    Returns:
        List of validated and fixed Corruption objects.

    """
    for corruption in corruptions:
        idx = corruption.anchor_step_index

        # Ensure index is valid (1-based in the schema, so subtract 1 for list indexing)
        if not (0 <= idx <= len(steps)):
            raise ValueError(f"anchor_step_index {idx} out of bounds. Length of steps is {len(steps)}")

        if corruption.operation == "replace":
            expected = steps[idx].strip()
            actual = (corruption.original_statement or "").strip()
            if expected != actual:
                # Instead of raising, fix the mismatch
                print(
                    f"[validate_and_fix_corruptions] Mismatch at step {idx}: "
                    f"Expected {expected!r}, got {actual!r}. Auto-correcting."
                )
                corruption.original_statement = expected

        elif corruption.operation in {"insert_before", "insert_after"}:
            if corruption.original_statement is not None:
                raise ValueError(
                    f"original_statement must be None for operation '{corruption.operation}' "
                    f"(got {corruption.original_statement!r})."
                )

    return corruptions


def validate_each_step_has_all_combinations_of_corruptions(df: pd.DataFrame) -> None:
    """Ensure each step has all combinations of error types and difficulties."""
    error_types = df["error_type"].unique()
    difficulties = df["difficulty"].unique()

    expected_combinations = set(itertools.product(error_types, difficulties))

    # Group by step
    grouped = df.groupby(["pathway_id", "anchor_step_index"])

    for (pathway_id, anchor_step_index), group in grouped:
        actual_combinations = set(zip(group["error_type"], group["difficulty"]))

        missing_combinations = expected_combinations - actual_combinations
        if missing_combinations:
            raise ValueError(
                f"Step (pathway_id={pathway_id}, anchor_step_index={anchor_step_index}) "
                f"is missing the following corruption combinations: {missing_combinations}"
            )


def load_pathway_file(tsv_path: Path) -> tuple[str, str, list[str]]:
    """Read pathway .tsv and extract ID, title, and steps."""
    df = pd.read_csv(tsv_path, sep="\t")

    if "name" not in df.columns:
        raise ValueError(f"Missing 'name' column in {tsv_path}")

    title = str(df.iloc[0]["name"]).lower().replace(" ", "_")
    steps = [str(x) for x in df["name"].tolist()[1:]]

    if not steps:
        raise ValueError(f"No steps found in {tsv_path} (only a title row present).")

    return tsv_path.stem, title, steps

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from autogen import AssistantAgent

from tiny_moves.utils.llm_configs.llm_configs import (
    chatgpt4o_generate_corruptions,
    gpt4o_structure_corruptions,
)

from .prompt_text import (
    ADD_UNSUPPORTED_STEP_PROMPT,
    WRONG_DIRECTION_PROMPT,
    WRONG_ENTITY_PROMPT,
)
from .structured_outputs import Corruption, ListOfCorruptions
from .utils import (
    load_pathway_file,
    render_prompt,
    safe_json_loads,
    strip_code_fences,
    validate_and_fix_corruptions,
    validate_each_step_has_all_combinations_of_corruptions,
)

ERROR_SYSTEMS: dict[str, str] = {
    "wrong_entity": WRONG_ENTITY_PROMPT,
    "wrong_direction": WRONG_DIRECTION_PROMPT,
    "add_unsupported_step": ADD_UNSUPPORTED_STEP_PROMPT,
}


def call_llm(prompt: str, seed: int, system_message: str) -> ListOfCorruptions:
    """Invoke the LLM agent and parse the response."""
    generator = AssistantAgent(
        name="corruption_generator",
        system_message=system_message,
        llm_config=chatgpt4o_generate_corruptions(seed=seed),
    )
    structuror = AssistantAgent(
        name="corruption_structuror",
        system_message="You are a structured output parser. "
        "Your task is to convert the LLM's response into structured data.",
        llm_config=gpt4o_structure_corruptions(seed=seed),
    )
    response = generator.generate_reply([{"role": "user", "content": prompt}])
    content = response.get("content") if isinstance(response, dict) else response
    data = safe_json_loads(strip_code_fences(content)) if isinstance(content, str) else content

    # structure data
    structured_response = structuror.generate_reply([{"role": "user", "content": str(data)}])
    structured_content = (
        structured_response.get("content") if isinstance(structured_response, dict) else structured_response
    )
    structured_data = (
        safe_json_loads(strip_code_fences(structured_content)) if isinstance(content, str) else structured_content  # type: ignore[arg-type]
    )

    return ListOfCorruptions.model_validate(structured_data)


def build_output_dataframe(
    corruptions: list[Corruption],
    *,
    model_name: str,
    seed: int,
    pathway_id: str,
    pathway_title: str,
    step_count: int,
) -> pd.DataFrame:
    """Convert list of corruptions into output DataFrame."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [
        {
            "corruption_id": str(uuid.uuid4()),
            "created_at": now,
            "model_name": model_name,
            "seed": seed,
            "pathway_id": pathway_id,
            "pathway_title": pathway_title,
            "pathway_step_count": step_count,
            "anchor_step_index": corruption.anchor_step_index,
            "operation": corruption.operation,
            "error_type": corruption.error_type,
            "difficulty": corruption.difficulty,
            "original_statement": corruption.original_statement,
            "corrupted_statement": corruption.corrupted_statement,
            "category_rationale": corruption.category_rationale,
        }
        for corruption in corruptions
    ]
    return pd.DataFrame(rows)


def generate_corruptions_for_pathway(
    tsv_path: Path | str,
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    """
    Generate all possible corruptions for a given pathway.

    Args:
        tsv_path: Path to the pathway .tsv file.
        model_name: Name of the LLM model to use for generation.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing the generated corruptions.

    """
    tsv_path = Path(tsv_path)
    pathway_id, title, steps = load_pathway_file(tsv_path)

    corruption_list: list[Corruption] = []

    for error_type, system_message in ERROR_SYSTEMS.items():
        prompt = render_prompt(
            title,
            steps,
            error_type=error_type,
        )
        parsed_output = call_llm(prompt, seed, system_message)
        corruption_list.extend(parsed_output.corruptions)

    corruption_list = validate_and_fix_corruptions(corruption_list, steps)

    df = build_output_dataframe(
        corruption_list,
        model_name=model_name,
        seed=seed,
        pathway_id=pathway_id,
        pathway_title=title,
        step_count=len(steps),
    )
    validate_each_step_has_all_combinations_of_corruptions(df)
    return df

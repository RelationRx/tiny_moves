import json
from datetime import datetime
from pathlib import Path

from autogen import GroupChatManager

from tiny_moves.state.hypothesis_store import global_hypothesis_store


def make_timestamp_label() -> str:
    """
    Create a timestamp string in the format HH:MM:SS_DD_MM_YYYY.

    Returns:
        A string representing the current timestamp.

    """
    now = datetime.now()
    return (
        "_".join(map(str, [now.year, now.month, now.day]))
        + "_"
        + ":".join(map(str, [now.hour, now.minute, now.second]))
    )


def checkpoint_chat(
    chat_manager: GroupChatManager,
    checkpoint_dir: Path,
    checkpoint_label: str | None = None,
    checkpoint_hypotheses: bool = False,
) -> None:
    """
    Save the chat history as json.

    Contains all necessary information to restart a chat if desired.

    Args:
        chat_manager: The GroupChatManager manager object that stores the chat history
        checkpoint_dir: The directory where the checkpoint file will be saved
        checkpoint_label: arbitrary label for checkpoint file
        checkpoint_hypotheses: Whether to save the hypotheses in a separate file


    """
    timestamp_label = make_timestamp_label()

    checkpoint_filename = timestamp_label if not checkpoint_label else checkpoint_label + "_" + timestamp_label
    checkpoint_path = checkpoint_dir / f"{checkpoint_filename}.json"

    chat_summary = chat_manager.groupchat.messages
    print(f"saving chat to {checkpoint_path}")
    checkpoint_path.write_text(chat_manager.messages_to_string(chat_summary))
    if checkpoint_hypotheses:
        # also save the hypotheses
        hypotheses_path = checkpoint_dir / f"{checkpoint_filename}_hypotheses.json"
        print(f"saving hypotheses to {hypotheses_path}")
        # NB: tech debt - need to update this later to support Hypothesis objects
        hypotheses_path.write_text(json.dumps(global_hypothesis_store.history(), indent=4))

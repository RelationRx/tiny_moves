from collections.abc import Callable
from typing import Any

from autogen import Agent, GroupChat

from tiny_moves.agents.agents import IOMessageTypes, get_chat_io_message


def make_termination_checker(term: str | None) -> Callable[[dict[str, Any]], bool]:
    """
    Create a termination checker function that checks if a specific term is present in the message content.

    Args:
        term: The term to check for in the message content. If None, the checker will always return False.
        If a string is provided, the checker will return True if the term is found in the message content.
        This is used to determine if the chat should be terminated.

    Returns:
        A function that takes a message dictionary and returns True if the term is found in the message content,
        otherwise returns False.

    """

    def checker(msg: dict[str, Any]) -> bool:
        return term is not None and term in msg["content"]

    return checker


def select_next_speaker_from_game_master(last_speaker: Agent, group_chat: GroupChat) -> Agent | str | None:
    """Select the next speaker for the group chat based on the last speaker and the game master."""
    if last_speaker.name != "game_master":
        return _get_agent_with_name("game_master", group_chat.agents)

    last_message = get_chat_io_message(group_chat.messages, IOMessageTypes.latest)

    next_speaker = _parse_game_master_message_for_speaker(last_message, group_chat.agents)

    return next_speaker


def _parse_game_master_message_for_speaker(message: str, agent_list: list[Agent]) -> Agent:
    try:
        agent_name = message.split(":", 1)[0].strip()
        agent = _get_agent_with_name(agent_name, agent_list)

    except Exception:
        print("Failed to parse game master message for speaker. Defaulting to 'game_master'.")
        agent = _get_agent_with_name("game_master", agent_list)

    return agent


def _get_agent_with_name(name: str, agent_list: list[Agent]) -> Agent:
    return next(a for a in agent_list if a.name == name)

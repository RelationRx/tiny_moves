from typing import cast

import autogen
from autogen import GroupChatManager

from tiny_moves.agents.agents import extract_hypothesis
from tiny_moves.state.hypothesis_store import global_hypothesis_store
from tiny_moves.utils.agent_io import load_agent_from_options
from tiny_moves.utils.chat.chat_control import make_termination_checker, select_next_speaker_from_game_master
from tiny_moves.utils.chat.chat_options import ChatConfig
from tiny_moves.utils.checkpoint import checkpoint_chat
from tiny_moves.utils.entry_points.cli_hydra_setup import cli, parse_args
from tiny_moves.utils.entry_points.template_helper import fill_templated_prompts
from tiny_moves.utils.token_usage import log_usage_summary


args, overrides = parse_args()

def main(chat_config: ChatConfig) -> None:
    """
    Run the chat using the specified configuration.

    Args:
        chat_config: The configuration for the chat session

    """
    prompt = create_chat_initiation_prompt(chat_config)
    fill_templated_prompts(chat_config)

    agents = {name: load_agent_from_options(agent_options) for name, agent_options in chat_config.agent_options.items()}
    agent_list = list(agents.values())

    if chat_config.user_proxy:
        user_proxy = load_agent_from_options(chat_config.user_proxy)
        agent_list = [user_proxy, *agent_list]

    if chat_config.chat_manager:
        additional_args = dict(select_speaker_message_template=chat_config.chat_manager.system_message)
    else:
        additional_args = dict(speaker_selection_method=select_next_speaker_from_game_master)  # type: ignore

    groupchat = autogen.GroupChat(
        agents=agent_list,
        messages=[],
        max_round=chat_config.chat_parameters.max_rounds,
        **additional_args,
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=chat_config.chat_manager.llm_config if chat_config.chat_manager else None,
        is_termination_msg=make_termination_checker(chat_config.termination_string),
    )

    if chat_config.chat_history:
        prev_messages = chat_config.chat_history.to_dict_list()

        last_agent, last_message = manager.resume(messages=prev_messages)

        # Resume the chat using the last agent and message
        last_agent.initiate_chat(recipient=manager, message=last_message, clear_history=False)

    else:
        manager.initiate_chat(manager, message=prompt)

    manager = hot_fix_append_final_hypothesis(manager)

    checkpoint_chat(
        manager,
        checkpoint_label=chat_config.checkpoint_prefix,
        checkpoint_dir=chat_config.checkpoint_dir,
        checkpoint_hypotheses=True,
    )

    log_usage_summary(manager)


def create_chat_initiation_prompt(chat_config: ChatConfig) -> str:
    """
    Create a chat initiation prompt based on the chat configuration.

    Append the available agents and the report template to the user prompt.

    Args:
        chat_config: The configuration for the chat session

    Returns:
        The chat initiation prompt to display at the beginning of the chat session

    """
    user_prompt = chat_config.prompt.render()

    # if prompt contains a hypothesis, extract it and store it in the global hypothesis store
    hypothesis = extract_hypothesis(user_prompt)
    if hypothesis:
        print(f"Extracted hypothesis: {hypothesis}")
        global_hypothesis_store.set(hypothesis)

    # add available agent descriptions and report template
    agent_descriptions = {agent.name: agent.description for agent in chat_config.agent_options.values()}  # type: ignore
    user_prompt = (
        "\nAvailable agents are:\n"
        + "\n".join(f"- {name}: {desc}" for name, desc in agent_descriptions.items())
        + "\n"
        + user_prompt
    )

    return user_prompt


def hot_fix_append_final_hypothesis(manager: GroupChatManager) -> GroupChatManager:
    """
    Append the final hypothesis to the last message in the group chat if it does not already exist.

    Note: This is a hot fix and should be removed asap.

    """
    TERMINATION_STRING = "FINAL HYPOTHESIS:"
    final_message = manager.groupchat.messages[-1]

    last_hypothesis = str(global_hypothesis_store.get())
    final_message = TERMINATION_STRING + " " + last_hypothesis
    manager.groupchat.messages.append(
        {"content": final_message, "role": "user", "name": "hot_fix_hypothesis_extractor"}
    )
    return manager


if __name__ == "__main__":
    """Run the main app."""
    chat_config = cli(config_cls=ChatConfig, args=args, overrides=overrides)
    main(chat_config=chat_config)

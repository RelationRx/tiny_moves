from collections.abc import Callable
from pathlib import Path
from typing import Any

from autogen import AssistantAgent, GroupChatManager, UserProxyAgent

from tiny_moves.agents.agent_options import (
    AgentBaseOptions,
    AssistantAgentOptions,
    ClashOfClaimsOptions,
    NestedChatAgentOptions,
    SequentialChatAgentOptions,
    UserProxyAgentOptions,
    WorkerAgentOptions,
)
from tiny_moves.agents.agents import NestedChatAgent, SequentialChatAgent, WorkerAgent
from tiny_moves.tools import ToolRegistry, register_functions_with_docs
from tiny_moves.utils.agent_options_io import load_agent_options_from_yaml


def load_agent_from_yaml(
    yaml_path: str | Path,
) -> UserProxyAgent | AssistantAgent | GroupChatManager | NestedChatAgent | SequentialChatAgent:
    """
    Load and parse a YAML file into an instantiated Agent.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        An instance of UserProxyAgent or AssistantAgent.

    """
    options = load_agent_options_from_yaml(str(yaml_path))
    return load_agent_from_options(options)


def load_agent_from_options(
    options: AgentBaseOptions,
) -> UserProxyAgent | AssistantAgent | GroupChatManager | NestedChatAgent | SequentialChatAgent:
    """
    Convert a Pydantic agent options object into an instantiated Agent.

    Args:
        options: A Pydantic object parsed from YAML.

    Returns:
        An instance of UserProxyAgent or AssistantAgent.

    """
    if isinstance(options, UserProxyAgentOptions):
        return UserProxyAgent(**options.model_dump(exclude={"agent_type"}))

    elif isinstance(options, WorkerAgentOptions):
        agent = WorkerAgent(**options.model_dump(exclude={"agent_type", "tools"}))
        if options.tools:
            for tool in options.tools:
                func2: list[Callable[..., Any]] = ToolRegistry.get(tool)
                register_functions_with_docs(func2, agent, agent)

        return agent
    elif isinstance(options, AssistantAgentOptions):
        agent = AssistantAgent(**options.model_dump(exclude={"agent_type", "tools"}))
        if options.tools:
            for tool in options.tools:
                func: list[Callable[..., Any]] = ToolRegistry.get(tool)
                register_functions_with_docs(func, agent, agent)

        return agent

    elif isinstance(options, NestedChatAgentOptions):
        return _load_nested_chat_from_options(options)
    elif isinstance(options, ClashOfClaimsOptions):
        return _load_clash_of_claims_from_options(options)

    elif isinstance(options, SequentialChatAgentOptions):
        return _load_sequential_chat_from_options(options)

    raise ValueError(f"Unsupported agent type: {type(options).__name__}")


def _load_nested_chat_from_options(
    options: NestedChatAgentOptions,
) -> NestedChatAgent:
    """
    Convert a Pydantic agent options object into an instantiated GroupChat.

    Args:
        options: A Pydantic object parsed from YAML.

    Returns:
        An instance of NestedChat.

    """
    agent_options = options.agents
    agents = {name: load_agent_from_options(options) for name, options in agent_options.items()}

    return NestedChatAgent(
        name=options.name,
        nested_agents=list(agents.values()),
        system_message=options.system_message,
        llm_config=options.llm_config,
        max_round=options.max_round,
        input_message_type=options.input_message_type,
        output_message_type=options.output_message_type,
    )


def _load_sequential_chat_from_options(
    options: SequentialChatAgentOptions,
) -> SequentialChatAgent:
    """
    Convert a Pydantic agent options object into an instantiated GroupChat.

    Args:
        options: A Pydantic object parsed from YAML.

    Returns:
        An instance of SequentialChat.

    """
    agent_options = options.agents
    agents = {name: load_agent_from_options(options) for name, options in agent_options.items()}

    return SequentialChatAgent(
        name=options.name,
        # TODO sorting will only work for 0-9 agent types within a move....
        nested_agents=[agent for _, agent in sorted(agents.items())],
        system_message=options.system_message,
        llm_config=options.llm_config,
        input_message_type=options.input_message_type,
        output_message_type=options.output_message_type,
        inject_hypothesis=options.inject_hypothesis,
        register_final_hypothesis=options.register_final_hypothesis,
    )


def _load_clash_of_claims_from_options(
    options: ClashOfClaimsOptions,
) -> NestedChatAgent:
    """
    Convert a Pydantic agent options object into an instantiated GroupChat.

    Args:
        options: A Pydantic object parsed from YAML.

    Returns:
        An instance of GroupChat.

    """
    claimsmith_options = options.claimsmith_options
    num_claimsmiths = options.num_claimsmiths

    # claimsmith options with identities
    claimsmith_options_with_identities = list()
    for i in range(num_claimsmiths):
        current = claimsmith_options.model_copy()
        # prepend to system message: You are Claimsmith i
        current.system_message = f"You are Claimsmith {i}. " + claimsmith_options.system_message
        current.name = f"{current.name}_{i}"
        claimsmith_options_with_identities.append(current)

    # agents = [load_agent_from_options(claimsmith_options) for _ in range(num_claimsmiths)]
    agents = [load_agent_from_options(options) for options in claimsmith_options_with_identities]

    return NestedChatAgent(
        name=options.name,
        nested_agents=agents,
        system_message=options.system_message,
        llm_config=options.llm_config,
        max_round=options.max_round,
        input_message_type=options.input_message_type,
        output_message_type=options.output_message_type,
    )

    # KEEP THIS FOR FUTURE REFERENCE on register_nested_chats
    # converted_queue = []
    # for step in queue:
    #     recipient = step["recipient"]
    #     assert isinstance(recipient, str), f"Recipient must be a string, got {type(recipient)}"

    #     converted_queue.append({
    #         **step,
    #         "recipient": agents[recipient],
    #     })

    # groupchat = GroupChat(
    #     agents=agents.values(),
    #     messages=[],
    #     max_round=len(converted_queue) + 3,
    #     # speaker_selection_method=make_next_speaker_in_line(speaker_order),
    # )

    # manager = GroupChatManager(
    #     name=options.name,
    #     groupchat=groupchat,
    #     system_message=options.system_message,
    #     llm_config=options.llm_config,
    # )

    # manager.register_nested_chats(converted_queue, trigger=from_game_master) # trigger=lambda _: True)

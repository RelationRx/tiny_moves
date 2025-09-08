import yaml

from tiny_moves.agents.agent_options import (
    AgentBaseOptions,
    AssistantAgentOptions,
    ChatManagerAgentOptions,
    NestedChatAgentOptions,
    UserProxyAgentOptions,
    WorkerAgentOptions,
)


def load_agent_options_from_yaml(yaml_path: str) -> AgentBaseOptions:
    """
    Load and parse a YAML file into a Pydantic options object.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        An instance of UserProxyAgentOptions or AssistantAgentOptions, depending on agent_type.

    """
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        agent_type = data.get("agent_type")

        if agent_type == "UserProxyAgent":
            return UserProxyAgentOptions(**data)
        elif agent_type == "AssistantAgent":
            return AssistantAgentOptions(**data)
        elif agent_type == "ChatManagerAgent":
            return ChatManagerAgentOptions(**data)
        elif agent_type == "GroupChatManagerAgent":
            return NestedChatAgentOptions(**data)
        elif agent_type == "SequentialChatAgent":
            return NestedChatAgentOptions(**data)
        elif agent_type == "WorkerAgent":
            return WorkerAgentOptions(**data)
        else:
            raise ValueError(f"Invalid agent_type: {agent_type}")

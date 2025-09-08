from enum import Enum
from typing import Annotated, Any, Literal, Union

from autogen.runtime_logging import LLMConfig
from pydantic import BaseModel, Field, field_validator

from tiny_moves.tools import ToolRegistry
from tiny_moves.utils.llm_configs import LLMConfigRegistry, StructuredLLmConfig

AgentOptionsUnion = Union[
    "AssistantAgentOptions",
    "UserProxyAgentOptions",
    "NestedChatAgentOptions",
    "SequentialChatAgentOptions",
    "ClashOfClaimsOptions",
    "WorkerAgentOptions",
]


class IOMessageTypes(str, Enum):
    """Enum to help in deciding how to extact messages."""

    summary = "summary"
    latest = "latest"
    all = "all"
    none = "none"


class AgentBaseOptions(BaseModel):
    """Base class for agent configuration."""

    name: str
    system_message: str = ""
    llm_config: LLMConfig | StructuredLLmConfig | None | bool = Field(default_factory=dict["str", Any])

    @field_validator("llm_config", mode="before")
    @classmethod
    def validate_llm_config(cls, value: LLMConfig) -> LLMConfig:
        """
        Create the right llm config object based on the input value.

        If llm_config is a string (e.g., "gpt4o"), load the corresponding configuration
        from the centralised config module. Otherwise, pass it as-is.
        """
        if isinstance(value, str):
            return LLMConfigRegistry.get(value)()

        return value


class UserProxyAgentOptions(AgentBaseOptions):
    """Options for a UserProxyAgent."""

    human_input_mode: str = "ALWAYS"
    code_execution_config: Any
    max_consecutive_auto_reply: int = 5
    is_termination_msg: Any = None
    agent_type: Literal["UserProxyAgent"]

    @field_validator("is_termination_msg", mode="before")
    @classmethod
    def validate_is_termination_msg(cls, value: str) -> Any:
        """If is_termination_msg is a lambda string, safely evaluate it into a callable."""
        if isinstance(value, str) and "lambda" in value:
            try:
                return eval(value)
            except Exception as e:
                raise ValueError(f"Invalid lambda in is_termination_msg: {value}") from e
        return value


class AssistantAgentOptions(AgentBaseOptions):
    """Options for an AssistantAgent."""

    description: str = ""
    tools: list[str] | None = None
    agent_type: Literal["AssistantAgent"]

    @field_validator("tools", mode="before")
    @classmethod
    def validate_tools(cls, value: list[str] | None) -> list[str] | None:
        """
        Validate that all tools in the list are registered in the ToolRegistry.

        Args:
            value: The list of tools to validate.

        Returns:
            The list of tools if all are registered.

        """
        if value is not None:
            for tool in value:
                if tool not in ToolRegistry.get_registered_keys():
                    raise ValueError(f"Tool '{tool}' is not registered in the ToolRegistry.")
        return value


class WorkerAgentOptions(AgentBaseOptions):
    """Options for a WorkerAgent."""

    input_message_type: IOMessageTypes
    output_message_type: IOMessageTypes
    inject_hypothesis: str | None = None
    register_final_hypothesis: bool = False
    description: str = ""
    tools: list[str] | None = None
    agent_type: Literal["WorkerAgent"]

    @field_validator("tools", mode="before")
    @classmethod
    def validate_tools(cls, value: list[str] | None) -> list[str] | None:
        """
        Validate that all tools in the list are registered in the ToolRegistry.

        Args:
            value: The list of tools to validate.

        Returns:
            The list of tools if all are registered.

        """
        if value is not None:
            for tool in value:
                if tool not in ToolRegistry.get_registered_keys():
                    raise ValueError(f"Tool '{tool}' is not registered in the ToolRegistry.")
        return value


class NestedChatAgentOptions(AgentBaseOptions):
    """Options for a NestedChatAgent."""

    agent_type: Literal["NestedChatAgent"]
    description: str
    agents: dict[str, Annotated[AgentOptionsUnion, Field(discriminator="agent_type")]]
    max_round: int
    input_message_type: IOMessageTypes
    output_message_type: IOMessageTypes


class SequentialChatAgentOptions(AgentBaseOptions):
    """Options for a SequentialChatAgentOptions."""

    agent_type: Literal["SequentialChatAgent"]
    description: str
    agents: dict[str, Annotated[AgentOptionsUnion, Field(discriminator="agent_type")]]
    input_message_type: IOMessageTypes
    output_message_type: IOMessageTypes
    inject_hypothesis: str | None = None
    register_final_hypothesis: bool = False


class ClashOfClaimsOptions(AgentBaseOptions):
    """Options for a ClashOfClaims."""

    agent_type: Literal["ClashOfClaims"]
    description: str
    claimsmith_options: AssistantAgentOptions
    max_round: int
    num_claimsmiths: int
    input_message_type: IOMessageTypes
    output_message_type: IOMessageTypes


class ChatManagerAgentOptions(AgentBaseOptions):
    """Options for a ChatManagerAgent."""

    description: str = ""


NestedChatAgentOptions.model_rebuild()
SequentialChatAgentOptions.model_rebuild()

AgentOptionsType = Annotated[
    Union[
        AssistantAgentOptions,
        UserProxyAgentOptions,
        NestedChatAgentOptions,
        ClashOfClaimsOptions,
        SequentialChatAgentOptions,
        WorkerAgentOptions,
    ],
    Field(discriminator="agent_type"),
]

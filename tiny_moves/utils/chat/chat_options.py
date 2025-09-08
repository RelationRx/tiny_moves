from pathlib import Path
from typing import Any

from omegaconf import MISSING
from pydantic import BaseModel, Field, field_validator

# this is to force the .rebuild() method to be called allowing ChatConfig to be resolved
from tiny_moves.agents.agent_options import *  # noqa: F403
from tiny_moves.agents.agent_options import (
    AgentOptionsType,
    AssistantAgentOptions,
    ChatManagerAgentOptions,
    UserProxyAgentOptions,
)
from tiny_moves.utils.chat_history import MessageList
from tiny_moves.utils.paths import get_tmp_dir

DEFAULT_CHECKPOINTS_DIR = get_tmp_dir("checkpoints")


class ChatParameters(BaseModel):
    """Parameters that determine the behavior of the chat."""

    max_rounds: int = Field(..., description="Maximum number of turns in a chat.", gt=0)


class UserPrompt(BaseModel):
    """A user prompt template with placeholders for parameters."""

    prompt: str = Field(..., description="The prompt template with placeholders")
    params: dict[str, Any] = Field(MISSING, description="Parameters to fill in the prompt")

    def render(self) -> str:
        """
        Render the prompt with the provided parameters.

        Returns:
            The rendered prompt.

        """
        if self.params != MISSING:
            return self.prompt.format(**self.params)
        return self.prompt


class ChatConfig(BaseModel):
    """Configuration for a chat session."""

    agent_options: dict[str, AgentOptionsType]
    report: dict[str, Any] | None = None
    chat_parameters: ChatParameters
    chat_manager: ChatManagerAgentOptions | None = None
    user_proxy: UserProxyAgentOptions | None = None
    prompt: UserPrompt
    chat_history: MessageList | None = None
    extra_context: MessageList | None = None
    checkpoint_prefix: str | None = None
    checkpoint_dir: Path = Field(DEFAULT_CHECKPOINTS_DIR)
    termination_string: str | None = None
    log_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chat_history", mode="before")
    def load_chat_history_from_file(cls, f: str | None) -> MessageList | None:
        """Read in chat history if supplied by chat_config.yaml ."""
        if f:
            return MessageList.from_json_file(Path(f))

        return None

    @field_validator("extra_context", mode="before")
    def load_extra_context_from_file(cls, f: str | None) -> MessageList | None:
        """Read in extra context if supplied by chat_config.yaml ."""
        if f:
            path = Path(f)
            if path.is_dir():
                return MessageList.from_json_files(path)
            else:
                return MessageList.from_json_file(Path(f))
        return None

    @field_validator("checkpoint_dir", mode="after")
    def create_checkpoint_dir(cls, v: Path) -> Path:
        """Create the checkpoint directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v



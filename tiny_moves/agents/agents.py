from typing import Any

from autogen import (
    Agent,
    AssistantAgent,
    GroupChat,
    GroupChatManager,
)
from autogen.agentchat import ConversableAgent

from tiny_moves.agents.agent_options import IOMessageTypes
from tiny_moves.state.hypothesis_store import global_hypothesis_store
from tiny_moves.utils.llm_configs import chatgpt_4o_config


def get_chat_io_message(messages: list[dict[str, str]], input_message_type: IOMessageTypes) -> str:
    """Convert the input messages based on the input_message_type."""
    if input_message_type == IOMessageTypes.all:
        return " ".join([x["content"] for x in messages])
    elif input_message_type == IOMessageTypes.latest:
        breakpoint()
        return messages[-1]["content"]
    elif input_message_type == IOMessageTypes.summary:
        summarizer = AssistantAgent(
            name="summariser_agent",
            system_message="You are responsible for summarizing the given message history",
            llm_config=chatgpt_4o_config(),
        )
        # we may not have all the claimsmith messages, just the last from each
        summary = str(summarizer.generate_reply(messages=messages))
        return summary
    elif input_message_type == IOMessageTypes.none:
        return ""

    return ""


def inject_hypothesis(current_message: str, inject_hypothesis: str | None) -> str:
    """
    Inject the current hypothesis into the message if specified.

    Args:
        current_message: The current message to which the hypothesis will be injected.
        inject_hypothesis: The type of hypothesis injection. Can be "latest", "all", or None.

    Returns:
        The modified message with the hypothesis injected, or the original message if no injection is specified.

    """
    if inject_hypothesis is None:
        return current_message

    if inject_hypothesis == "latest":
        latest_hypothesis = global_hypothesis_store.get()
        if latest_hypothesis:
            return f"{current_message}\n\nCurrent hypothesis: {latest_hypothesis}"
    elif inject_hypothesis == "all":
        all_hypotheses = global_hypothesis_store.history()
        if all_hypotheses:
            return f"{current_message}\n\nHypothesis log (old to new):\n" + "\n".join(str(h) for h in all_hypotheses)

    return current_message


def extract_hypothesis(current_message: str) -> str:
    """Extract the hypothesis from the current message."""
    if "current_hypothesis:" in current_message:
        return current_message.split("current_hypothesis:")[-1].strip()
    elif "FINAL HYPOTHESIS:" in current_message:
        return current_message.split("FINAL HYPOTHESIS:")[-1].strip()
    elif "base_hypothesis:" in current_message:
        return current_message.split("base_hypothesis:")[-1].strip()
    return ""


class NestedChatAgent(ConversableAgent):  # type: ignore
    """An agent for handling nested conversations."""

    def __init__(
        self,
        name: str,
        nested_agents: list[Agent],
        input_message_type: IOMessageTypes,
        output_message_type: IOMessageTypes,
        system_message: str = "",
        max_round: int = 10,
        **kwargs: Any,
    ):
        """
        Initialize the NestedChatAgent.

        Args:
            name: The name of the agent.
            nested_agents: A list of agents to be used in the nested conversation.
            input_message_type: the way input messages are summarized before passing to the agent,
            output_message_type: the way the output messages are summarized before passing to the agent,
            system_message: An optional system message for the agent.
            max_round: The maximum number of rounds for the nested conversation.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(name=name, system_message=system_message, **kwargs)
        self.input_message_type = input_message_type
        self.output_message_type = output_message_type
        self.nested_agents = nested_agents
        self.max_round = max_round
        self.register_reply(
            # triger is any agent not in nested_agents
            lambda sender: sender not in self.nested_agents,
            self._run_nested_chat,
        )

    def _run_nested_chat(
        self, agent: Agent, messages: list[dict[str, str]], sender: Agent, config: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Run the nested conversation 'reply'.

        Args:
            agent: The agent that triggered the nested conversation.
            messages: The conversation history.
            sender: The sender of the latest message.
            config: Configuration for the nested chat.

        """
        # Extract the latest message to pass to nested chat
        input_message = get_chat_io_message(messages, self.input_message_type)

        # Create a nested group chat
        group_chat = GroupChat(
            agents=self.nested_agents,
            max_round=self.max_round,
        )
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.llm_config,
            system_message=self.system_message,
            is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
        )

        # Run the nested conversation
        manager.run(f"{input_message}")

        # Extract the final result from the nested conversation
        output_message = get_chat_io_message(group_chat.messages, self.output_message_type)

        # Return this result to the parent conversation
        return True, output_message


class SequentialChatAgent(ConversableAgent):  # type: ignore
    """An agent for a sequential chat conversation."""

    def __init__(
        self,
        name: str,
        nested_agents: list[Agent],
        input_message_type: IOMessageTypes,
        output_message_type: IOMessageTypes,
        inject_hypothesis: str | None,
        register_final_hypothesis: bool,
        system_message: str = "",
        **kwargs: Any,
    ):
        """
        Initialize the SequentialChatAgent.

        Args:
            name: The name of the agent.
            nested_agents: A list of agents to be used in the nested conversation.
            input_message_type: the way input messages are summarized before passing to the agent,
            output_message_type: the way the output messages are summarized before passing to the agent,
            inject_hypothesis: The type of hypothesis injection ("latest", "all", or None).
            register_final_hypothesis: Whether to register the final hypothesis in the global store.
            system_message: An optional system message for the agent.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(name=name, system_message=system_message, **kwargs)
        self.input_message_type = input_message_type
        self.output_message_type = output_message_type
        self.inject_hypothesis = inject_hypothesis
        self.register_final_hypothesis = register_final_hypothesis
        self.nested_agents = nested_agents
        n_structured_agents = len([agent for agent in self.nested_agents if agent.llm_config.response_format])
        n_tool_agents = sum(hasattr(agent, "tools") and bool(agent.tools) for agent in self.nested_agents)
        n_conversational_agents = len(self.nested_agents) - n_structured_agents - n_tool_agents
        self.max_round = 1 + (2 * n_tool_agents) + n_conversational_agents + n_structured_agents
        self.register_reply(
            # triger is any agent not in nested_agents
            lambda sender: sender not in self.nested_agents,
            self._run_nested_chat,
        )

    def _run_nested_chat(
        self, agent: Agent, messages: list[dict[str, str]], sender: Agent, config: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Run the sequential conversation 'reply'.

        Args:
            agent: The agent that triggered the nested conversation.
            messages: The conversation history.
            sender: The sender of the latest message.
            config: Configuration for the nested chat.

        """
        # Extract full message history for sufficient context
        input_message = get_chat_io_message(messages, self.input_message_type)
        input_message = inject_hypothesis(
            current_message=input_message,
            inject_hypothesis=self.inject_hypothesis,
        )

        # Create a nested group chat
        group_chat = GroupChat(
            agents=self.nested_agents, max_round=self.max_round, speaker_selection_method="round_robin"
        )
        manager = GroupChatManager(groupchat=group_chat, llm_config=self.llm_config, system_message=self.system_message)

        # Run the nested conversation
        manager.run(f"{input_message}")
        # Extract the final result from the nested conversation
        output_message = get_chat_io_message(group_chat.messages, self.output_message_type)

        # Register the final hypothesis if required
        if self.register_final_hypothesis:
            hypothesis = extract_hypothesis(output_message)
            if hypothesis:
                global_hypothesis_store.set(hypothesis)
            # else:
            #     # for now, let's kill process if no hypothesis is found.
            #     assert False, "No hypothesis found in the output message."

        # Return this result to the parent conversation
        return True, output_message


class WorkerAgent(AssistantAgent):  # type: ignore
    """An agent for handling worker tasks in a chat conversation."""

    def __init__(
        self,
        name: str,
        input_message_type: IOMessageTypes,
        output_message_type: IOMessageTypes,
        inject_hypothesis: str | None,
        register_final_hypothesis: bool,
        system_message: str = "",
        **kwargs: Any,
    ):
        """Initialize the WorkerAgent."""
        super().__init__(name=name, system_message=system_message, **kwargs)

        # store config like you did in SequentialChatAgent
        self.input_message_type = input_message_type
        self.output_message_type = output_message_type
        self.inject_hypothesis = inject_hypothesis
        self.register_final_hypothesis = register_final_hypothesis

    def generate_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any] | None:
        """Generate a reply based on the provided messages and sender."""
        if messages is None and sender is not None:
            messages = self._oai_messages[sender]

        # 1) preprocess
        if messages:
            last_message = get_chat_io_message(messages, IOMessageTypes.latest)
            last_message = inject_hypothesis(last_message, self.inject_hypothesis)

            # 2) build a working copy of messages with the augmented input
            messages[-1]["content"] = last_message

            if self.input_message_type == IOMessageTypes.latest:
                messages = [messages[-1]]
            if self.input_message_type == IOMessageTypes.none:
                messages = [messages[-1].copy()]
                messages[-1]["content"] = ""

        # 3) call the default assistant generation on the modified list
        reply: str | dict[str, Any] | None = super().generate_reply(messages, sender, **kwargs)

        # 4) postprocess
        if self.register_final_hypothesis and isinstance(reply, (str, dict)):
            if isinstance(reply, str) or (isinstance(reply, dict) and reply.get("content") is not None):
                text = reply if isinstance(reply, str) else reply.get("content", "")
                hypothesis = extract_hypothesis(text)
                if hypothesis:
                    global_hypothesis_store.set(hypothesis)

        return reply

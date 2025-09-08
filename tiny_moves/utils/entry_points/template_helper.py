import jinja2

from tiny_moves.utils.chat.chat_options import ChatConfig


def fill_templated_prompts(chat_config: ChatConfig) -> None:
    """
    Update the system messages of agents with templated prompts.

    Args:
        chat_config: The configuration for the chat session

    """
    user_prompt = chat_config.prompt.render()

    # add available agent descriptions and report template
    agent_descriptions = {
        agent.name: agent.description  # type: ignore
        for agent in chat_config.agent_options.values()
        if agent.name != "game_master"
    }
    moves = "\nAvailable agents are:\n" + "\n".join(f"- {name}: {desc}" for name, desc in agent_descriptions.items())

    context = {"moves": moves, "user_prompt": user_prompt}
    for agent in chat_config.agent_options.values():
        if agent.name == "game_master":
            for curr_agent in agent.agents.values():  # type: ignore
                environment = jinja2.Environment()
                template = environment.from_string(curr_agent.system_message)
                system_message = template.render(**context)
                curr_agent.system_message = system_message

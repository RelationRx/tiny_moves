import yaml
from autogen import GroupChatManager, gather_usage_summary


def log_usage_summary(manager: GroupChatManager) -> None:
    """
    Print the token usage (actual excluding cache and total).

    Will print the token usage for each model used in the GroupChat.

    Entries under `usage_excluding_cached_inferenece` are the actual tokens used for the chat,
    excluding those tokens that openAI automatically caches. See
    https://platform.openai.com/docs/guides/prompt-caching for details.

    Args:
        manager: The GroupChatManager instance for the chat

    """
    usage_summary = gather_usage_summary(manager._groupchat.agents)
    print(yaml.dump(usage_summary, allow_unicode=True, default_flow_style=False))

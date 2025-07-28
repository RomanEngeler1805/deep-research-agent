from openai import OpenAI
import json
import os
from typing import List, Dict, Any


def generate_completion_with_tools(
    messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], model: str = "gpt-4o"
) -> Any:
    """
    Generate a completion with tools using OpenAI API.

    Args:
        messages: List of conversation messages
        tools: List of available tools
        model: Model to use for completion

    Returns:
        OpenAI completion response
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if tools:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
    else:
        return client.chat.completions.create(
            model=model,
            messages=messages,
        )


def truncate_messages(
    messages: List[Dict[str, Any]], max_tokens: int = 120000
) -> List[Dict[str, Any]]:
    """
    Truncate messages to avoid context length issues.

    Args:
        messages: List of conversation messages
        max_tokens: Maximum tokens to keep (rough estimate)

    Returns:
        Truncated messages list
    """
    if not messages:
        return messages

    # Always keep the system message if it exists
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    other_messages = [msg for msg in messages if msg.get("role") != "system"]

    # Rough token estimation (4 chars ≈ 1 token)
    total_chars = sum(len(json.dumps(msg)) for msg in messages)
    estimated_tokens = total_chars // 4

    if estimated_tokens <= max_tokens:
        return messages

    # Keep recent messages, remove from the middle
    keep_recent = min(10, len(other_messages))
    truncated_others = other_messages[-keep_recent:] if other_messages else []

    return system_messages + truncated_others

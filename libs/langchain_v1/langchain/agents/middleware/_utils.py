def _generate_correction_tool_messages(content: str, tool_calls: list):
    tool_messages = []
    for tool_call in tool_calls:
        tool_messages.append({
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call["id"]
        })
    return tool_messages

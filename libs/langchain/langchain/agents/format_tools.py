def format_tools_with_description(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
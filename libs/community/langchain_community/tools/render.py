from langchain_community.tools.convert_to_openai import (
    format_tool_to_openai_function,
    format_tool_to_openai_tool,
)


# backward compatibility
def __getattr__(name):
    if name == "format_tool_to_openai_function":
        return format_tool_to_openai_function
    if name == "format_tool_to_openai_tool":
        return format_tool_to_openai_tool

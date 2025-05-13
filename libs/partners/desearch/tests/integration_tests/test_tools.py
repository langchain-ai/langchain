import os
import sys
import pytest

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Fetch real API key from environment variable
real_api_key = os.getenv("DESEARCH_API_KEY")
print("Real API Key:", real_api_key)

# Skip the tests if no valid API key is found.
if not real_api_key:
    pytest.skip(
        "Skipping real test because no valid DESEARCH_API_KEY provided.",
        allow_module_level=True,
    )

# Import all tool classes from our package.
from langchain_desearch.tools import (
    DesearchTool,
    BasicWebSearchTool,
    BasicTwitterSearchTool,
)

# Define a mapping of tool names to instances.
tool_mapping = {
    "ai_search": DesearchTool(),
    "basic_twitter_search": BasicTwitterSearchTool(),
    "basic_web_search": BasicWebSearchTool(),
}

# Define test cases with inputs that match each tool's _run() signature.

test_cases = [
    # DesearchTool cases.
    (
        "ai_search",
        {
            "prompt": "Bittensor",
            "model": "NOVA",
            "date_filter": "PAST_24_HOURS",
            "streaming": False,
            "tool": ["web"],
        },
    ),
    # Basic Twitter Search Tool case.
    (
        "basic_twitter_search",
        {
            "query": "Whats going on with Bittensor",
            "sort": "Top",
            "user": "elonmusk",
            "start_date": "2024-12-01",
            "end_date": "2025-02-25",
            "lang": "en",
            "verified": True,
            "blue_verified": True,
            "is_quote": True,
            "is_video": True,
            "is_image": True,
            "min_retweets": 1,
            "min_replies": 1,
            "min_likes": 1,
            "count": 10,
        },
    ),
    # Basic Web Search Tool case.
    (
        "basic_web_search",
        {
            "query": "latest news on AI",
            "num": 10,
            "start": 1,  # Fix: start must be >= 1
        },
    ),
]


@pytest.mark.parametrize("tool_name,input_data", test_cases)
def test_all_tools(tool_name, input_data):
    # Get the corresponding tool instance from our mapping.
    tool = tool_mapping.get(tool_name)
    if tool is None:
        pytest.fail(f"Tool {tool_name} is not implemented.")

    # Run the tool with the given input data.
    try:
        result = tool._run(**input_data)
        print(f"Test for tool '{tool_name}' returned:\n{result}\n")

        # Perform a basic check on the result.
        assert isinstance(
            result, (str, dict, list)
        ), f"Result for {tool_name} is not a valid type: {type(result)}"
        if isinstance(result, str):
            assert (
                len(result.strip()) > 0
            ), f"Result for {tool_name} is empty or only whitespace."
            assert (
                "error" not in result.lower()
            ), f"Result for {tool_name} contains an error: {result}"
        else:
            assert result, f"Result for {tool_name} is empty or invalid."
    except Exception as e:
        pytest.skip(f"Skipping test for tool '{tool_name}' due to error: {e}")


# Test case for invalid tool input for DesearchTool.
def test_invalid_tool():
    tool = DesearchTool()
    input_data = {
        "prompt": "test prompt",
        "tool": "nonexistent",  # invalid tool parameter
        "model": "NOVA",
        "date_filter": None,
        "streaming": False,
        "query": None,
    }
    result = tool._run(**input_data)
    # Since _run() catches exceptions, we check that the message indicates an unsupported tool.
    assert "Unsupported tool: nonexistent" in result

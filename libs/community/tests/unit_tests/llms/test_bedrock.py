import json
from typing import AsyncGenerator, Dict
from unittest.mock import MagicMock, patch

import pytest

from langchain_community.llms.bedrock import (
    ALTERNATION_ERROR,
    Bedrock,
    _human_assistant_format,
)

TEST_CASES = {
    """Hey""": """

Human: Hey

Assistant:""",
    """

Human: Hello

Assistant:""": """

Human: Hello

Assistant:""",
    """Human: Hello

Assistant:""": """

Human: Hello

Assistant:""",
    """
Human: Hello

Assistant:""": """

Human: Hello

Assistant:""",
    """

Human: Human: Hello

Assistant:""": (
        "Error: Prompt must alternate between '\n\nHuman:' and '\n\nAssistant:'."
    ),
    """Human: Hello

Assistant: Hello

Human: Hello

Assistant:""": """

Human: Hello

Assistant: Hello

Human: Hello

Assistant:""",
    """

Human: Hello

Assistant: Hello

Human: Hello

Assistant:""": """

Human: Hello

Assistant: Hello

Human: Hello

Assistant:""",
    """

Human: Hello

Assistant: Hello

Human: Hello

Assistant: Hello

Assistant: Hello""": ALTERNATION_ERROR,
    """

Human: Hi.

Assistant: Hi.

Human: Hi.

Human: Hi.

Assistant:""": ALTERNATION_ERROR,
    """
Human: Hello""": """

Human: Hello

Assistant:""",
    """

Human: Hello
Hello

Assistant""": """

Human: Hello
Hello

Assistant

Assistant:""",
    """Hello

Assistant:""": """

Human: Hello

Assistant:""",
    """Hello

Human: Hello

""": """Hello

Human: Hello



Assistant:""",
    """

Human: Assistant: Hello""": """

Human: 

Assistant: Hello""",
    """

Human: Human

Assistant: Assistant

Human: Assistant

Assistant: Human""": """

Human: Human

Assistant: Assistant

Human: Assistant

Assistant: Human""",
    """
Assistant: Hello there, your name is:

Human.

Human: Hello there, your name is: 

Assistant.""": """

Human: 

Assistant: Hello there, your name is:

Human.

Human: Hello there, your name is: 

Assistant.

Assistant:""",
    """

Human: Human: Hi

Assistant: Hi""": ALTERNATION_ERROR,
    """Human: Hi

Human: Hi""": ALTERNATION_ERROR,
    """

Assistant: Hi

Human: Hi""": """

Human: 

Assistant: Hi

Human: Hi

Assistant:""",
    """

Human: Hi

Assistant: Yo

Human: Hey

Assistant: Sup

Human: Hi

Assistant: Hi
Human: Hi
Assistant:""": """

Human: Hi

Assistant: Yo

Human: Hey

Assistant: Sup

Human: Hi

Assistant: Hi

Human: Hi

Assistant:""",
    """

Hello.

Human: Hello.

Assistant:""": """

Hello.

Human: Hello.

Assistant:""",
}


def test__human_assistant_format() -> None:
    for input_text, expected_output in TEST_CASES.items():
        if expected_output == ALTERNATION_ERROR:
            with pytest.warns(UserWarning, match=ALTERNATION_ERROR):
                _human_assistant_format(input_text)
        else:
            output = _human_assistant_format(input_text)
            assert output == expected_output


# Sample mock streaming response data
MOCK_STREAMING_RESPONSE = [
    {"chunk": {"bytes": b'{"text": "nice"}'}},
    {"chunk": {"bytes": b'{"text": " to meet"}'}},
    {"chunk": {"bytes": b'{"text": " you"}'}},
]


async def async_gen_mock_streaming_response() -> AsyncGenerator[Dict, None]:
    for item in MOCK_STREAMING_RESPONSE:
        yield item


@pytest.mark.asyncio
async def test_bedrock_async_streaming_call() -> None:
    # Mock boto3 import
    mock_boto3 = MagicMock()
    session = MagicMock()
    session.region_name = "region"
    mock_boto3.Session.return_value = session
    mock_boto3.Session.return_value.client.return_value = (
        session  # Mocking the client method of the Session object
    )

    with patch.dict(
        "sys.modules", {"boto3": mock_boto3}
    ):  # Mocking boto3 at the top level using patch.dict
        # Mock the `Bedrock` class's method that invokes the model
        mock_invoke_method = MagicMock(return_value=async_gen_mock_streaming_response())
        with patch.object(
            Bedrock, "_aprepare_input_and_invoke_stream", mock_invoke_method
        ):
            # Instantiate the Bedrock LLM
            llm = Bedrock(
                client=None,
                model_id="anthropic.claude-v2",
                streaming=True,
            )
            # Call the _astream method
            chunks = [
                json.loads(chunk["chunk"]["bytes"])["text"]  # type: ignore
                async for chunk in llm._astream("Hey, how are you?")
            ]

    # Assertions
    assert len(chunks) == 3
    assert chunks[0] == "nice"
    assert chunks[1] == " to meet"
    assert chunks[2] == " you"

from langchain.chat_models.fake import FakeListChatModel
from langchain.schema import HumanMessage


def test_fake_chat_model_open_ai_functions():
    """Test fake chat model."""
    response1 = {
        "name": "get_current_weather",
        "arguments": (
            "{" \
            '"temperature": 25,' \
            '"unit": "celsius",' \
            '"location": "San Francisco, CA"' \
            "}"
        ),
    }
    response2 = {
        "name": "get_current_weather2",
        "arguments": (
            "{" \
            '"temperature": 25,' \
            '"unit": "celsius",' \
            '"location": "San Francisco, CA"' \
            "}"
        ),
    }
    model = FakeListChatModel(responses=[
        response1,
        response2,
    ])
    assert model(
        [HumanMessage(content="What is the weather?")]
    ).additional_kwargs["function_call"] == response1
    assert model(
        [HumanMessage(content="What is the weather?")]
    ).additional_kwargs["function_call"] == response2
    assert model(
        [HumanMessage(content="What is the weather?")]
    ).additional_kwargs["function_call"] == response1

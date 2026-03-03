import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from langchain.agents import create_agent


# This is a mock ChatModel used for testing streaming output
class FakeStreamingModel(FakeListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
async def test_create_agent_streaming_config_propagation():
    """Test if create_agent correctly propagates the RunnableConfig.

    If the fix is effective, we should receive 'on_chat_model_stream' events.
    """
    responses = ["Step 1", "Final Answer"]
    model = FakeStreamingModel(responses=responses)

    def dummy_tool(x: str) -> str:
        """This is a dummy tool that simulates some processing."""
        return "dummy result"

    agent = create_agent(
        model=model,
        tools=[dummy_tool],
    )

    seen_stream_events = False

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="Hello")]}, version="v2"
    ):
        event_type = event["event"]

        if event_type == "on_chat_model_stream":
            seen_stream_events = True

    assert seen_stream_events, (
        "Failed: 'on_chat_model_stream' event not captured. Config not propagated."
    )

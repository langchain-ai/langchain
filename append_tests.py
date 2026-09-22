with open('libs/core/tests/unit_tests/language_models/test_compat_bridge.py', 'a') as f:
    f.write("""

def test_chunks_to_events_parallel_tool_calls_no_index() -> None:
    \"\"\"Parallel tool_calls in separate no-index chunks survive.

    Regression for langchain-ai/langchain#40392.
    \"\"\"
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.messages import AIMessageChunk
    from langchain_core.language_models.chat_models import _chunks_to_events

    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {
                        "type": "tool_call",
                        "id": "tc1",
                        "name": "get_weather",
                        "args": {"city": "San Francisco"},
                    }
                ]
            )
        ),
        ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {
                        "type": "tool_call",
                        "id": "tc2",
                        "name": "get_weather",
                        "args": {"city": "Seattle"},
                    }
                ]
            )
        ),
    ]

    events = list(_chunks_to_events(chunks))
    
    # We should have two events, one for each tool call
    assert len(events) > 0
    # Final event
    final_event = events[-1]
    msg = final_event["message"]
    
    # We should have both tool calls
    assert len(msg.tool_calls) == 2
    assert msg.tool_calls[0]["id"] == "tc1"
    assert msg.tool_calls[1]["id"] == "tc2"

async def test_achunks_to_events_parallel_tool_calls_no_index() -> None:
    \"\"\"Async parallel tool_calls in separate no-index chunks survive.\"\"\"
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.messages import AIMessageChunk
    from langchain_core.language_models.chat_models import _achunks_to_events

    async def _async_gen():
        chunks = [
            ChatGenerationChunk(
                message=AIMessageChunk(
                    content=[
                        {
                            "type": "tool_call",
                            "id": "tc1",
                            "name": "get_weather",
                            "args": {"city": "San Francisco"},
                        }
                    ]
                )
            ),
            ChatGenerationChunk(
                message=AIMessageChunk(
                    content=[
                        {
                            "type": "tool_call",
                            "id": "tc2",
                            "name": "get_weather",
                            "args": {"city": "Seattle"},
                        }
                    ]
                )
            ),
        ]
        for c in chunks:
            yield c

    events = [e async for e in _achunks_to_events(_async_gen())]
    
    assert len(events) > 0
    final_event = events[-1]
    msg = final_event["message"]
    
    assert len(msg.tool_calls) == 2
    assert msg.tool_calls[0]["id"] == "tc1"
    assert msg.tool_calls[1]["id"] == "tc2"
""")

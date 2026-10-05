from __future__ import annotations

import unittest
from typing import Any

from langchain.voice.audio import resample_pcm16
from langchain.voice.providers.gemini_live import gemini_task_tool
from langchain.voice.providers.openai_realtime import (
    openai_task_tools,
    session_config,
    truncate_playback,
)
from langchain.voice.transport import PlaybackReceipt


class InterruptedTransport:
    def __init__(self) -> None:
        self.cleared = False

    async def interrupt_output(self) -> PlaybackReceipt:
        self.cleared = True
        return PlaybackReceipt(
            stream_id="item-1",
            content_index=0,
            played_samples=1_200,
            sample_rate=24_000,
        )


class ConversationItem:
    def __init__(self) -> None:
        self.truncations: list[dict[str, Any]] = []

    async def truncate(self, **kwargs: Any) -> None:
        self.truncations.append(kwargs)


class Connection:
    def __init__(self) -> None:
        self.conversation = type("Conversation", (), {})()
        self.conversation.item = ConversationItem()


class FakeFunctionDeclaration:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeTool:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeGeminiTypes:
    class Behavior:
        NON_BLOCKING = "non-blocking"
        BLOCKING = "blocking"

    FunctionDeclaration = FakeFunctionDeclaration
    Tool = FakeTool


class ProviderContractTests(unittest.TestCase):
    def test_openai_session_owns_coordination_and_manual_turns(self) -> None:
        config = session_config("assembled instructions")
        turn_detection = config["audio"]["input"]["turn_detection"]

        assert [tool["name"] for tool in openai_task_tools()] == [
            "create_task",
            "update_task",
            "cancel_task",
        ]
        assert not turn_detection["create_response"]
        assert turn_detection["interrupt_response"]
        assert config["parallel_tool_calls"]

    def test_gemini_registers_the_same_coordination_contract(self) -> None:
        tool = gemini_task_tool(FakeGeminiTypes)

        assert [declaration.name for declaration in tool.function_declarations] == [
            "create_task",
            "update_task",
            "cancel_task",
        ]
        for declaration in tool.function_declarations:
            assert not declaration.parameters_json_schema["additionalProperties"]

    def test_resamples_pcm_without_provider_dependencies(self) -> None:
        source = b"\x00\x00" * 240

        result = resample_pcm16(source, 24_000, 16_000)

        assert len(result) == 320


class InterruptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_truncates_openai_context_to_audio_actually_played(self) -> None:
        transport: Any = InterruptedTransport()
        connection = Connection()

        item_id = await truncate_playback(connection, transport)

        assert item_id == "item-1"
        assert transport.cleared
        assert connection.conversation.item.truncations == [
            {"item_id": "item-1", "content_index": 0, "audio_end_ms": 50}
        ]


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import asyncio
import unittest
from typing import TYPE_CHECKING, Any

import pytest

from langchain import voice
from langchain.voice import (
    GeminiLiveConversationLayer,
    TransportDisconnected,
    create_voice_agent,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from langchain.voice.events import VoiceEvent


async def wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    async def poll() -> None:
        while not predicate():  # noqa: ASYNC110 - cooperative test polling
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout)


class ImmediateGraph:
    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, str]:
        del config
        return {"response": f"answer: {state['messages'][0]['content']}"}


class BlockingGraph:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> None:
        del state, config
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class FakeLiveConversation:
    def __init__(self, instructions: str = "Be brief.") -> None:
        self.instructions = instructions
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    async def run(self, session: Any, **kwargs: Any) -> None:
        self.calls.append((session, kwargs))


class FakeTransport:
    output_sample_rate = 24_000
    output_active = False

    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def events(self) -> AsyncIterator[Any]:
        yield TransportDisconnected()

    async def send_audio(
        self,
        frame: Any,
        *,
        stream_id: str | None = None,
        content_index: int = 0,
    ) -> None:
        del frame, stream_id, content_index

    async def wait_output_idle(self) -> None:
        return

    async def interrupt_output(self) -> None:
        return

    async def aclose(self) -> None:
        self.closed = True


class LegacyAudioInput:
    sample_rate = 16_000

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    async def frames(self) -> AsyncIterator[bytes]:
        yield b"\x00\x00"

    def stop(self) -> None:
        self.stopped = True


class LegacyAudioOutput:
    sample_rate = 24_000

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.callback: Any = None

    def start(self) -> None:
        self.started = True

    def write(self, data: bytes) -> None:
        del data

    def buffered_bytes(self) -> int:
        return 0

    def clear(self) -> None:
        return

    def set_played_callback(self, callback: Any) -> None:
        self.callback = callback

    def stop(self) -> None:
        self.stopped = True


class VoiceSessionTests(unittest.IsolatedAsyncioTestCase):
    def test_top_level_api_only_exposes_application_concepts(self) -> None:
        assert set(voice.__all__) == {
            "AudioFormat",
            "AudioFrame",
            "AudioIOTransport",
            "AudioInput",
            "AudioOutput",
            "AudioPlayed",
            "AudioReceived",
            "ConversationLayer",
            "GeminiLiveConversationLayer",
            "LiveKitAudioTransport",
            "LocalAudioTransport",
            "NullUI",
            "OpenAIRealtimeConversationLayer",
            "PlaybackReceipt",
            "StatusUI",
            "TransportDisconnected",
            "UserSpeechEnded",
            "UserSpeechStarted",
            "VoiceAgent",
            "VoiceTransport",
            "create_voice_agent",
        }

    async def test_run_delegates_to_the_configured_conversation_layer(self) -> None:
        conversation = FakeLiveConversation()
        agent = create_voice_agent(ImmediateGraph(), conversation=conversation)
        transport = FakeTransport()

        await agent.run(
            transport=transport,
            project_name="voice-project",
        )

        assert len(conversation.calls) == 1
        session, kwargs = conversation.calls[0]
        assert session.instructions == agent.instructions
        assert kwargs["transport"] is transport
        assert kwargs["project_name"] == "voice-project"
        assert transport.started
        assert transport.closed
        with pytest.raises(RuntimeError, match="closed"):
            await session.create_task("work after shutdown")

    async def test_legacy_audio_pair_is_adapted_and_closed(self) -> None:
        conversation = FakeLiveConversation()
        agent = create_voice_agent(ImmediateGraph(), conversation=conversation)
        audio_in = LegacyAudioInput()
        audio_out = LegacyAudioOutput()

        await agent.run(audio_in=audio_in, audio_out=audio_out)

        transport = conversation.calls[0][1]["transport"]
        assert isinstance(transport, voice.AudioIOTransport)
        assert audio_in.started
        assert audio_in.stopped
        assert audio_out.started
        assert audio_out.stopped
        assert audio_out.callback is None

    async def test_in_process_send_and_event_iterator(self) -> None:
        agent = create_voice_agent(
            ImmediateGraph(),
            conversation=FakeLiveConversation(),
        )
        session = agent._create_session()
        events = session.events()

        ready = await anext(events)
        await session.send({"type": "task.create", "instruction": "find a flight"})
        emitted: list[VoiceEvent] = []
        while not any(event.type == "task.completed" for event in emitted):
            emitted.append(await anext(events))

        assert ready.type == "session.ready"
        assert any(event.type == "task.created" for event in emitted)
        assert any(event.type == "task.completed" for event in emitted)
        await session.aclose()

    async def test_assembles_framework_and_conversation_instructions(self) -> None:
        conversation = FakeLiveConversation(instructions="Sound warm and specialize in weather.")
        agent = create_voice_agent(
            ImmediateGraph(),
            conversation=conversation,
        )
        session = agent._create_session(lambda _event: asyncio.sleep(0))

        assert "create_task(instruction)" in agent.instructions
        assert "update_task(task_id, instruction)" in agent.instructions
        assert "cancel_task(task_id)" in agent.instructions
        assert "never read IDs aloud" in agent.instructions
        assert "Sound warm and specialize in weather." in agent.instructions
        assert "[LANGCHAIN_VOICE_TASK_EVENT]" in agent.instructions
        assert "one coherent user objective" in agent.instructions
        assert "follow-ups about the same objective" in agent.instructions
        assert "responsible for answering the customer" in agent.instructions
        assert "all useful pending results" in agent.instructions
        await session.aclose()

    async def test_configures_gemini_without_exposing_coordination(self) -> None:
        conversation = GeminiLiveConversationLayer(
            instructions="Speak warmly.",
            model="gemini-live-test",
            voice="Aoede",
        )
        agent = create_voice_agent(
            ImmediateGraph(),
            conversation=conversation,
        )

        assert agent.conversation is conversation
        assert conversation.model == "gemini-live-test"
        assert conversation.voice == "Aoede"

    async def test_record_transcript_does_not_trigger_conversation(self) -> None:
        events: list[VoiceEvent] = []

        async def send(event: VoiceEvent) -> None:
            events.append(event)

        agent = create_voice_agent(ImmediateGraph(), conversation=FakeLiveConversation())
        session = agent._create_session(send)

        recorded = await session.record_transcript("assistant", "It is sunny.")

        assert recorded == "It is sunny."
        assert events[-1].as_dict() == {
            "type": "conversation.transcript",
            "role": "assistant",
            "text": "It is sunny.",
        }
        await session.aclose()

    async def test_default_low_level_session_does_not_create_background_work(
        self,
    ) -> None:
        events: list[VoiceEvent] = []

        async def send(event: VoiceEvent) -> None:
            events.append(event)

        session = create_voice_agent(
            ImmediateGraph(), conversation=FakeLiveConversation()
        )._create_session(send)
        await session.receive_text("hello")

        assert [event.type for event in events] == ["session.ready", "conversation.transcript"]
        await session.aclose()

    async def test_disconnect_cancels_active_graph_run(self) -> None:
        async def send(event: VoiceEvent) -> None:
            pass

        graph = BlockingGraph()
        agent = create_voice_agent(graph, conversation=FakeLiveConversation())
        session = agent._create_session(send)
        await session.create_task("long work")
        await graph.started.wait()

        await session.aclose()

        assert graph.cancelled.is_set()


if __name__ == "__main__":
    unittest.main()

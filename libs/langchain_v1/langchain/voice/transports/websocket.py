"""Optional JSON WebSocket transport for LangChain Voice sessions."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any, Protocol, cast

from langchain.voice.events import ConversationReply, ErrorVoiceEvent, VoiceEvent
from langchain.voice.tasks import TaskError, TaskTools

MAX_MESSAGE_BYTES = 64 * 1024
logger = logging.getLogger(__name__)


class TextConversationAdapter(Protocol):
    """Optional text-only coordinator for the JSON WebSocket transport."""

    async def on_user_text(self, text: str, tools: TaskTools) -> ConversationReply | None:
        """Handle one final user text message."""
        ...

    async def on_task_event(self, event: VoiceEvent, tools: TaskTools) -> ConversationReply | None:
        """Handle one task lifecycle event."""
        ...


ConversationFactory = Callable[[str], TextConversationAdapter]


class ProtocolError(ValueError):
    """A safe WebSocket protocol error with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        """Initialize the safe error code and client-facing message."""
        super().__init__(message)
        self.code = code


class WebSocketServer:
    """Expose a VoiceAgent over one isolated session per WebSocket."""

    def __init__(
        self,
        agent: Any,
        *,
        origins: list[str] | None = None,
        conversation_factory: ConversationFactory | None = None,
    ) -> None:
        """Initialize a WebSocket server for a reusable voice agent."""
        self._agent = agent
        self._origins = origins
        self._conversation_factory = conversation_factory

    async def serve_forever(self, host: str, port: int) -> None:
        """Listen for JSON WebSocket sessions until cancelled."""
        try:
            from websockets.asyncio.server import serve  # noqa: PLC0415 - optional extra
        except ImportError as exc:  # pragma: no cover - environment dependent
            msg = "WebSocket support requires `pip install langchain[voice-websocket]`"
            raise RuntimeError(msg) from exc

        async with serve(
            self._handle_connection,
            host,
            port,
            origins=cast("Any", self._origins),
            max_size=MAX_MESSAGE_BYTES,
            max_queue=16,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as server:
            await server.serve_forever()

    async def _handle_connection(self, websocket: Any) -> None:
        text_conversation = (
            self._conversation_factory(self._agent.instructions)
            if self._conversation_factory is not None
            else None
        )
        session = self._agent._create_session()  # noqa: SLF001 - internal transport boundary
        send_lock = asyncio.Lock()

        async def send_event(event: VoiceEvent) -> None:
            payload = json.dumps(event.as_dict(), ensure_ascii=False, separators=(",", ":"))
            async with send_lock:
                await websocket.send(payload)

        async def send_events() -> None:
            async for event in session.events():
                await send_event(event)
                if text_conversation is not None:
                    reply = await text_conversation.on_task_event(event, session.tools)
                    if reply is not None:
                        await send_event(reply.as_event())

        await session.start()
        await send_event(await session.next_event())
        outgoing = asyncio.create_task(send_events(), name="langchain-voice-websocket-send")
        try:
            async for raw_message in websocket:
                try:
                    if isinstance(raw_message, bytes):
                        msg = "binary_not_supported"
                        raise ProtocolError(  # noqa: TRY301 - handled by the protocol loop
                            msg,
                            "Binary audio frames are reserved for an audio transport.",
                        )
                    message = self._decode(raw_message)
                    if message["type"] == "session.close":
                        await session.aclose()
                        await websocket.close(code=1000, reason="session closed")
                        break
                    if message["type"] == "input.text" and text_conversation:
                        text = await session.receive_text(message["text"])
                        reply = await text_conversation.on_user_text(text, session.tools)
                        if reply is not None:
                            await send_event(reply.as_event())
                    else:
                        await session.send(message)
                except ProtocolError as exc:
                    await self._send_error(websocket, exc.code, str(exc))
                except TaskError as exc:
                    await self._send_error(websocket, exc.code, str(exc))
                except (ValueError, TypeError) as exc:
                    await self._send_error(websocket, "invalid_message", str(exc))
                except Exception:
                    logger.exception("Unexpected LangChain Voice session error")
                    await self._send_error(
                        websocket,
                        "internal_error",
                        "The session could not process that message.",
                    )
        finally:
            await session.aclose()
            await asyncio.gather(outgoing, return_exceptions=True)

    async def _dispatch(self, session: Any, raw_message: str) -> bool:
        """Compatibility helper used by protocol unit tests."""
        message = self._decode(raw_message)
        if message["type"] == "session.close":
            return True
        await session.send(message)
        return False

    @staticmethod
    def _decode(raw_message: str) -> dict[str, Any]:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            msg = "invalid_json"
            raise ProtocolError(msg, "Message must be valid JSON.") from exc
        if not isinstance(message, dict):
            msg = "invalid_message"
            raise ProtocolError(msg, "Message must be a JSON object.")
        message_type = message.get("type")
        if not isinstance(message_type, str):
            msg = "invalid_message"
            raise ProtocolError(msg, "Message type must be a string.")
        allowed = {
            "input.text",
            "task.create",
            "task.update",
            "task.cancel",
            "session.close",
        }
        if message_type not in allowed:
            msg = "unknown_message_type"
            raise ProtocolError(msg, f"Unknown message type: {message_type}")
        return message

    @staticmethod
    async def _send_error(websocket: Any, code: str, message: str) -> None:
        event = ErrorVoiceEvent(code=code, message=message)
        await websocket.send(json.dumps(event.as_dict(), ensure_ascii=False, separators=(",", ":")))

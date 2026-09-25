"""Public live-provider conversation contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from langchain.voice.agent import VoiceSession
    from langchain.voice.audio import StatusUI
    from langchain.voice.transport import VoiceTransport


class ConversationLayer(Protocol):
    """Provider adapter for one live, multimodal conversation.

    Applications configure the voice persona in `instructions`. LangChain Voice
    supplies each run with an isolated session containing the assembled prompt
    and scoped task tools, plus provider-neutral audio and observability hooks.
    Implementations own the provider connection and translate its event stream.
    """

    @property
    def instructions(self) -> str:
        """Return the application-configured voice persona instructions."""
        ...

    async def run(
        self,
        session: VoiceSession,
        *,
        transport: VoiceTransport,
        ui: StatusUI | None,
        project_name: str | None,
    ) -> None:
        """Run one live conversation until the provider connection closes."""
        ...

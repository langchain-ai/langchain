from __future__ import annotations

import asyncio
import unittest
from typing import Any

from langchain.voice.providers.openai_realtime import (
    BACKGROUND_RESPONSE,
    USER_RESPONSE,
    ResponseScheduler,
    _runtime_context_item,
)


class FakeResponseAPI:
    def __init__(self) -> None:
        self.calls = 0
        self.created = asyncio.Event()

    async def create(self) -> None:
        self.calls += 1
        self.created.set()


class FakeConnection:
    def __init__(self) -> None:
        self.response = FakeResponseAPI()


class IdleTransport:
    async def wait_output_idle(self) -> None:
        return


class OpenAIRealtimeTests(unittest.IsolatedAsyncioTestCase):
    def test_runtime_results_are_system_context(self) -> None:
        item = _runtime_context_item("task result")

        assert item["type"] == "message"
        assert item["role"] == "system"
        assert item["content"][0]["text"] == "task result"

    async def test_user_turn_consumes_queued_background_result(self) -> None:
        connection = FakeConnection()
        transport: Any = IdleTransport()
        scheduler = ResponseScheduler(connection, transport)
        pending = ["result"]
        injected: list[str] = []
        stale_background_skipped = asyncio.Event()

        async def inject_pending() -> bool:
            if not pending:
                stale_background_skipped.set()
                return False
            injected.extend(pending)
            pending.clear()
            return True

        async def prepare_user_response() -> bool:
            await inject_pending()
            return True

        scheduler.start()
        try:
            scheduler.user_speech_started()
            await scheduler.request(BACKGROUND_RESPONSE, inject_pending)
            await asyncio.sleep(0)
            assert connection.response.calls == 0

            await scheduler.request(USER_RESPONSE, prepare_user_response)
            scheduler.user_speech_finished()
            await asyncio.wait_for(connection.response.created.wait(), 1)

            assert injected == ["result"]
            assert connection.response.calls == 1
            scheduler.mark_done()
            await asyncio.wait_for(stale_background_skipped.wait(), 1)
            assert connection.response.calls == 1
        finally:
            await scheduler.aclose()


if __name__ == "__main__":
    unittest.main()

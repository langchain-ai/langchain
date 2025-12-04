import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from langchain_mistralai.chat_models import ChatMistralAI


def test_streaming_retry_on_stream_failure() -> None:
    """
    Test that retries are attempted when a failure occurs *during* streaming.
    We use httpx.MockTransport to simulate a network error mid-stream.
    """

    async def _run_test() -> None:
        request_count = 0

        async def request_handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1

            # Simulate a stream that yields one valid chunk then crashes
            async def faulty_stream() -> AsyncIterator[bytes]:
                # Valid SSE chunk (Mistral format)
                data = (
                    b'data: {"choices": [{"index": 0, "delta": '
                    b'{"role": "assistant", "content": "Hi"}}]}\n\n'
                )
                yield data
                # Simulate network cut
                msg = "Network Error"
                raise httpx.StreamError(msg)

            return httpx.Response(
                200,
                content=faulty_stream(),
                headers={"content-type": "text/event-stream"},
            )

        # Inject our mock client directly with a base_url to support
        # relative paths used by ChatMistralAI.
        mock_client = httpx.AsyncClient(
            transport=httpx.MockTransport(request_handler),
            base_url="https://api.mistral.ai/v1",
        )

        chat = ChatMistralAI(  
            model="test",# type: ignore[call-arg]
            max_retries=3,
            async_client=mock_client,
            api_key="secret",  # Required to avoid validation error
        )

        # Verify that the exception bubbles up (after retries failed)
        with pytest.raises(httpx.StreamError):
            async for _ in chat.astream("Hello"):
                pass

        # request_count should be > 1 because retries happened
        assert request_count > 1

    asyncio.run(_run_test())

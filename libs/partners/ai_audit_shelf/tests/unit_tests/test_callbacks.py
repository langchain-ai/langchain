from typing import Any
from uuid import uuid4

import pytest
from langchain_core.outputs import Generation, LLMResult

from langchain_ai_audit_shelf.callbacks import AIAuditCallbackHandler


def test_on_llm_end(mocker: Any) -> None:
    """Test on_llm_end logs a chapter."""
    handler = AIAuditCallbackHandler(api_url="http://fake", api_key="test-key")

    mock_post = mocker.patch("requests.post")
    mock_post.return_value.json.return_value = {
        "status": "created",
        "chapter": {"id": "c_001"},
    }

    # Mock LLMResult
    result = LLMResult(generations=[[Generation(text="Test generation")]])
    run_id = uuid4()

    handler.on_llm_end(result, run_id=run_id, prompts=["Test prompt"])

    mock_post.assert_called_once()
    assert "c_001" in handler._chapters


@pytest.mark.asyncio
async def test_on_llm_end_async(mocker: Any) -> None:
    """Test on_llm_end_async logs a chapter."""
    handler = AIAuditCallbackHandler(api_url="http://fake", api_key="test-key")

    # Mock httpx.AsyncClient.post
    mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=mocker.AsyncMock)
    mock_response = mocker.MagicMock()
    mock_response.json.return_value = {"status": "created", "chapter": {"id": "c_002"}}
    mock_post.return_value = mock_response

    result = LLMResult(generations=[[Generation(text="Test async generation")]])
    run_id = uuid4()

    await handler.on_llm_end_async(result, run_id=run_id, prompts=["Test async prompt"])

    mock_post.assert_called_once()
    assert "c_002" in handler._chapters

"""Unit tests for langchain_ai_identity.callback module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from langchain_ai_identity.callback import AIIdentityCallbackHandler


class TestCallbackPostsOnLLMStart:
    def test_posts_audit_on_llm_start(self) -> None:
        handler = AIIdentityCallbackHandler(
            api_key="aid_sk_test",
            agent_id="test-uuid",
            fail_closed=False,
        )
        run_id = uuid4()

        with patch("langchain_ai_identity.callback.post_audit") as mock_post:
            handler.on_llm_start(
                serialized={"name": "gpt-4"},
                prompts=["hello"],
                run_id=run_id,
            )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        # event_type should be llm_start
        assert call_kwargs[1]["event_type"] == "llm_start" or (
            len(call_kwargs[0]) > 2 and call_kwargs[0][2] == "llm_start"
        )


class TestCallbackLatencyTracking:
    def test_latency_ms_calculated(self) -> None:
        handler = AIIdentityCallbackHandler(
            api_key="aid_sk_test",
            agent_id="test-uuid",
            fail_closed=False,
        )
        run_id = uuid4()

        with patch("langchain_ai_identity.callback.post_audit") as mock_post:
            handler.on_llm_start(
                serialized={"name": "gpt-4"},
                prompts=["hello"],
                run_id=run_id,
            )

            mock_llm_result = MagicMock()
            handler.on_llm_end(response=mock_llm_result, run_id=run_id)

        # on_llm_end is the second call
        assert mock_post.call_count == 2
        end_call_kwargs = mock_post.call_args_list[1][1]
        assert "latency_ms" in end_call_kwargs
        assert end_call_kwargs["latency_ms"] is not None
        assert end_call_kwargs["latency_ms"] >= 0

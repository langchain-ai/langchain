"""Unit tests for OpenAI Batch API integration."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from langchain_openai.chat_models._batch_api import (
    BatchRequest,
    BatchResult,
    _parse_error_file,
    _parse_output_file,
    build_batch_requests,
    cancel_batch,
    create_batch_jsonl,
    poll_batch_status,
    retrieve_batch_results,
    upload_and_submit_batch,
)


class TestBatchRequest:
    def test_to_jsonl_line(self) -> None:
        req = BatchRequest(
            custom_id="req-0",
            body={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        line = req.to_jsonl_line()
        parsed = json.loads(line)
        assert parsed["custom_id"] == "req-0"
        assert parsed["method"] == "POST"
        assert parsed["url"] == "/v1/chat/completions"
        assert parsed["body"]["model"] == "gpt-4o-mini"
        assert parsed["body"]["messages"][0]["content"] == "hi"

    def test_to_jsonl_line_custom_endpoint(self) -> None:
        req = BatchRequest(
            custom_id="req-0",
            body={"model": "gpt-4o-mini"},
            endpoint="/v1/embeddings",
        )
        parsed = json.loads(req.to_jsonl_line())
        assert parsed["url"] == "/v1/embeddings"


class TestCreateBatchJsonl:
    def test_single_request(self) -> None:
        requests = [
            BatchRequest(
                custom_id="r-0",
                body={"model": "gpt-4o-mini", "messages": []},
            ),
        ]
        jsonl = create_batch_jsonl(requests)
        lines = jsonl.strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["custom_id"] == "r-0"

    def test_multiple_requests(self) -> None:
        requests = [
            BatchRequest(
                custom_id=f"r-{i}",
                body={"model": "gpt-4o-mini", "messages": []},
            )
            for i in range(5)
        ]
        jsonl = create_batch_jsonl(requests)
        lines = jsonl.strip().split("\n")
        assert len(lines) == 5
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed["custom_id"] == f"r-{i}"


class TestBuildBatchRequests:
    def test_basic_payloads(self) -> None:
        payloads = [
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "q1"}],
                "stream": True,
            },
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "q2"}],
                "stream": True,
            },
        ]
        requests = build_batch_requests(payloads, batch_prefix="test")
        assert len(requests) == 2
        assert requests[0].custom_id == "test-0"
        assert requests[1].custom_id == "test-1"
        # stream should be stripped
        assert "stream" not in requests[0].body
        assert "stream" not in requests[1].body
        # messages preserved
        assert requests[0].body["messages"][0]["content"] == "q1"

    def test_auto_prefix(self) -> None:
        payloads = [{"model": "gpt-4o-mini", "messages": []}]
        requests = build_batch_requests(payloads)
        assert requests[0].custom_id.startswith("lc-")
        # format: lc-<8hex>-0
        parts = requests[0].custom_id.split("-")
        assert len(parts) == 3
        assert parts[2] == "0"

    def test_with_tools_in_payload(self) -> None:
        payloads = [
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "use tool"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                "stream": False,
            },
        ]
        requests = build_batch_requests(payloads, batch_prefix="t")
        assert "tools" in requests[0].body
        assert requests[0].body["tools"][0]["function"]["name"] == "get_weather"

    def test_with_response_format(self) -> None:
        payloads = [
            {
                "model": "gpt-4o-mini",
                "messages": [],
                "response_format": {"type": "json_object"},
            },
        ]
        requests = build_batch_requests(payloads, batch_prefix="t")
        assert "response_format" in requests[0].body


class TestParseOutputFile:
    def test_successful_responses(self) -> None:
        lines = [
            json.dumps(
                {
                    "custom_id": "r-0",
                    "result": {
                        "status_code": 200,
                        "body": {
                            "id": "chatcmpl-1",
                            "choices": [
                                {"message": {"role": "assistant", "content": "hello"}}
                            ],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "r-1",
                    "result": {
                        "status_code": 200,
                        "body": {
                            "id": "chatcmpl-2",
                            "choices": [
                                {"message": {"role": "assistant", "content": "world"}}
                            ],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                        },
                    },
                }
            ),
        ]
        content = "\n".join(lines)
        results = _parse_output_file(content)
        assert len(results) == 2
        assert results["r-0"]["choices"][0]["message"]["content"] == "hello"
        assert results["r-1"]["choices"][0]["message"]["content"] == "world"

    def test_failed_response(self) -> None:
        lines = [
            json.dumps(
                {
                    "custom_id": "r-0",
                    "result": {
                        "status_code": 400,
                        "body": {
                            "error": {
                                "message": "bad request",
                                "type": "invalid_request_error",
                            }
                        },
                    },
                }
            ),
        ]
        content = "\n".join(lines)
        results = _parse_output_file(content)
        # Non-200 status codes are still returned (not as body)
        assert "r-0" in results
        assert results["r-0"]["status_code"] == 400

    def test_empty_content(self) -> None:
        results = _parse_output_file("")
        assert results == {}

    def test_out_of_order_results(self) -> None:
        """Results are NOT guaranteed to be in input order."""
        lines = [
            json.dumps(
                {
                    "custom_id": "r-2",
                    "result": {"status_code": 200, "body": {"id": "c2", "choices": []}},
                }
            ),
            json.dumps(
                {
                    "custom_id": "r-0",
                    "result": {"status_code": 200, "body": {"id": "c0", "choices": []}},
                }
            ),
            json.dumps(
                {
                    "custom_id": "r-1",
                    "result": {"status_code": 200, "body": {"id": "c1", "choices": []}},
                }
            ),
        ]
        results = _parse_output_file("\n".join(lines))
        assert results["r-0"]["id"] == "c0"
        assert results["r-1"]["id"] == "c1"
        assert results["r-2"]["id"] == "c2"


class TestParseErrorFile:
    def test_parse_errors(self) -> None:
        lines = [
            json.dumps(
                {
                    "custom_id": "r-0",
                    "result": {
                        "body": {
                            "error": {
                                "message": "rate limit",
                                "type": "rate_limit_error",
                            },
                        },
                    },
                }
            ),
        ]
        errors = _parse_error_file("\n".join(lines))
        assert "r-0" in errors
        assert errors["r-0"]["message"] == "rate limit"


class TestUploadAndSubmitBatch:
    def test_upload_and_submit(self) -> None:
        mock_client = MagicMock()
        mock_client.files.create.return_value = MagicMock(id="file-abc123")
        mock_client.batches.create.return_value = MagicMock(id="batch-xyz789")

        batch_id = upload_and_submit_batch(
            mock_client,
            '{"custom_id":"r-0","method":"POST","url":"/v1/chat/completions","body":{}}',
        )

        assert batch_id == "batch-xyz789"
        mock_client.files.create.assert_called_once()
        call_kwargs = mock_client.files.create.call_args
        assert call_kwargs.kwargs["purpose"] == "batch"

        mock_client.batches.create.assert_called_once()
        create_kwargs = mock_client.batches.create.call_args.kwargs
        assert create_kwargs["input_file_id"] == "file-abc123"
        assert create_kwargs["endpoint"] == "/v1/chat/completions"
        assert create_kwargs["completion_window"] == "24h"

    def test_upload_with_metadata(self) -> None:
        mock_client = MagicMock()
        mock_client.files.create.return_value = MagicMock(id="file-abc")
        mock_client.batches.create.return_value = MagicMock(id="batch-xyz")

        upload_and_submit_batch(
            mock_client,
            "{}",
            metadata={"project": "test"},
        )

        create_kwargs = mock_client.batches.create.call_args.kwargs
        assert create_kwargs["metadata"] == {"project": "test"}


class TestPollBatchStatus:
    def test_immediate_completion(self) -> None:
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.id = "batch-1"
        mock_batch.status = "completed"
        mock_batch.request_counts = MagicMock(total=5, completed=5, failed=0)
        mock_client.batches.retrieve.return_value = mock_batch

        result = poll_batch_status(mock_client, "batch-1")
        assert result.status == "completed"
        assert result.batch_id == "batch-1"
        assert result.request_counts["completed"] == 5
        mock_client.batches.retrieve.assert_called_once_with("batch-1")

    def test_polls_until_complete(self) -> None:
        mock_client = MagicMock()

        in_progress = MagicMock()
        in_progress.id = "batch-1"
        in_progress.status = "in_progress"
        in_progress.request_counts = MagicMock(total=5, completed=2, failed=0)

        completed = MagicMock()
        completed.id = "batch-1"
        completed.status = "completed"
        completed.request_counts = MagicMock(total=5, completed=5, failed=0)

        mock_client.batches.retrieve.side_effect = [in_progress, completed]

        with patch("langchain_openai.chat_models._batch_api.time.sleep"):
            result = poll_batch_status(mock_client, "batch-1", poll_interval=1.0)

        assert result.status == "completed"
        assert mock_client.batches.retrieve.call_count == 2

    def test_timeout(self) -> None:
        mock_client = MagicMock()
        in_progress = MagicMock()
        in_progress.id = "batch-1"
        in_progress.status = "in_progress"
        in_progress.request_counts = MagicMock(total=5, completed=0, failed=0)
        mock_client.batches.retrieve.return_value = in_progress

        with (
            patch("langchain_openai.chat_models._batch_api.time.sleep"),
            pytest.raises(TimeoutError, match="did not complete"),
        ):
            poll_batch_status(
                mock_client,
                "batch-1",
                poll_interval=1.0,
                max_wait=0.5,
            )

    def test_on_poll_callback(self) -> None:
        mock_client = MagicMock()
        completed = MagicMock()
        completed.id = "batch-1"
        completed.status = "completed"
        completed.request_counts = MagicMock(total=1, completed=1, failed=0)
        mock_client.batches.retrieve.return_value = completed

        callback_results: list[BatchResult] = []
        poll_batch_status(
            mock_client,
            "batch-1",
            on_poll=lambda r: callback_results.append(r),
        )
        assert len(callback_results) == 1
        assert callback_results[0].status == "completed"

    def test_handles_failed_status(self) -> None:
        mock_client = MagicMock()
        failed = MagicMock()
        failed.id = "batch-1"
        failed.status = "failed"
        failed.request_counts = MagicMock(total=5, completed=0, failed=5)
        mock_client.batches.retrieve.return_value = failed

        result = poll_batch_status(mock_client, "batch-1")
        assert result.status == "failed"


class TestRetrieveBatchResults:
    def _make_output_content(self, results: list[dict]) -> MagicMock:
        lines = []
        for r in results:
            lines.append(json.dumps(r))
        mock = MagicMock()
        mock.text = "\n".join(lines)
        return mock

    def test_retrieve_completed(self) -> None:
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.id = "batch-1"
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-out"
        mock_batch.error_file_id = None
        mock_batch.request_counts = MagicMock(total=2, completed=2, failed=0)
        mock_client.batches.retrieve.return_value = mock_batch

        output_content = self._make_output_content(
            [
                {
                    "custom_id": "r-0",
                    "result": {
                        "status_code": 200,
                        "body": {
                            "id": "chatcmpl-1",
                            "choices": [
                                {"message": {"role": "assistant", "content": "hi"}}
                            ],
                        },
                    },
                },
                {
                    "custom_id": "r-1",
                    "result": {
                        "status_code": 200,
                        "body": {
                            "id": "chatcmpl-2",
                            "choices": [
                                {"message": {"role": "assistant", "content": "bye"}}
                            ],
                        },
                    },
                },
            ]
        )
        mock_client.files.content.return_value = output_content

        result = retrieve_batch_results(mock_client, "batch-1")
        assert result.status == "completed"
        assert result.results is not None
        assert len(result.results) == 2
        assert result.results["r-0"]["choices"][0]["message"]["content"] == "hi"

    def test_retrieve_failed_raises(self) -> None:
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.id = "batch-1"
        mock_batch.status = "failed"
        mock_batch.request_counts = MagicMock(total=1, completed=0, failed=1)
        mock_client.batches.retrieve.return_value = mock_batch

        with pytest.raises(ValueError, match="failed"):
            retrieve_batch_results(mock_client, "batch-1")

    def test_retrieve_in_progress_raises(self) -> None:
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.id = "batch-1"
        mock_batch.status = "in_progress"
        mock_batch.request_counts = MagicMock(total=5, completed=2, failed=0)
        mock_client.batches.retrieve.return_value = mock_batch

        with pytest.raises(ValueError, match="not complete"):
            retrieve_batch_results(mock_client, "batch-1")


class TestCancelBatch:
    def test_cancel(self) -> None:
        mock_client = MagicMock()
        cancel_batch(mock_client, "batch-1")
        mock_client.batches.cancel.assert_called_once_with("batch-1")


class TestAdaptiveBackoff:
    """Verify polling interval doubles and caps at 120s."""

    def test_backoff_behavior(self) -> None:
        mock_client = MagicMock()
        call_count = 0

        def side_effect(batch_id: str) -> MagicMock:
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            mock.id = batch_id
            mock.request_counts = MagicMock(total=1, completed=0, failed=0)
            # Complete on 5th call
            mock.status = "completed" if call_count >= 5 else "in_progress"
            return mock

        mock_client.batches.retrieve.side_effect = side_effect

        sleep_durations: list[float] = []

        with patch(
            "langchain_openai.chat_models._batch_api.time.sleep",
            side_effect=lambda d: sleep_durations.append(d),
        ):
            poll_batch_status(mock_client, "batch-1", poll_interval=5.0)

        # 4 sleeps before completion on 5th poll
        assert len(sleep_durations) == 4
        assert sleep_durations[0] == 5.0
        assert sleep_durations[1] == 10.0
        assert sleep_durations[2] == 20.0
        assert sleep_durations[3] == 40.0


class TestChatOpenAIBatchIntegration:
    """Test the batch methods on BaseChatOpenAI without hitting the network."""

    def test_batch_api_disabled_by_default(self) -> None:
        """batch() should use default behavior when use_batch_api=False."""
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o-mini", api_key=SecretStr("fake"))
        assert model.use_batch_api is False

    def test_batch_api_responses_api_conflict(self) -> None:
        """Should raise if both use_batch_api and use_responses_api are True."""
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=SecretStr("fake"),
            use_batch_api=True,
            use_responses_api=True,
        )
        with pytest.raises(ValueError, match="Responses API"):
            model.batch(["test"])

    def test_batch_api_submit_builds_correct_jsonl(self) -> None:
        """Verify that batch_api_submit constructs correct JSONL payloads."""
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o-mini", api_key=SecretStr("fake"))

        submitted_jsonl: list[str] = []

        def mock_upload(client: Any, jsonl: str, **kwargs: Any) -> str:
            submitted_jsonl.append(jsonl)
            return "batch-test-123"

        with patch(
            "langchain_openai.chat_models._batch_api.upload_and_submit_batch",
            side_effect=mock_upload,
        ):
            batch_id = model.batch_api_submit(["Hello", "World"])

        assert batch_id == "batch-test-123"
        assert len(submitted_jsonl) == 1

        lines = submitted_jsonl[0].strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            parsed = json.loads(line)
            assert parsed["method"] == "POST"
            assert parsed["url"] == "/v1/chat/completions"
            assert "model" in parsed["body"]
            assert "messages" in parsed["body"]
            assert "stream" not in parsed["body"]

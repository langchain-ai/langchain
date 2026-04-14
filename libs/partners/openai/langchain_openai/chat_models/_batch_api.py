"""Utilities for OpenAI Batch API integration.

Handles the lifecycle of OpenAI Batch API requests: JSONL creation,
file upload, batch submission, polling, and result retrieval.

See https://platform.openai.com/docs/guides/batch for API details.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatchRequest:
    """A single request within a batch submission."""

    custom_id: str
    body: dict[str, Any]
    endpoint: str = "/v1/chat/completions"
    method: str = "POST"

    def to_jsonl_line(self) -> str:
        """Serialize to a single JSONL line."""
        return json.dumps(
            {
                "custom_id": self.custom_id,
                "method": self.method,
                "url": self.endpoint,
                "body": self.body,
            }
        )


@dataclass
class BatchResult:
    """Result of a batch API operation."""

    batch_id: str
    status: str
    results: dict[str, Any] | None = None
    errors: dict[str, Any] | None = None
    request_counts: dict[str, int] = field(default_factory=dict)


def create_batch_jsonl(requests: list[BatchRequest]) -> str:
    """Convert BatchRequest objects to JSONL string for upload.

    Args:
        requests: List of batch requests to serialize.

    Returns:
        JSONL-formatted string with one request per line.
    """
    return "\n".join(req.to_jsonl_line() for req in requests)


def build_batch_requests(
    payloads: list[dict[str, Any]],
    *,
    batch_prefix: str | None = None,
) -> list[BatchRequest]:
    """Build BatchRequest objects from LangChain request payloads.

    Args:
        payloads: List of request payloads from ``_get_request_payload()``.
        batch_prefix: Optional prefix for custom_ids. Defaults to a random UUID
            segment to group requests from the same batch call.

    Returns:
        List of BatchRequest objects ready for JSONL serialization.
    """
    prefix = batch_prefix or f"lc-{uuid.uuid4().hex[:8]}"
    requests = []
    for i, payload in enumerate(payloads):
        body = {k: v for k, v in payload.items() if k != "stream"}
        requests.append(
            BatchRequest(
                custom_id=f"{prefix}-{i}",
                body=body,
            )
        )
    return requests


def upload_and_submit_batch(
    client: Any,
    jsonl_content: str,
    *,
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
    metadata: dict[str, str] | None = None,
) -> str:
    """Upload JSONL file and create a batch job.

    Args:
        client: An ``openai.OpenAI`` root client instance.
        jsonl_content: JSONL-formatted string of batch requests.
        endpoint: The API endpoint for batch processing.
        completion_window: Time window for batch completion.
        metadata: Optional metadata key-value pairs for the batch.

    Returns:
        The batch ID string.
    """
    file_obj = io.BytesIO(jsonl_content.encode("utf-8"))
    file_obj.name = "batch_requests.jsonl"
    batch_file = client.files.create(file=file_obj, purpose="batch")

    create_kwargs: dict[str, Any] = {
        "input_file_id": batch_file.id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    if metadata:
        create_kwargs["metadata"] = metadata

    batch = client.batches.create(**create_kwargs)
    return batch.id


async def async_upload_and_submit_batch(
    client: Any,
    jsonl_content: str,
    *,
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
    metadata: dict[str, str] | None = None,
) -> str:
    """Async version: upload JSONL file and create a batch job.

    Args:
        client: An ``openai.AsyncOpenAI`` root client instance.
        jsonl_content: JSONL-formatted string of batch requests.
        endpoint: The API endpoint for batch processing.
        completion_window: Time window for batch completion.
        metadata: Optional metadata key-value pairs for the batch.

    Returns:
        The batch ID string.
    """
    file_obj = io.BytesIO(jsonl_content.encode("utf-8"))
    file_obj.name = "batch_requests.jsonl"
    batch_file = await client.files.create(file=file_obj, purpose="batch")

    create_kwargs: dict[str, Any] = {
        "input_file_id": batch_file.id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    if metadata:
        create_kwargs["metadata"] = metadata

    batch = await client.batches.create(**create_kwargs)
    return batch.id


_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "expired", "cancelled", "canceled"}
)
_MAX_POLL_INTERVAL = 120.0


def _parse_batch_to_result(batch: Any) -> BatchResult:
    """Convert an OpenAI Batch object to a BatchResult."""
    counts = {}
    if hasattr(batch, "request_counts") and batch.request_counts:
        rc = batch.request_counts
        counts = {
            "total": getattr(rc, "total", 0),
            "completed": getattr(rc, "completed", 0),
            "failed": getattr(rc, "failed", 0),
        }
    return BatchResult(
        batch_id=batch.id,
        status=batch.status,
        request_counts=counts,
    )


def poll_batch_status(
    client: Any,
    batch_id: str,
    *,
    poll_interval: float = 10.0,
    max_wait: float | None = None,
    on_poll: Callable[[BatchResult], None] | None = None,
) -> BatchResult:
    """Poll a batch until it reaches a terminal status.

    Uses adaptive backoff: interval doubles each poll, capped at 120s.

    Args:
        client: An ``openai.OpenAI`` root client instance.
        batch_id: The batch ID to monitor.
        poll_interval: Initial polling interval in seconds.
        max_wait: Maximum total wait time in seconds. ``None`` waits indefinitely.
        on_poll: Optional callback invoked with current status on each poll.

    Returns:
        Final BatchResult once the batch reaches a terminal status.

    Raises:
        TimeoutError: If ``max_wait`` is exceeded before batch completes.
    """
    interval = poll_interval
    elapsed = 0.0
    while True:
        batch = client.batches.retrieve(batch_id)
        result = _parse_batch_to_result(batch)
        if on_poll:
            on_poll(result)
        if result.status in _TERMINAL_STATUSES:
            return result
        if max_wait is not None and elapsed >= max_wait:
            msg = (
                f"Batch {batch_id} did not complete within {max_wait}s "
                f"(status: {result.status}). Retrieve results later with "
                f"batch_api_retrieve('{batch_id}')."
            )
            raise TimeoutError(msg)
        time.sleep(interval)
        elapsed += interval
        interval = min(interval * 2, _MAX_POLL_INTERVAL)


async def async_poll_batch_status(
    client: Any,
    batch_id: str,
    *,
    poll_interval: float = 10.0,
    max_wait: float | None = None,
    on_poll: Callable[[BatchResult], None] | None = None,
) -> BatchResult:
    """Async poll a batch until it reaches a terminal status.

    Args:
        client: An ``openai.AsyncOpenAI`` root client instance.
        batch_id: The batch ID to monitor.
        poll_interval: Initial polling interval in seconds.
        max_wait: Maximum total wait time in seconds. ``None`` waits indefinitely.
        on_poll: Optional callback invoked with current status on each poll.

    Returns:
        Final BatchResult once the batch reaches a terminal status.

    Raises:
        TimeoutError: If ``max_wait`` is exceeded before batch completes.
    """
    interval = poll_interval
    elapsed = 0.0
    while True:
        batch = await client.batches.retrieve(batch_id)
        result = _parse_batch_to_result(batch)
        if on_poll:
            on_poll(result)
        if result.status in _TERMINAL_STATUSES:
            return result
        if max_wait is not None and elapsed >= max_wait:
            msg = (
                f"Batch {batch_id} did not complete within {max_wait}s "
                f"(status: {result.status}). Retrieve results later with "
                f"abatch_api_retrieve('{batch_id}')."
            )
            raise TimeoutError(msg)
        await asyncio.sleep(interval)
        elapsed += interval
        interval = min(interval * 2, _MAX_POLL_INTERVAL)


def _parse_output_file(content: str) -> dict[str, Any]:
    """Parse a batch output JSONL file into a dict keyed by custom_id.

    Args:
        content: Raw text content of the output file.

    Returns:
        Dict mapping custom_id to the response body.
    """
    results: dict[str, Any] = {}
    for line in content.strip().split("\n"):
        if not line:
            continue
        entry = json.loads(line)
        custom_id = entry["custom_id"]
        result_body = entry.get("result", {})
        if result_body.get("status_code") == 200:
            results[custom_id] = result_body.get("body")
        else:
            results[custom_id] = result_body
    return results


def _parse_error_file(content: str) -> dict[str, Any]:
    """Parse a batch error JSONL file into a dict keyed by custom_id.

    Args:
        content: Raw text content of the error file.

    Returns:
        Dict mapping custom_id to the error body.
    """
    errors: dict[str, Any] = {}
    for line in content.strip().split("\n"):
        if not line:
            continue
        entry = json.loads(line)
        custom_id = entry["custom_id"]
        errors[custom_id] = entry.get("result", {}).get("body", {}).get("error", {})
    return errors


def retrieve_batch_results(
    client: Any,
    batch_id: str,
) -> BatchResult:
    """Retrieve and parse results for a completed batch.

    Args:
        client: An ``openai.OpenAI`` root client instance.
        batch_id: The batch ID to retrieve results for.

    Returns:
        BatchResult with parsed results and errors.

    Raises:
        ValueError: If the batch has not completed successfully.
    """
    batch = client.batches.retrieve(batch_id)
    result = _parse_batch_to_result(batch)

    if batch.status == "failed":
        msg = f"Batch {batch_id} failed."
        raise ValueError(msg)

    if batch.status not in _TERMINAL_STATUSES:
        msg = (
            f"Batch {batch_id} is not complete (status: {batch.status}). "
            f"Poll with batch_api_status() first."
        )
        raise ValueError(msg)

    if batch.output_file_id:
        output_content = client.files.content(batch.output_file_id)
        result.results = _parse_output_file(output_content.text)

    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        result.errors = _parse_error_file(error_content.text)

    return result


async def async_retrieve_batch_results(
    client: Any,
    batch_id: str,
) -> BatchResult:
    """Async retrieve and parse results for a completed batch.

    Args:
        client: An ``openai.AsyncOpenAI`` root client instance.
        batch_id: The batch ID to retrieve results for.

    Returns:
        BatchResult with parsed results and errors.

    Raises:
        ValueError: If the batch has not completed successfully.
    """
    batch = await client.batches.retrieve(batch_id)
    result = _parse_batch_to_result(batch)

    if batch.status == "failed":
        msg = f"Batch {batch_id} failed."
        raise ValueError(msg)

    if batch.status not in _TERMINAL_STATUSES:
        msg = (
            f"Batch {batch_id} is not complete (status: {batch.status}). "
            f"Poll with batch_api_status() first."
        )
        raise ValueError(msg)

    if batch.output_file_id:
        output_content = await client.files.content(batch.output_file_id)
        result.results = _parse_output_file(output_content.text)

    if batch.error_file_id:
        error_content = await client.files.content(batch.error_file_id)
        result.errors = _parse_error_file(error_content.text)

    return result


def cancel_batch(client: Any, batch_id: str) -> None:
    """Cancel an in-progress batch.

    Args:
        client: An ``openai.OpenAI`` root client instance.
        batch_id: The batch ID to cancel.
    """
    client.batches.cancel(batch_id)


async def async_cancel_batch(client: Any, batch_id: str) -> None:
    """Async cancel an in-progress batch.

    Args:
        client: An ``openai.AsyncOpenAI`` root client instance.
        batch_id: The batch ID to cancel.
    """
    await client.batches.cancel(batch_id)

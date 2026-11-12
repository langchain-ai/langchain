"""Conversion, validation, and transport helpers for LangChain A2A integration."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import posixpath
import queue
import re
from collections.abc import (
    AsyncIterator,
    Callable,
    Coroutine,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import urlparse

import httpx
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)
from langgraph.types import Command
from pydantic import BaseModel

try:
    from a2a.client import A2ACardResolver, Client, ClientConfig, create_client
    from a2a.client.card_resolver import parse_agent_card
    from a2a.helpers import (
        get_data_parts,
        get_message_text,
        new_data_part,
        new_message,
        new_text_part,
    )
    from a2a.types import (
        AgentCard,
        AgentInterface,
        Message,
        Part,
        Role,
        SendMessageRequest,
        StreamResponse,
        TaskState,
    )
    from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol
except ImportError as exc:
    msg = (
        "The 'a2a-sdk' package is required to use `langchain.a2a`. "
        "Install it with `pip install 'langchain[a2a]'` or "
        "`uv add 'langchain[a2a]'`."
    )
    raise ImportError(msg) from exc

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

_T = TypeVar("_T")
logger = logging.getLogger(__name__)

_KNOWN_CARD_SUFFIXES = (
    "/.well-known/agent-card.json",
    "/.well-known/agent.json",
)
_FAILURE_STATES = frozenset(
    {
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
        TaskState.TASK_STATE_AUTH_REQUIRED,
    }
)
_STREAM_DONE = object()
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


class A2AExecutionError(RuntimeError):
    """Error raised when an A2A agent fails, cancels, rejects, or blocks a request."""

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        context_id: str | None = None,
        state: str | None = None,
    ) -> None:
        """Initialize an A2A execution error.

        Args:
            message: Human-readable error message.
            task_id: Optional A2A task identifier associated with the failure.
            context_id: Optional A2A context identifier associated with the failure.
            state: Optional A2A task state name at failure time.
        """
        super().__init__(message)
        self.task_id = task_id
        self.context_id = context_id
        self.state = state


def validate_a2a_url(url: str) -> str:
    """Validate and normalize an HTTP/HTTPS URL for A2A endpoints.

    Args:
        url: Candidate endpoint or agent-card URL.

    Returns:
        Trimmed URL string.

    Raises:
        ValueError: If the URL is empty, contains whitespace, lacks a host, or
            does not use `http` or `https`.
    """
    cleaned = url.strip()
    if not cleaned:
        msg = "A2A URL cannot be empty."
        raise ValueError(msg)
    if any(ch.isspace() for ch in cleaned):
        msg = f"A2A URL must not contain whitespace: {url!r}"
        raise ValueError(msg)

    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        msg = f"A2A URL must be an absolute http(s) URL, got: {url!r}"
        raise ValueError(msg)

    return cleaned


def split_agent_card_url(url: str) -> tuple[str, str | None]:
    """Split a full agent-card JSON URL into `(base_url, relative_card_path)`.

    Supports base URLs (returning `(base_url, None)`), standard well-known card
    URLs (`/.well-known/agent-card.json` or `/.well-known/agent.json`), and
    custom `.json` card paths.

    Args:
        url: Base A2A URL or full URL pointing to an agent card JSON document.

    Returns:
        Tuple of `(base_url, relative_card_path)`.
    """
    cleaned = validate_a2a_url(url)
    parsed = urlparse(cleaned)
    path = parsed.path or ""

    for suffix in _KNOWN_CARD_SUFFIXES:
        if path.endswith(suffix):
            prefix = path[: -len(suffix)]
            base = parsed._replace(path=prefix, params="", query="", fragment="").geturl()
            return base.rstrip("/"), suffix.lstrip("/")

    if path.endswith(".json"):
        parent_dir = posixpath.dirname(path)
        filename = posixpath.basename(path)
        if parent_dir == "/":
            parent_dir = ""
        base = parsed._replace(path=parent_dir, params="", query="", fragment="").geturl()
        relative = filename
        if parsed.query:
            relative = f"{filename}?{parsed.query}"
        return base.rstrip("/"), relative

    return cleaned.rstrip("/"), None


def join_url(base_url: str, path: str) -> str:
    """Join a base URL and a relative path cleanly.

    Args:
        base_url: Base HTTP/HTTPS URL.
        path: Relative endpoint path.

    Returns:
        Joined URL string.
    """
    normalized_base = base_url.rstrip("/")
    normalized_path = path.lstrip("/")
    if not normalized_path:
        return normalized_base
    return f"{normalized_base}/{normalized_path}"


def slugify_identifier(name: str, *, max_length: int = 64, fallback: str = "a2a_agent") -> str:
    """Convert an arbitrary label into a safe `[a-zA-Z0-9_-]` identifier.

    Args:
        name: Source label or agent name.
        max_length: Maximum identifier length.
        fallback: Fallback identifier if sanitization yields an empty string.

    Returns:
        Sanitized identifier string.
    """
    replaced = _SAFE_NAME_RE.sub("_", name.strip())
    collapsed = _MULTI_UNDERSCORE_RE.sub("_", replaced).strip("_-")
    if not collapsed:
        return fallback[:max_length]
    return collapsed[:max_length]


def derive_agent_name_from_url(url: str) -> str:
    """Derive a human-readable identifier from an A2A endpoint URL.

    Args:
        url: HTTP/HTTPS endpoint URL.

    Returns:
        Sanitized identifier derived from the URL host and path.
    """
    parsed = urlparse(validate_a2a_url(url))
    base_part = parsed.netloc.split(":")[0]
    path_part = parsed.path.strip("/")
    raw = f"{base_part}_{path_part}" if path_part else base_part
    return slugify_identifier(raw)


def extract_text_from_content(content: Any) -> str:
    """Extract plain text from a LangChain message `content` field.

    Args:
        content: String content or multimodal sequence of blocks.

    Returns:
        Joined plain-text representation.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, Mapping):
                text_val = item.get("text")
                if isinstance(text_val, str):
                    chunks.append(text_val)
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content)


def to_json_compatible(value: Any) -> Any:
    """Recursively convert Python/LangChain objects into JSON-serializable values.

    Args:
        value: Arbitrary Python value, Pydantic model, or LangChain message.

    Returns:
        JSON-serializable Python structure compatible with Protobuf `Value`.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, BaseMessage):
        return message_to_dict(value)
    if isinstance(value, BaseModel):
        return to_json_compatible(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(k): to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, bytes | bytearray):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, Sequence | set | frozenset):
        return [to_json_compatible(item) for item in value]
    return str(value)


def _coerce_message(item: Any) -> BaseMessage:
    """Convert a serialized dict or message-like object into a LangChain `BaseMessage`."""
    if isinstance(item, BaseMessage):
        return item
    if isinstance(item, Mapping):
        if "type" in item and "data" in item:
            loaded = messages_from_dict([dict(item)])
            if loaded:
                return loaded[0]
        role_raw = str(item.get("role", "user")).lower()
        content_raw = item.get("content", "")
        msg_id = item.get("id")
        msg_id_str = str(msg_id) if msg_id is not None else None
        if role_raw in {"assistant", "ai", "model"}:
            return AIMessage(content=content_raw, id=msg_id_str)
        if role_raw == "system":
            return SystemMessage(content=content_raw, id=msg_id_str)
        if role_raw == "tool":
            tool_call_id = str(item.get("tool_call_id", ""))
            return ToolMessage(content=content_raw, tool_call_id=tool_call_id, id=msg_id_str)
        return HumanMessage(content=content_raw, id=msg_id_str)
    return HumanMessage(content=str(item))


def deserialize_messages(raw_messages: Sequence[Any]) -> list[BaseMessage]:
    """Deserialize serialized message dictionaries into LangChain message instances.

    Args:
        raw_messages: Sequence of message objects or serialized dictionaries.

    Returns:
        List of LangChain `BaseMessage` instances.
    """
    return [_coerce_message(item) for item in raw_messages]


def _inspect_annotation_fields(annotation: Any) -> dict[str, Any]:
    """Extract property metadata from a Pydantic model or TypedDict annotation."""
    model_fields = getattr(annotation, "model_fields", None)
    if isinstance(model_fields, Mapping) and model_fields:
        return {
            str(k): {
                "type": "string" if getattr(field_info, "annotation", None) is str else "object"
            }
            for k, field_info in model_fields.items()
        }
    annotations = getattr(annotation, "__annotations__", None)
    if isinstance(annotations, Mapping) and annotations:
        return {
            str(k): {"type": "string" if v is str else "object"} for k, v in annotations.items()
        }
    return {}


def _get_input_schema_properties(graph: Runnable[Any, Any]) -> dict[str, Any]:
    """Inspect a runnable's input JSON schema properties safely."""
    channels = getattr(graph, "channels", None)
    try:
        schema_obj = graph.get_input_schema()
        model_fields = getattr(schema_obj, "model_fields", None)
        if isinstance(model_fields, Mapping) and model_fields:
            if set(model_fields.keys()) == {"root"}:
                root_annotation = getattr(model_fields["root"], "annotation", None)
                unwrapped = _inspect_annotation_fields(root_annotation)
                if unwrapped:
                    return unwrapped
            else:
                return _inspect_annotation_fields(schema_obj)
        json_schema = (
            schema_obj.model_json_schema()
            if hasattr(schema_obj, "model_json_schema")
            else schema_obj.schema()
        )
        properties = json_schema.get("properties")
        if isinstance(properties, dict) and properties:
            return properties
    except Exception:
        logger.debug("Could not inspect graph input schema", exc_info=True)
    if isinstance(channels, Mapping):
        return {
            str(k): {}
            for k in channels
            if not str(k).startswith("__") and not str(k).startswith("branch:")
        }
    return {}


def graph_expects_messages(graph: Runnable[Any, Any]) -> bool:
    """Determine whether a runnable/graph accepts a `messages` state key."""
    return "messages" in _get_input_schema_properties(graph)


def _graph_single_text_key(graph: Runnable[Any, Any]) -> str | None:
    """Infer a single text input key when the graph does not use `messages`."""
    properties = _get_input_schema_properties(graph)
    if not properties:
        return None
    for candidate in ("input", "query", "text", "prompt", "question"):
        if candidate in properties:
            return candidate
    if len(properties) == 1:
        only_key, prop_schema = next(iter(properties.items()))
        if isinstance(prop_schema, Mapping) and prop_schema.get("type") == "string":
            return only_key
    return None


def extract_message_payload(message: Message) -> tuple[str, list[Any]]:
    """Extract combined plain text and structured data parts from an A2A `Message`.

    Args:
        message: Inbound or outbound A2A `Message`.

    Returns:
        Tuple of `(text, data_parts)`.
    """
    parts = list(message.parts)
    text = get_message_text(message)
    data_items = get_data_parts(parts)
    return text, data_items


def _merge_data_dicts(data_items: Sequence[Any]) -> dict[str, Any] | None:
    """Merge dictionary entries from A2A `data` parts into a single mapping."""
    merged: dict[str, Any] = {}
    found_mapping = False
    for item in data_items:
        if isinstance(item, Mapping):
            found_mapping = True
            for k, v in item.items():
                merged[str(k)] = v
    return merged if found_mapping else None


def a2a_message_to_graph_input(
    message: Message,
    graph: Runnable[Any, Any],
    *,
    input_key: str | None = None,
    resume_mode: bool = False,
) -> Any:
    """Convert an inbound A2A `Message` into an input payload for a LangGraph/Runnable.

    Args:
        message: Inbound A2A user message.
        graph: Target LangGraph graph or LangChain runnable.
        input_key: Optional explicit dictionary key for text-only inputs.
        resume_mode: Whether the server is resuming an interrupted LangGraph task.

    Returns:
        Converted graph input payload (or `Command(resume=...)` when resuming).
    """
    text, data_items = extract_message_payload(message)
    merged_dict = _merge_data_dicts(data_items)

    if resume_mode:
        if merged_dict is not None and "__langgraph_resume__" in merged_dict:
            return Command(resume=merged_dict["__langgraph_resume__"])
        if merged_dict is not None and not text:
            return Command(resume=merged_dict)
        if data_items and not text and len(data_items) == 1:
            return Command(resume=data_items[0])
        return Command(resume=text)

    if merged_dict is not None:
        if "__langgraph_resume__" in merged_dict:
            return Command(resume=merged_dict["__langgraph_resume__"])
        payload = dict(merged_dict)
        if "messages" in payload and isinstance(payload["messages"], Sequence):
            payload["messages"] = deserialize_messages(payload["messages"])
        elif graph_expects_messages(graph) and text:
            payload["messages"] = [HumanMessage(content=text)]
        return payload

    if input_key is not None:
        return {input_key: text}

    if graph_expects_messages(graph):
        return {"messages": [HumanMessage(content=text)]}

    if single_key := _graph_single_text_key(graph):
        return {single_key: text}

    properties = _get_input_schema_properties(graph)
    if properties:
        return {"messages": [HumanMessage(content=text)]}

    return text


def _extract_last_ai_text(messages: Sequence[Any]) -> str | None:
    """Return text from the latest assistant/AI message in a message sequence."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return extract_text_from_content(msg.content)
        if isinstance(msg, Mapping):
            role = str(msg.get("role", msg.get("type", ""))).lower()
            if role in {"ai", "assistant", "model"}:
                content = msg.get("content", "")
                if isinstance(content, str | Sequence):
                    return extract_text_from_content(content)
    for msg in reversed(messages):
        if isinstance(msg, BaseMessage):
            return extract_text_from_content(msg.content)
    return None


def _parts_from_mapping_result(result: Mapping[str, Any]) -> list[Part]:
    """Convert a dictionary graph output into A2A `Part` instances."""
    parts: list[Part] = []
    text: str | None = None

    messages = result.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, str | bytes):
        text = _extract_last_ai_text(messages)

    structured = result.get("structured_response")
    if text is None and structured is not None:
        text = json.dumps(to_json_compatible(structured), ensure_ascii=False)

    if text is None:
        for scalar_key in ("output", "response", "answer", "result"):
            candidate = result.get(scalar_key)
            if isinstance(candidate, str):
                text = candidate
                break

    extra_data = {
        str(k): to_json_compatible(v)
        for k, v in result.items()
        if str(k) not in {"messages", "__interrupt__"}
    }

    if text is None and extra_data:
        text = json.dumps(extra_data, ensure_ascii=False)

    if text:
        parts.append(new_text_part(text))
    if extra_data:
        parts.append(new_data_part(extra_data))

    if not parts:
        parts.append(new_text_part(""))
    return parts


def graph_result_to_a2a_parts(result: Any) -> tuple[list[Part], bool]:
    """Convert a LangGraph execution result into A2A `Part` objects.

    Also detects LangGraph `interrupt()` payloads under `"__interrupt__"`.

    Args:
        result: Return value from `graph.ainvoke(...)` or `graph.invoke(...)`.

    Returns:
        Tuple of `(parts, is_interrupt)`.
    """
    unwrapped = getattr(result, "value", result)
    raw_interrupts = getattr(result, "interrupts", None)

    if raw_interrupts is None and isinstance(unwrapped, Mapping):
        raw_interrupts = unwrapped.get("__interrupt__")

    if raw_interrupts:
        values = [to_json_compatible(getattr(item, "value", item)) for item in raw_interrupts]
        if len(values) == 1 and isinstance(values[0], str):
            prompt_text = values[0]
        else:
            prompt_text = json.dumps(
                values[0] if len(values) == 1 else values,
                ensure_ascii=False,
            )
        return (
            [
                new_text_part(prompt_text),
                new_data_part({"__interrupts__": values}),
            ],
            True,
        )

    if isinstance(unwrapped, str):
        return [new_text_part(unwrapped)], False

    if isinstance(unwrapped, BaseMessage):
        return [new_text_part(extract_text_from_content(unwrapped.content))], False

    if isinstance(unwrapped, Mapping):
        return _parts_from_mapping_result(unwrapped), False

    if isinstance(unwrapped, BaseModel):
        dumped = to_json_compatible(unwrapped)
        text = json.dumps(dumped, ensure_ascii=False)
        return [new_text_part(text), new_data_part(dumped)], False

    return [new_text_part(str(unwrapped))], False


def langchain_input_to_a2a_message(
    input_data: Any,
    *,
    context_id: str | None = None,
    task_id: str | None = None,
) -> Message:
    """Convert a LangChain or LangGraph input value into an A2A user `Message`.

    Args:
        input_data: Prompt string, `BaseMessage`, `Command`, or state dict.
        context_id: Optional A2A conversation context ID.
        task_id: Optional A2A task ID for multi-turn continuation or resume.

    Returns:
        A2A `Message` with `ROLE_USER`.
    """
    parts: list[Part] = []

    if isinstance(input_data, Command):
        resume_val = input_data.resume
        if isinstance(resume_val, str):
            parts.append(new_text_part(resume_val))
        else:
            parts.append(
                new_text_part(json.dumps(to_json_compatible(resume_val), ensure_ascii=False))
            )
        parts.append(new_data_part({"__langgraph_resume__": to_json_compatible(resume_val)}))
    elif isinstance(input_data, str):
        parts.append(new_text_part(input_data))
    elif isinstance(input_data, BaseMessage):
        parts.append(new_text_part(extract_text_from_content(input_data.content)))
    elif isinstance(input_data, Mapping):
        if "messages" in input_data and isinstance(input_data["messages"], Sequence):
            raw_msgs = list(input_data["messages"])
            serialized = to_json_compatible(raw_msgs)
            latest_text = ""
            if raw_msgs:
                last_msg = raw_msgs[-1]
                if isinstance(last_msg, BaseMessage):
                    latest_text = extract_text_from_content(last_msg.content)
                elif isinstance(last_msg, Mapping):
                    latest_text = extract_text_from_content(last_msg.get("content", ""))
                else:
                    latest_text = str(last_msg)
            extra = {
                str(k): to_json_compatible(v) for k, v in input_data.items() if str(k) != "messages"
            }
            if latest_text:
                parts.append(new_text_part(latest_text))
            if extra or len(raw_msgs) > 1 or not latest_text:
                data_payload: dict[str, Any] = {"messages": serialized, **extra}
                parts.append(new_data_part(data_payload))
        else:
            dumped = to_json_compatible(input_data)
            parts.append(new_data_part(dumped))
            scalar_text: str | None = None
            for key in ("query", "input", "prompt", "text", "question"):
                if isinstance(input_data.get(key), str):
                    scalar_text = input_data[key]
                    break
            parts.append(
                new_text_part(
                    scalar_text
                    if scalar_text is not None
                    else json.dumps(dumped, ensure_ascii=False)
                )
            )
    else:
        parts.append(new_text_part(str(input_data)))

    if not parts:
        parts.append(new_text_part(""))

    return new_message(
        parts=parts,
        context_id=context_id,
        task_id=task_id,
        role=Role.ROLE_USER,
    )


def _parts_text(parts: Sequence[Part], delimiter: str = "\n") -> str:
    """Extract joined text from a sequence of A2A `Part` objects."""
    return delimiter.join(part.text for part in parts if part.text)


@dataclass
class A2ACollectedResult:
    """Aggregated response extracted from an A2A stream or task result."""

    text: str
    data: list[Any] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    task_id: str | None = None
    context_id: str | None = None
    state: TaskState | None = None
    is_input_required: bool = False


class A2AStreamAccumulator:
    """Incrementally collect A2A `StreamResponse` events into a unified result."""

    def __init__(self) -> None:
        """Initialize an empty accumulator."""
        self.task_id: str | None = None
        self.context_id: str | None = None
        self.state: TaskState | None = None
        self._artifact_parts: dict[str, list[Part]] = {}
        self._artifact_names: dict[str, str] = {}
        self._artifact_order: list[str] = []
        self._status_message_parts: list[Part] = []
        self._direct_message_parts: list[Part] = []

    def _record_artifact(
        self,
        artifact_id: str,
        name: str,
        parts: Sequence[Part],
        *,
        append: bool,
    ) -> str:
        """Store artifact parts and return newly emitted text delta."""
        key = artifact_id or "__default__"
        if key not in self._artifact_parts:
            self._artifact_parts[key] = []
            self._artifact_order.append(key)
        if name:
            self._artifact_names[key] = name

        delta_text = _parts_text(parts, delimiter="")
        if append:
            target = self._artifact_parts[key]
            for part in parts:
                if (
                    target
                    and part.text
                    and not part.raw
                    and not part.url
                    and not part.HasField("data")
                ):
                    prev = target[-1]
                    if prev.text and not prev.raw and not prev.url and not prev.HasField("data"):
                        merged = Part()
                        merged.CopyFrom(prev)
                        merged.text = prev.text + part.text
                        target[-1] = merged
                        continue
                target.append(part)
            return delta_text

        previous_text = _parts_text(self._artifact_parts[key])
        self._artifact_parts[key] = list(parts)
        if delta_text and delta_text != previous_text:
            return delta_text
        return ""

    def consume(self, event: StreamResponse) -> str | None:
        """Consume a single `StreamResponse` and return any new text delta.

        Args:
            event: A2A `StreamResponse` yielded by `Client.send_message`.

        Returns:
            Newly added text delta if present, otherwise `None`.
        """
        if event.HasField("message"):
            msg = event.message
            if msg.task_id:
                self.task_id = msg.task_id
            if msg.context_id:
                self.context_id = msg.context_id
            parts = list(msg.parts)
            self._direct_message_parts.extend(parts)
            delta = _parts_text(parts)
            return delta or None

        if event.HasField("artifact_update"):
            upd = event.artifact_update
            if upd.task_id:
                self.task_id = upd.task_id
            if upd.context_id:
                self.context_id = upd.context_id
            artifact = upd.artifact
            delta = self._record_artifact(
                artifact.artifact_id,
                artifact.name,
                list(artifact.parts),
                append=upd.append,
            )
            return delta or None

        if event.HasField("status_update"):
            upd_status = event.status_update
            if upd_status.task_id:
                self.task_id = upd_status.task_id
            if upd_status.context_id:
                self.context_id = upd_status.context_id
            status = upd_status.status
            self.state = status.state
            if status.HasField("message"):
                parts = list(status.message.parts)
                self._status_message_parts = parts
                if status.state == TaskState.TASK_STATE_WORKING:
                    delta = _parts_text(parts)
                    return delta or None
            return None

        if event.HasField("task"):
            task = event.task
            if task.id:
                self.task_id = task.id
            if task.context_id:
                self.context_id = task.context_id
            if task.HasField("status"):
                self.state = task.status.state
                if task.status.HasField("message"):
                    self._status_message_parts = list(task.status.message.parts)
            delta_chunks: list[str] = []
            for artifact in task.artifacts:
                chunk = self._record_artifact(
                    artifact.artifact_id,
                    artifact.name,
                    list(artifact.parts),
                    append=False,
                )
                if chunk:
                    delta_chunks.append(chunk)
            if not task.artifacts and task.history:
                for hist_msg in reversed(task.history):
                    if hist_msg.role == Role.ROLE_AGENT:
                        self._direct_message_parts = list(hist_msg.parts)
                        break
            return "\n".join(delta_chunks) or None

        return None

    def finalize(
        self,
        *,
        raise_on_failure: bool = True,
        raise_on_input_required: bool = False,
    ) -> A2ACollectedResult:
        """Finalize accumulated events and validate terminal A2A task status.

        Args:
            raise_on_failure: Raise `A2AExecutionError` on failed/canceled/rejected
                terminal states.
            raise_on_input_required: Raise `A2AExecutionError` when the remote task
                pauses in `TASK_STATE_INPUT_REQUIRED`.

        Returns:
            Aggregated `A2ACollectedResult`.

        Raises:
            A2AExecutionError: If the task ended in a failure state (or
                `INPUT_REQUIRED` when `raise_on_input_required=True`).
        """
        status_text = _parts_text(self._status_message_parts)
        is_input_required = self.state == TaskState.TASK_STATE_INPUT_REQUIRED

        if raise_on_failure and self.state in _FAILURE_STATES:
            state_name = TaskState.Name(self.state)
            detail = status_text or f"Remote A2A task ended in state {state_name}."
            raise A2AExecutionError(
                detail,
                task_id=self.task_id,
                context_id=self.context_id,
                state=state_name,
            )

        if raise_on_input_required and is_input_required:
            state_name = TaskState.Name(TaskState.TASK_STATE_INPUT_REQUIRED)
            detail = status_text or "Remote A2A agent requires additional input."
            raise A2AExecutionError(
                detail,
                task_id=self.task_id,
                context_id=self.context_id,
                state=state_name,
            )

        artifact_summaries: list[dict[str, Any]] = []
        artifact_texts: list[str] = []
        collected_data: list[Any] = []

        for key in self._artifact_order:
            parts = self._artifact_parts[key]
            part_text = _parts_text(parts)
            part_data = get_data_parts(parts)
            if part_text:
                artifact_texts.append(part_text)
            collected_data.extend(part_data)
            artifact_summaries.append(
                {
                    "artifact_id": None if key == "__default__" else key,
                    "name": self._artifact_names.get(key) or None,
                    "text": part_text,
                    "data": part_data,
                }
            )

        if not artifact_summaries:
            fallback_parts = self._status_message_parts or self._direct_message_parts
            if fallback_parts:
                collected_data.extend(get_data_parts(fallback_parts))
        elif status_text and self._status_message_parts:
            collected_data.extend(get_data_parts(self._status_message_parts))

        combined_text = "\n".join(artifact_texts)
        if not combined_text:
            combined_text = status_text or _parts_text(self._direct_message_parts)
        if not combined_text and collected_data:
            combined_text = json.dumps(
                collected_data[-1] if len(collected_data) == 1 else collected_data,
                ensure_ascii=False,
            )

        return A2ACollectedResult(
            text=combined_text,
            data=collected_data,
            artifacts=artifact_summaries,
            task_id=self.task_id,
            context_id=self.context_id,
            state=self.state,
            is_input_required=is_input_required,
        )


def a2a_result_to_langchain_output(
    collected: A2ACollectedResult,
    original_input: Any,
    *,
    output_key: str | None = None,
) -> Any:
    """Convert an `A2ACollectedResult` into a LangChain/LangGraph output value.

    Args:
        collected: Collected response from A2A execution.
        original_input: Original input passed by the caller or parent `StateGraph`.
        output_key: Optional explicit dictionary key to populate.

    Returns:
        Converted output matching the caller's shape (`str`, `AIMessage`, or `dict`).
    """
    last_dict: dict[str, Any] | None = None
    for item in reversed(collected.data):
        if isinstance(item, Mapping):
            last_dict = {str(k): v for k, v in item.items()}
            break

    if output_key is not None:
        if collected.text:
            return {output_key: collected.text}
        if collected.data:
            return {output_key: collected.data[-1]}
        return {output_key: ""}

    if isinstance(original_input, str):
        return collected.text

    if isinstance(original_input, BaseMessage):
        return AIMessage(content=collected.text)

    if isinstance(original_input, Command):
        if last_dict is not None and "structured_response" not in last_dict:
            return {
                k: v
                for k, v in last_dict.items()
                if k not in {"messages", "__interrupts__", "__langgraph_resume__"}
            }
        return {"messages": [AIMessage(content=collected.text)]}

    if isinstance(original_input, Mapping) and "messages" not in original_input:
        if last_dict is not None:
            return {
                k: v
                for k, v in last_dict.items()
                if k not in {"__interrupts__", "__langgraph_resume__"}
            }
        return {"output": collected.text}

    output: dict[str, Any] = {"messages": [AIMessage(content=collected.text)]}
    if last_dict is not None:
        output.update(
            {
                k: v
                for k, v in last_dict.items()
                if k not in {"messages", "__interrupts__", "__langgraph_resume__"}
            }
        )
    return output


def _ensure_card_interfaces(card: AgentCard, fallback_url: str) -> AgentCard:
    """Attach a default JSON-RPC interface if an `AgentCard` defines none."""
    if card.supported_interfaces:
        return card
    copied = AgentCard()
    copied.CopyFrom(card)
    copied.supported_interfaces.append(
        AgentInterface(
            url=fallback_url.rstrip("/"),
            protocol_binding=TransportProtocol.JSONRPC,
            protocol_version=PROTOCOL_VERSION_CURRENT,
        )
    )
    return copied


def resolve_agent_card_sync(
    agent: str | AgentCard,
    *,
    httpx_client: httpx.Client | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> AgentCard:
    """Synchronously resolve an `AgentCard` from a URL or existing `AgentCard`.

    Args:
        agent: Base URL, full agent-card JSON URL, or pre-built `AgentCard`.
        httpx_client: Optional caller-managed synchronous `httpx.Client`.
        headers: Optional HTTP headers sent when fetching the agent card.
        timeout: Optional HTTP request timeout in seconds.

    Returns:
        Resolved A2A `AgentCard`.
    """
    if isinstance(agent, AgentCard):
        return agent

    base_url, relative_card_path = split_agent_card_url(agent)
    card_path = relative_card_path or ".well-known/agent-card.json"
    card_url = join_url(base_url, card_path)

    owns_client = httpx_client is None
    client = httpx_client or httpx.Client(
        headers=dict(headers) if headers else None,
        timeout=timeout if timeout is not None else 30.0,
    )
    try:
        response = client.get(card_url, headers=dict(headers) if headers else None)
        response.raise_for_status()
        card = parse_agent_card(response.json())
    finally:
        if owns_client:
            client.close()

    return _ensure_card_interfaces(card, base_url)


async def resolve_agent_card(
    agent: str | AgentCard,
    *,
    httpx_client: httpx.AsyncClient | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> AgentCard:
    """Asynchronously resolve an `AgentCard` from a URL or existing `AgentCard`.

    Args:
        agent: Base URL, full agent-card JSON URL, or pre-built `AgentCard`.
        httpx_client: Optional caller-managed `httpx.AsyncClient`.
        headers: Optional HTTP headers sent when fetching the agent card.
        timeout: Optional HTTP request timeout in seconds.

    Returns:
        Resolved A2A `AgentCard`.
    """
    if isinstance(agent, AgentCard):
        return agent

    base_url, relative_card_path = split_agent_card_url(agent)
    owns_client = httpx_client is None
    client = httpx_client or httpx.AsyncClient(
        headers=dict(headers) if headers else None,
        timeout=timeout if timeout is not None else 30.0,
    )
    try:
        resolver = (
            A2ACardResolver(client, base_url, agent_card_path=relative_card_path)
            if relative_card_path is not None
            else A2ACardResolver(client, base_url)
        )
        http_kwargs = {"headers": dict(headers)} if headers else None
        card = await resolver.get_agent_card(http_kwargs=http_kwargs)
    finally:
        if owns_client:
            await client.aclose()

    return _ensure_card_interfaces(card, base_url)


async def create_a2a_client(
    card: AgentCard,
    *,
    client_config: ClientConfig | None = None,
    httpx_client: httpx.AsyncClient | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> tuple[Client, httpx.AsyncClient | None]:
    """Build an A2A `Client` together with any ephemeral `httpx.AsyncClient` to close.

    Args:
        card: Resolved A2A `AgentCard`.
        client_config: Optional pre-configured `ClientConfig`.
        httpx_client: Optional shared `httpx.AsyncClient` (never closed by us).
        headers: Optional HTTP headers used when building an ephemeral client.
        timeout: Optional HTTP timeout used when building an ephemeral client.

    Returns:
        Tuple of `(a2a_client, owned_httpx_client)`.
    """
    owned_http: httpx.AsyncClient | None = None
    if client_config is not None:
        if client_config.httpx_client is None and httpx_client is not None:
            resolved_config = ClientConfig(
                streaming=client_config.streaming,
                polling=client_config.polling,
                httpx_client=httpx_client,
                supported_protocol_bindings=list(client_config.supported_protocol_bindings),
                use_client_preference=client_config.use_client_preference,
                accepted_output_modes=list(client_config.accepted_output_modes),
            )
        else:
            resolved_config = client_config
    else:
        effective_http = httpx_client
        if effective_http is None:
            owned_http = httpx.AsyncClient(
                headers=dict(headers) if headers else None,
                timeout=timeout if timeout is not None else 60.0,
            )
            effective_http = owned_http
        resolved_config = ClientConfig(httpx_client=effective_http)

    a2a_client = await create_client(card, client_config=resolved_config)
    return a2a_client, owned_http


async def stream_a2a_events(
    card: AgentCard,
    message: Message,
    *,
    client_config: ClientConfig | None = None,
    httpx_client: httpx.AsyncClient | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> AsyncIterator[StreamResponse]:
    """Send an A2A message and stream `StreamResponse` events with cleanup.

    Args:
        card: Resolved target `AgentCard`.
        message: Outbound A2A `Message`.
        client_config: Optional `ClientConfig`.
        httpx_client: Optional shared `httpx.AsyncClient`.
        headers: Optional HTTP headers for ephemeral HTTP clients.
        timeout: Optional timeout in seconds for ephemeral HTTP clients.

    Yields:
        `StreamResponse` items from the remote A2A agent.
    """
    a2a_client, owned_http = await create_a2a_client(
        card,
        client_config=client_config,
        httpx_client=httpx_client,
        headers=headers,
        timeout=timeout,
    )
    try:
        request = SendMessageRequest(message=message)
        async for event in a2a_client.send_message(request):
            yield event
    finally:
        try:
            await a2a_client.close()
        finally:
            if owned_http is not None:
                await owned_http.aclose()


def run_async_safely(coro_factory: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
    """Execute an async coroutine safely from synchronous code.

    Falls back to a dedicated worker thread when called from within a running
    event loop so callers never encounter `asyncio.run() cannot be called from
    a running event loop`.

    Args:
        coro_factory: Zero-argument callable returning a fresh coroutine.

    Returns:
        Result returned by the coroutine.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro_factory())).result()


def async_iter_to_sync(
    aiter_factory: Callable[[], AsyncIterator[_T]],
) -> Iterator[_T]:
    """Bridge an async iterator into a synchronous Python iterator cleanly.

    Args:
        aiter_factory: Zero-argument factory returning an `AsyncIterator`.

    Yields:
        Items yielded by the underlying async iterator.
    """
    item_queue: queue.Queue[object] = queue.Queue()
    stop_event = False

    async def _pump() -> None:
        stream_iter = aiter_factory()
        try:
            async for item in stream_iter:
                if stop_event:
                    break
                item_queue.put(item)
        except BaseException as exc:
            item_queue.put(exc)
        finally:
            aclose = getattr(stream_iter, "aclose", None)
            if callable(aclose):
                await aclose()
            item_queue.put(_STREAM_DONE)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(_pump()))
        try:
            while True:
                item = item_queue.get()
                if item is _STREAM_DONE:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item  # type: ignore[misc]
        finally:
            stop_event = True
            future.result()

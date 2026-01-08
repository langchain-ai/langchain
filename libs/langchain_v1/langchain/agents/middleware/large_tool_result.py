"""Large tool result middleware.

Offloads large tool results to the filesystem to prevent context overflow.
Results exceeding a configurable threshold are written to temporary files
with a truncated preview kept in the message.
"""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AnyMessage, ToolMessage
from typing_extensions import NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.runtime import Runtime

__all__ = ["LargeToolResultMiddleware"]

_DEFAULT_THRESHOLD_FRACTION = 0.10
_DEFAULT_PREVIEW_LENGTH = 500
_OFFLOAD_METADATA_KEY = "large_tool_result_offloaded"

ContextFraction = tuple[Literal["fraction"], float]
ContextTokens = tuple[Literal["tokens"], int]
ContextSize = ContextFraction | ContextTokens


class LargeToolResultState(AgentState):
    """Extended state for large tool result middleware."""

    offloaded_results_dir: NotRequired[Annotated[str | None, PrivateStateAttr]]
    """Directory where large tool results are stored."""


class LargeToolResultMiddleware(AgentMiddleware[LargeToolResultState, Any]):
    """Offloads large tool results to filesystem to prevent context overflow.

    This middleware monitors tool result sizes and automatically writes results
    exceeding a threshold to temporary files, keeping a truncated preview in the
    message. This prevents massive tool outputs from triggering aggressive
    summarization or exceeding model context limits.

    The middleware uses `wrap_model_call` to process messages before each model
    invocation, ensuring large results are offloaded before token counting occurs.
    """

    state_schema = LargeToolResultState
    tools: Sequence = ()

    def __init__(
        self,
        *,
        threshold: ContextSize = ("fraction", _DEFAULT_THRESHOLD_FRACTION),
        preview_length: int = _DEFAULT_PREVIEW_LENGTH,
        temp_dir: Path | str | None = None,
        cleanup_on_end: bool = True,
    ) -> None:
        """Initialize large tool result middleware.

        Args:
            threshold: Size threshold that triggers offloading to disk.

                Provide a tuple specifying the threshold type:

                - `('fraction', 0.10)`: Offload if result exceeds 10% of model's
                    max input tokens (default)
                - `('tokens', 5000)`: Offload if result exceeds 5000 tokens

            preview_length: Number of characters to keep as preview in the message.
                Defaults to 500 characters.

            temp_dir: Directory for storing offloaded results.

                If `None` (default), creates a temporary directory that is cleaned
                up when the agent session ends.

                If provided, uses the specified directory and does not delete it
                on cleanup (user-managed).

            cleanup_on_end: Whether to clean up the temp directory when the agent
                session ends.

                Only applies when `temp_dir` is `None` (auto-created directory).

                Defaults to `True`.
        """
        super().__init__()
        self.threshold = self._validate_threshold(threshold)
        self.preview_length = preview_length
        self.user_temp_dir = Path(temp_dir) if temp_dir else None
        self.cleanup_on_end = cleanup_on_end
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def _validate_threshold(self, threshold: ContextSize) -> ContextSize:
        """Validate threshold configuration."""
        kind, value = threshold
        if kind == "fraction":
            if not 0 < value <= 1:
                msg = f"Fractional threshold must be between 0 and 1, got {value}."
                raise ValueError(msg)
        elif kind == "tokens":
            if value <= 0:
                msg = f"Token threshold must be greater than 0, got {value}."
                raise ValueError(msg)
        else:
            msg = f"Unsupported threshold type: {kind}"
            raise ValueError(msg)
        return threshold

    def _get_threshold_tokens(self, model: BaseChatModel | None) -> int:
        """Calculate threshold in tokens based on configuration."""
        kind, value = self.threshold
        if kind == "tokens":
            return int(value)

        # Fractional threshold - need model profile
        if model is None:
            # Fallback to character-based estimate (4 chars per token)
            return int(value * 100_000)

        max_input_tokens = self._get_model_max_tokens(model)
        if max_input_tokens is None:
            # Fallback
            return int(value * 100_000)

        return int(max_input_tokens * value)

    def _get_model_max_tokens(self, model: BaseChatModel) -> int | None:
        """Get model's max input tokens from profile."""
        try:
            profile = model.profile
        except AttributeError:
            return None

        if not isinstance(profile, Mapping):
            return None

        max_input_tokens = profile.get("max_input_tokens")
        return max_input_tokens if isinstance(max_input_tokens, int) else None

    def _get_or_create_temp_dir(self) -> Path:
        """Get or create the temporary directory for storing results."""
        if self.user_temp_dir is not None:
            self.user_temp_dir.mkdir(parents=True, exist_ok=True)
            return self.user_temp_dir

        if self._temp_dir is not None:
            return Path(self._temp_dir.name)

        # Create new temp directory
        self._temp_dir = tempfile.TemporaryDirectory(prefix="langchain-large-results-")
        return Path(self._temp_dir.name)

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for a string."""
        # Approximate: ~4 characters per token
        return len(content) // 4

    def _should_offload(self, content: str, threshold_tokens: int) -> bool:
        """Determine if content should be offloaded based on size."""
        estimated_tokens = self._estimate_tokens(content)
        return estimated_tokens > threshold_tokens

    def _offload_content(self, tool_call_id: str, content: str) -> str:
        """Write content to file and return the file path."""
        temp_dir = self._get_or_create_temp_dir()
        # Sanitize tool_call_id for filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in tool_call_id)
        file_path = temp_dir / f"{safe_id}.txt"

        file_path.write_text(content, encoding="utf-8")
        return str(file_path)

    def _create_truncated_message(self, original_content: str, file_path: str) -> str:
        """Create a truncated message with file reference."""
        preview = original_content[: self.preview_length]
        if len(original_content) > self.preview_length:
            preview += "..."

        return (
            f"[TRUNCATED - Full result saved to: {file_path}]\n\n"
            f"Preview (first {self.preview_length} chars):\n{preview}"
        )

    def _process_messages(
        self,
        messages: list[AnyMessage],
        threshold_tokens: int,
    ) -> list[AnyMessage]:
        """Process messages and offload large tool results."""
        processed: list[AnyMessage] = []
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                processed.append(msg)
                continue

            # Check if already offloaded
            if msg.response_metadata.get(_OFFLOAD_METADATA_KEY):
                processed.append(msg)
                continue

            # Get content as string
            content = msg.content
            if isinstance(content, list):
                # Multimodal content - convert to string for size check
                content = str(content)
            if not isinstance(content, str):
                content = str(content)

            # Check if should offload
            if not self._should_offload(content, threshold_tokens):
                processed.append(msg)
                continue

            # Offload to file
            tool_call_id = msg.tool_call_id or f"unknown_{id(msg)}"
            file_path = self._offload_content(tool_call_id, content)

            # Create truncated message
            truncated_content = self._create_truncated_message(content, file_path)

            # Create new message with truncated content
            new_msg = msg.model_copy(
                update={
                    "content": truncated_content,
                    "response_metadata": {
                        **msg.response_metadata,
                        _OFFLOAD_METADATA_KEY: {
                            "offloaded": True,
                            "file_path": file_path,
                            "original_size_chars": len(content),
                        },
                    },
                }
            )
            processed.append(new_msg)

        return processed

    @override
    def before_agent(
        self, state: LargeToolResultState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Initialize temp directory tracking in state."""
        # Just return None - we'll create temp dir lazily when needed
        return None

    @override
    async def abefore_agent(
        self, state: LargeToolResultState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Initialize temp directory tracking in state (async)."""
        return None

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Process messages before model call, offloading large tool results."""
        if not request.messages:
            return handler(request)

        threshold_tokens = self._get_threshold_tokens(request.model)
        processed_messages = self._process_messages(
            deepcopy(list(request.messages)),
            threshold_tokens,
        )

        return handler(request.override(messages=processed_messages))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Process messages before model call, offloading large tool results (async)."""
        if not request.messages:
            return await handler(request)

        threshold_tokens = self._get_threshold_tokens(request.model)
        processed_messages = self._process_messages(
            deepcopy(list(request.messages)),
            threshold_tokens,
        )

        return await handler(request.override(messages=processed_messages))

    @override
    def after_agent(
        self, state: LargeToolResultState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Clean up temp directory on agent completion."""
        if self.cleanup_on_end and self._temp_dir is not None:
            with contextlib.suppress(Exception):
                self._temp_dir.cleanup()
            self._temp_dir = None
        return None

    @override
    async def aafter_agent(
        self, state: LargeToolResultState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Clean up temp directory on agent completion (async)."""
        if self.cleanup_on_end and self._temp_dir is not None:
            with contextlib.suppress(Exception):
                self._temp_dir.cleanup()
            self._temp_dir = None
        return None

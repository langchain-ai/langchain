"""Drop input content blocks the active model's profile marks unsupported."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from langchain_core.messages import HumanMessage, ToolMessage

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
    TracePolicy,
    omit_payload,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage, ContentBlock

_PROFILE_FIELD_BY_BLOCK_TYPE: Final[Mapping[str, str]] = {
    "image": "image_inputs",
    "audio": "audio_inputs",
    "video": "video_inputs",
    "file": "pdf_inputs",
}
"""`ModelProfile` field gating each block type outside of a `ToolMessage`."""

_TOOL_MESSAGE_FIELD_BY_BLOCK_TYPE: Final[Mapping[str, str]] = {
    "image": "image_tool_message",
    "file": "pdf_tool_message",
}
"""Additional `ModelProfile` field gating a block type inside a `ToolMessage`."""

_PDF_MIME_TYPE: Final = "application/pdf"


class UnsupportedContentMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Replace multimodal input blocks the active model can't accept with a text notice.

    Support is read from
    [`model.profile`](https://docs.langchain.com/oss/python/langchain/models#model-profiles).

    Place this middleware last in the `middleware` list, so that if
    `ModelRequest.model` changes, this middleware will apply to the correct one.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import UnsupportedContentMiddleware

        agent = create_agent(model, middleware=[*other_middleware, UnsupportedContentMiddleware()])
        ```
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    def _is_supported(
        self,
        block: ContentBlock,
        *,
        model: BaseChatModel,
        in_tool_message: bool,
    ) -> bool:
        """Return whether `model` accepts `block`.

        A model with no profile is read as an empty profile rather than skipped, so
        overrides gating on something other than profile data still run.

        Args:
            block: The input content block under consideration.
            model: The model the request will reach. Overrides can gate on details no
                profile field covers yet, such as the provider class.
            in_tool_message: Whether `block` sits in a `ToolMessage`, which some
                providers gate separately from ordinary input.

        Returns:
            `True` unless a profile field explicitly rejects the block.
        """
        profile = model.profile or {}
        block_type = block["type"]
        field = _PROFILE_FIELD_BY_BLOCK_TYPE.get(block_type)
        if field is None:
            return True
        if block_type == "file" and (
            "base64" not in block or block.get("mime_type") != _PDF_MIME_TYPE
        ):
            # URL- and file-ID-backed references are provider-managed, and no profile
            # field describes non-PDF payloads (`.docx`, `.pptx`, ...).
            return True
        if in_tool_message:
            tool_field = _TOOL_MESSAGE_FIELD_BY_BLOCK_TYPE.get(block_type)
            if tool_field is not None and profile.get(tool_field) is False:
                return False
        return profile.get(field) is not False

    def _replace(
        self,
        block: ContentBlock,
        message: AnyMessage,  # noqa: ARG002  # unused by the default text; here for overrides
    ) -> ContentBlock:
        """Build the text block replacing a `block` the active model can't accept.

        Args:
            block: The block the active model rejects.
            message: The message carrying `block`. Passed so overrides can describe
                where the content came from.

        Returns:
            The replacement content block.
        """
        return cast(
            "ContentBlock",
            {
                "type": "text",
                "text": f"[{block['type']} content omitted: unsupported by this model]",
            },
        )

    def _filter_message(self, message: AnyMessage, *, model: BaseChatModel) -> AnyMessage:
        """Return `message`, or a copy with unsupported blocks replaced."""
        in_tool_message = isinstance(message, ToolMessage)
        blocks = message.content_blocks
        new_blocks = [
            block
            if self._is_supported(block, model=model, in_tool_message=in_tool_message)
            else self._replace(block, message)
            for block in blocks
        ]
        if new_blocks == blocks:
            return message
        return message.model_copy(update={"content": new_blocks})

    def _filter_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Return `request`, or an override whose messages the active model accepts."""
        model = request.model
        messages: list[AnyMessage] = []
        changed = False
        for message in request.messages:
            if not isinstance(message, (HumanMessage, ToolMessage)):
                messages.append(message)
                continue
            filtered = self._filter_message(message, model=model)
            changed = changed or filtered is not message
            messages.append(filtered)
        return request.override(messages=messages) if changed else request

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Filter unsupported input blocks, then invoke the model.

        Args:
            request: Model request to execute.
            handler: Callback that executes the model request.

        Returns:
            The result of invoking the handler.
        """
        return handler(self._filter_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Filter unsupported input blocks, then invoke the model.

        Args:
            request: Model request to execute.
            handler: Async callback that executes the model request.

        Returns:
            The result of invoking the handler.
        """
        return await handler(self._filter_request(request))

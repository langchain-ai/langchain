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
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from langchain_core.language_models import BaseChatModel
    from langchain_core.language_models.model_profile import ModelProfile
    from langchain_core.messages import AnyMessage, ContentBlock

    OnUnsupported = Callable[[ContentBlock, AnyMessage], ContentBlock | str | None]
    """Builds the stand-in for an unsupported block. Return `None` to drop it outright."""

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

        Customize the notice the model sees:

        ```python
        def on_unsupported(block: ContentBlock, message: AnyMessage) -> str:
            return f"[{block['type']} attachment dropped — switch models to view it.]"


        UnsupportedContentMiddleware(on_unsupported=on_unsupported)
        ```
    """

    def __init__(self, *, on_unsupported: OnUnsupported | None = None) -> None:
        """Initialize `UnsupportedContentMiddleware`.

        Args:
            on_unsupported: Builds the replacement for an unsupported block, receiving the
                block and the message carrying it. Return a content block or a string to
                substitute it, or `None` to drop the block without a trace. Defaults to a
                text notice naming the block type.
        """
        super().__init__()
        self.on_unsupported = on_unsupported

    def is_supported(
        self,
        block: ContentBlock,
        *,
        model: BaseChatModel,  # noqa: ARG002  # unused by the profile gate; here for overrides
        profile: ModelProfile,
        in_tool_message: bool,
    ) -> bool:
        """Return whether `model` accepts `block`.

        Args:
            block: The input content block under consideration.
            model: The model the request will reach. Passed so overrides can gate on
                details no profile field covers yet, such as the provider class.
            profile: `model.profile`, or an empty mapping when the model has none.
            in_tool_message: Whether `block` sits in a `ToolMessage`, which some
                providers gate separately from ordinary input.

        Returns:
            `True` unless a profile field explicitly rejects the block.
        """
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
        if block_type == "image" and "url" in block and profile.get("image_url_inputs") is False:
            return False
        if in_tool_message:
            tool_field = _TOOL_MESSAGE_FIELD_BY_BLOCK_TYPE.get(block_type)
            if tool_field is not None and profile.get(tool_field) is False:
                return False
        return profile.get(field) is not False

    def replace(self, block: ContentBlock, message: AnyMessage) -> ContentBlock | None:
        """Build the stand-in for an unsupported `block`.

        Args:
            block: The block the active model rejects.
            message: The message carrying `block`.

        Returns:
            A replacement content block, or `None` to drop the block.
        """
        replacement = (
            self.on_unsupported(block, message)
            if self.on_unsupported is not None
            else f"[{block['type']} content omitted: unsupported by this model]"
        )
        if isinstance(replacement, str):
            return cast("ContentBlock", {"type": "text", "text": replacement})
        return replacement

    def _filter_message(
        self,
        message: AnyMessage,
        *,
        model: BaseChatModel,
        profile: ModelProfile,
    ) -> AnyMessage:
        """Return `message`, or a copy with unsupported blocks replaced."""
        in_tool_message = isinstance(message, ToolMessage)
        blocks = message.content_blocks
        new_blocks: list[ContentBlock] = []
        changed = False
        for block in blocks:
            if self.is_supported(
                block, model=model, profile=profile, in_tool_message=in_tool_message
            ):
                new_blocks.append(block)
                continue
            changed = True
            if (replacement := self.replace(block, message)) is not None:
                new_blocks.append(replacement)
        if not changed:
            return message
        return message.model_copy(update={"content": new_blocks})

    def _filter_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Return `request`, or an override whose messages the active model accepts."""
        model = request.model
        profile = model.profile or cast("ModelProfile", {})
        messages: list[AnyMessage] = []
        changed = False
        for message in request.messages:
            # String content can only hold text, so it never needs filtering. Skipping it
            # also avoids parsing `content_blocks` for the bulk of a long history.
            if isinstance(message.content, str) or not isinstance(
                message, (HumanMessage, ToolMessage)
            ):
                messages.append(message)
                continue
            filtered = self._filter_message(message, model=model, profile=profile)
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

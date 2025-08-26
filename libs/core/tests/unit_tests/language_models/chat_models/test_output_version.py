"""Test output_version functionality in BaseChatModel."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional
from unittest.mock import patch

import pytest
from pydantic import ConfigDict
from typing_extensions import override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class OutputVersionTrackingChatModel(GenericFakeChatModel):
    """Chat model that tracks output_version parameter for testing."""

    model_config = ConfigDict(extra="allow")
    last_output_version: Optional[str] = None

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        output_version: str = "v0",
        **kwargs: Any,
    ) -> ChatResult:
        """Store the output_version that was passed."""
        self.last_output_version = output_version
        message = AIMessage(content="test response")
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        output_version: str = "v0",
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Store the output_version that was passed."""
        self.last_output_version = output_version
        yield ChatGenerationChunk(message=AIMessageChunk(content="test"))
        yield ChatGenerationChunk(message=AIMessageChunk(content=" stream"))

    @override
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,  # type: ignore[override]
        *,
        output_version: str = "v0",
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Store the output_version that was passed."""
        self.last_output_version = output_version
        yield ChatGenerationChunk(message=AIMessageChunk(content="async"))
        yield ChatGenerationChunk(message=AIMessageChunk(content=" stream"))


@pytest.fixture
def messages() -> list[BaseMessage]:
    return [HumanMessage("Hello")]


class TestOutputVersionPassing:
    """Test that output_version parameter is correctly passed to model methods."""

    @pytest.mark.parametrize(
        ("method_name", "default_version", "provided_version", "expected_version"),
        [
            # Test invoke
            ("invoke", "v1", None, "v1"),  # Uses default when not provided
            ("invoke", "v0", "v1", "v1"),  # Uses provided version
            # Test stream
            ("stream", "v1", None, "v1"),  # Uses default when not provided
            ("stream", "v1", "v2", "v2"),  # Uses provided version
        ],
    )
    def test_sync_methods_output_version(
        self,
        messages: list[BaseMessage],
        method_name: str,
        default_version: str,
        provided_version: Optional[str],
        expected_version: str,
    ) -> None:
        """Test sync methods handle output_version correctly."""
        model = OutputVersionTrackingChatModel(
            messages=iter(["test response"]), output_version=default_version
        )
        method = getattr(model, method_name)

        if provided_version is not None:
            if method_name == "stream":
                list(method(messages, output_version=provided_version))
            else:
                method(messages, output_version=provided_version)
        elif method_name == "stream":
            list(method(messages))
        else:
            method(messages)

        assert model.last_output_version == expected_version

    @pytest.mark.parametrize(
        ("method_name", "default_version", "provided_version", "expected_version"),
        [
            # Test ainvoke
            ("ainvoke", "v1", None, "v1"),  # Uses default when not provided
            ("ainvoke", "v0", "v1", "v1"),  # Uses provided version
            # Test astream
            ("astream", "v1", None, "v1"),  # Uses default when not provided
            ("astream", "v1", "v0", "v0"),  # Uses provided version
        ],
    )
    async def test_async_methods_output_version(
        self,
        messages: list[BaseMessage],
        method_name: str,
        default_version: str,
        provided_version: Optional[str],
        expected_version: str,
    ) -> None:
        """Test async methods handle output_version correctly."""
        model = OutputVersionTrackingChatModel(
            messages=iter(["test response"]), output_version=default_version
        )
        method = getattr(model, method_name)

        if provided_version is not None:
            if method_name == "astream":
                async for _ in method(messages, output_version=provided_version):
                    pass
            else:
                await method(messages, output_version=provided_version)
        elif method_name == "astream":
            async for _ in method(messages):
                pass
        else:
            await method(messages)

        assert model.last_output_version == expected_version


class TestStreamFallback:
    """Test stream fallback behavior with output_version."""

    def test_stream_fallback_to_invoke_passes_output_version(
        self,
        messages: list[BaseMessage],
    ) -> None:
        """Test `stream()` fallback passes `output_version` correctly."""

        class NoStreamModel(BaseChatModel):
            model_config = ConfigDict(extra="allow")
            last_output_version: Optional[str] = None

            @override
            def _generate(
                self,
                messages: list[BaseMessage],
                stop: Optional[list[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                *,
                output_version: str = "v0",
                **kwargs: Any,
            ) -> ChatResult:
                self.last_output_version = output_version
                message = AIMessage(content="test response")
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])

            @property
            def _llm_type(self) -> str:
                return "no-stream-model"

        model = NoStreamModel(output_version="v1")
        # Stream should fallback to invoke and pass the output_version
        list(model.stream(messages, output_version="v2"))
        assert model.last_output_version == "v2"

    async def test_astream_fallback_to_ainvoke_passes_output_version(
        self,
        messages: list[BaseMessage],
    ) -> None:
        """Test `astream()` fallback passes `output_version` correctly."""

        class NoStreamModel(BaseChatModel):
            model_config = ConfigDict(extra="allow")
            last_output_version: Optional[str] = None

            @override
            def _generate(
                self,
                messages: list[BaseMessage],
                stop: Optional[list[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                *,
                output_version: str = "v0",
                **kwargs: Any,
            ) -> ChatResult:
                self.last_output_version = output_version
                message = AIMessage(content="test response")
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])

            @property
            def _llm_type(self) -> str:
                return "no-stream-model"

        model = NoStreamModel(output_version="v1")
        # astream should fallback to ainvoke and pass the output_version
        async for _ in model.astream(messages, output_version="v2"):
            pass
        assert model.last_output_version == "v2"


class TestOutputVersionInMessages:
    """Test output_version is added to message additional_kwargs."""

    def test_output_version_added_to_message_additional_kwargs(
        self,
        messages: list[BaseMessage],
    ) -> None:
        """Test that output_version is added to message additional_kwargs."""
        model = OutputVersionTrackingChatModel(
            messages=iter(["test response"]), output_version="v1"
        )
        result = model.invoke(messages, output_version="v2")
        assert result.additional_kwargs["output_version"] == "v2"

    def test_output_version_added_to_stream_message_additional_kwargs(
        self,
        messages: list[BaseMessage],
    ) -> None:
        """Test that output_version is added to streamed message additional_kwargs."""
        model = OutputVersionTrackingChatModel(
            messages=iter(["test response"]), output_version="v1"
        )
        chunks = list(model.stream(messages, output_version="v2"))

        # Check that content chunks (not the "last" chunk) have the output_version
        content_chunks = [chunk for chunk in chunks if chunk.content]
        assert len(content_chunks) >= 1  # Should have at least one content chunk

        for chunk in content_chunks:
            assert "output_version" in chunk.additional_kwargs
            assert chunk.additional_kwargs["output_version"] == "v2"

    async def test_output_version_added_to_astream_message_additional_kwargs(
        self,
        messages: list[BaseMessage],
    ) -> None:
        """Test output_version added to async streamed additional_kwargs."""
        model = OutputVersionTrackingChatModel(
            messages=iter(["test response"]), output_version="v1"
        )
        chunks = [chunk async for chunk in model.astream(messages, output_version="v2")]

        # Check that content chunks (not the "last" chunk) have the output_version
        content_chunks = [chunk for chunk in chunks if chunk.content]
        assert len(content_chunks) >= 1  # Should have at least one content chunk

        for chunk in content_chunks:
            assert "output_version" in chunk.additional_kwargs
            assert chunk.additional_kwargs["output_version"] == "v2"


class TestOutputVersionMerging:
    """Test output_version handling in merge operations."""

    def test_output_version_consistency_in_merge(self) -> None:
        """Test that merge_dicts raises error for inconsistent output_version."""
        from langchain_core.utils._merge import merge_dicts

        left_dict = {"output_version": "v1"}
        right_dict = {"output_version": "v2"}

        with pytest.raises(ValueError, match="Unable to merge.*output_version"):
            merge_dicts(left_dict, right_dict)

    def test_output_version_merge_same_value(self) -> None:
        """Test that merge_dicts works fine when output_version values are same."""
        from langchain_core.utils._merge import merge_dicts

        left_dict = {"output_version": "v1", "other": "data1"}
        right_dict = {"output_version": "v1", "more": "data2"}

        result = merge_dicts(left_dict, right_dict)
        assert result["output_version"] == "v1"
        assert result["other"] == "data1"
        assert result["more"] == "data2"


class TestBackwardsCompatibility:
    """Test backwards compatibility features."""

    def test_backwards_compatibility_with_v0_default(
        self,
        messages: list[BaseMessage],
    ) -> None:
        """Test that models default to v0 for backward compatibility."""
        model = OutputVersionTrackingChatModel(
            messages=iter(["test response"])
        )  # Don't specify output_version
        model.invoke(messages)
        # The default should be v0 for backward compatibility
        assert model.last_output_version == "v0"

    def test_output_version_preserved_through_chain_calls(
        self,
        messages: list[BaseMessage],
    ) -> None:
        """Test that output_version is preserved through internal method calls."""
        model = OutputVersionTrackingChatModel(
            messages=iter(["test response"]), output_version="v1"
        )

        # Test both with explicit and implicit (None) output_version
        with patch.object(
            model, "_generate_with_cache", wraps=model._generate_with_cache
        ) as mock_cache:
            model.invoke(messages, output_version="v2")
            # Verify the internal call received the right output_version
            mock_cache.assert_called_once()
            call_kwargs = mock_cache.call_args[1]
            assert call_kwargs.get("_output_version") == "v2"

        # Verify the model implementation received the correct output_version
        assert model.last_output_version == "v2"

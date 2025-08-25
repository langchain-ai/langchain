"""LangChain chat model integration for Llama Stack."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Union

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env

try:
    from llama_stack_client import LlamaStackClient
except ImportError:
    LlamaStackClient = None
    logger.warning(
        "llama-stack-client not available. Only direct Ollama fallback will be used."
    )

logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)


def _convert_langchain_message_to_llamastack(message: BaseMessage) -> Dict[str, Any]:
    """Convert a LangChain message to Llama Stack format."""
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")


def _convert_llamastack_message_to_langchain(message: Dict[str, Any]) -> BaseMessage:
    """Convert a Llama Stack message to LangChain format."""
    role = message["role"]
    content = message["content"]

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        return ChatMessage(role=role, content=content)


class ChatLlamaStack(BaseChatModel):
    """LangChain chat model for Llama Stack."""

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = Field(default="meta-llama/Llama-3.1-8B-Instruct")
    """Model name to use for chat completion."""

    base_url: str = Field(default="http://localhost:8321")
    """Base URL for the Llama Stack server."""

    streaming: bool = Field(default=False)
    """Whether to stream the results."""

    llamastack_available: bool = Field(default=False, exclude=True)
    """Whether LlamaStack is available."""

    available_models: List[str] = Field(default_factory=list, exclude=True)
    """List of available models from LlamaStack."""

    def __init__(self, **kwargs):
        """Initialize ChatLlamaStack."""
        super().__init__(**kwargs)
        self._setup_clients()

    def _setup_clients(self):
        """Setup both LlamaStack client (primary) and Ollama fallback."""
        # Initialize LlamaStack client (primary)
        self.llamastack_available = False
        self.client = None

        if LlamaStackClient is not None:
            try:
                client_kwargs = {"base_url": self.base_url}

                self.client = LlamaStackClient(**client_kwargs)

                # Test the connection by trying to list models
                models = self.client.models.list()
                self.llamastack_available = True
                logger.debug(
                    f"LlamaStack client initialized successfully at {self.base_url}"
                )

            except Exception as e:
                logger.warning(f"LlamaStack client initialization failed: {e}")
                self.llamastack_available = False

        # Get available models from LlamaStack (the only provider)
        self.available_models = (
            self._get_models_from_llamastack() if self.llamastack_available else []
        )

        if not self.available_models:
            logger.warning("No models found in LlamaStack")
            logger.warning(
                "Make sure LlamaStack is running and configured with providers (ollama, together, etc.)"
            )
        else:
            logger.info(
                f"Found {len(self.available_models)} available models from LlamaStack"
            )

    def _get_models_from_llamastack(self) -> List[str]:
        """Get list of available models from LlamaStack."""
        from .utils import list_available_models

        try:
            return list_available_models(base_url=self.base_url)
        except Exception as e:
            logger.warning(f"Failed to get models from LlamaStack: {e}")
            return []

    def _validate_model(self) -> None:
        """Validate and potentially update the model if needed."""
        if self.model not in self.available_models:
            logger.warning(f"Model {self.model} not found in available models")
            if self.available_models:
                logger.info(f"Using first available model: {self.available_models[0]}")
                self.model = self.available_models[0]
            else:
                raise ValueError("No models available in LlamaStack")

    def _prepare_api_params(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Prepare parameters for API call."""
        llamastack_messages = [
            _convert_langchain_message_to_llamastack(msg) for msg in messages
        ]

        params = {
            "model": self.model,
            "messages": llamastack_messages,
        }

        if stop:
            params["stop"] = stop

        params.update(kwargs)
        return params

    def _get_api_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        return headers

    def _handle_api_error(self, response_data: Dict[str, Any]) -> None:
        """Handle API error responses."""
        if "error" not in response_data:
            return

        error_msg = response_data["error"].get("message", str(response_data["error"]))

        if "not found in the routing table" in error_msg:
            available_models_msg = (
                f"Available models: {self.available_models}"
                if self.available_models
                else "No models available"
            )
            raise ValueError(
                f"Model '{self.model}' not found in LlamaStack routing table. "
                f"This usually means:\n"
                f"1. You're using a provider name (like 'ollama') instead of an actual model name\n"
                f"2. The model is not available in the configured providers\n"
                f"3. The provider is not properly configured in LlamaStack\n\n"
                f"{available_models_msg}\n\n"
                f"Try using an actual model name like 'llama3.1:8b' instead of 'ollama'."
            )
        else:
            raise ValueError(f"LlamaStack API error: {error_msg}")

    def _extract_content_from_response(self, response_data: Dict[str, Any]) -> str:
        """Extract content from API response."""
        if "completion" in response_data:
            if isinstance(response_data["completion"], dict):
                return response_data["completion"].get("content", "")
            else:
                return str(response_data["completion"])
        elif "choices" in response_data and response_data["choices"]:
            return response_data["choices"][0].get("message", {}).get("content", "")
        elif "content" in response_data:
            return response_data["content"]
        return ""

    def _extract_content_from_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Extract content from streaming chunk."""
        if "choices" in chunk_data and chunk_data["choices"]:
            delta = chunk_data["choices"][0].get("delta", {})
            return delta.get("content", "")
        elif "completion" in chunk_data:
            return chunk_data["completion"]
        elif "delta" in chunk_data:
            return chunk_data["delta"]
        elif "content" in chunk_data:
            return chunk_data["content"]
        return ""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion using LlamaStack."""
        self._validate_model()

        if not self.llamastack_available:
            raise ValueError(f"LlamaStack is not available at {self.base_url}")

        try:
            logger.info(f"Generating with LlamaStack model: {self.model}")

            params = self._prepare_api_params(messages, stop, **kwargs)
            headers = self._get_api_headers()

            api_url = f"{self.base_url}/v1/openai/v1/chat/completions"

            with httpx.Client(timeout=60.0) as client:
                response = client.post(api_url, json=params, headers=headers)
                response.raise_for_status()
                response_data = response.json()

            self._handle_api_error(response_data)
            content = self._extract_content_from_response(response_data)

            message = AIMessage(content=content)
            generation = ChatGeneration(
                message=message,
                generation_info={
                    "model": self.model,
                    "finish_reason": response_data.get("stop_reason"),
                    "usage": response_data.get("usage"),
                    "provider": "llamastack",
                },
            )

            return ChatResult(generations=[generation])

        except Exception as e:
            logger.error(f"LlamaStack generation failed: {e}")
            raise

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completion using LlamaStack."""
        self._validate_model()

        if not self.llamastack_available:
            raise ValueError(f"LlamaStack is not available at {self.base_url}")

        try:
            logger.info(f"Starting stream for model: {self.model}")

            params = self._prepare_api_params(messages, stop, stream=True, **kwargs)
            headers = self._get_api_headers()

            api_url = f"{self.base_url}/v1/openai/v1/chat/completions"

            with httpx.stream(
                "POST", api_url, json=params, headers=headers
            ) as response:
                response.raise_for_status()
                chunks_yielded = 0

                for line in response.iter_lines():
                    if not line.strip():
                        continue

                    chunk_data = self._parse_streaming_line(line)
                    if not chunk_data:
                        continue

                    self._handle_api_error(chunk_data)
                    content = self._extract_content_from_chunk(chunk_data)

                    if content:
                        chunks_yielded += 1
                        if chunks_yielded % 50 == 0:
                            logger.info(f"Generated {chunks_yielded} tokens...")

                        message = AIMessageChunk(content=content)
                        generation = ChatGenerationChunk(
                            message=message,
                            generation_info={
                                "model": self.model,
                                "finish_reason": None,
                                "provider": "llamastack",
                            },
                        )

                        if run_manager:
                            run_manager.on_llm_new_token(content)

                        yield generation

                logger.info(f"Total chunks yielded: {chunks_yielded}")
                if chunks_yielded == 0:
                    logger.warning("No chunks were yielded from streaming response")

        except Exception as e:
            logger.error(f"LlamaStack streaming failed: {e}")
            raise

    def _parse_streaming_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a streaming response line and return chunk data."""
        try:
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                if data.strip() == "[DONE]":
                    logger.debug("Received [DONE] marker")
                    return None
                return json.loads(data)
            else:
                # Try to parse as raw JSON (non-SSE format)
                return json.loads(line)
        except json.JSONDecodeError:
            logger.debug(f"Skipping non-JSON line: {line}")
            return None

    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "llamastack"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "streaming": self.streaming,
        }

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return self.available_models

    def get_llamastack_info(self) -> Dict[str, Any]:
        """Get LlamaStack connection and model information."""
        return {
            "available": self.llamastack_available,
            "models_count": len(self.available_models),
            "base_url": self.base_url,
            "models": (
                self.available_models[:10]
                if len(self.available_models) > 10
                else self.available_models
            ),  # Show first 10 models
        }

    @classmethod
    def list_available_models(
        cls,
        base_url: str = "http://localhost:8321",
    ) -> List[str]:
        """List available models from LlamaStack without creating a full instance."""
        from .utils import list_available_models

        return list_available_models(base_url)

    @classmethod
    def check_llamastack_connection(
        cls,
        base_url: str = "http://localhost:8321",
    ) -> Dict[str, Any]:
        """Check LlamaStack connection and return status information."""
        from .utils import check_llamastack_connection

        return check_llamastack_connection(base_url)

    def get_model_info(self, model_id: str = None) -> Dict[str, Any]:
        """Get information about a specific model."""
        model_to_check = model_id or self.model
        try:
            models = self.client.models.list()
            for model in models:
                if model.identifier == model_to_check:
                    return {
                        "identifier": model.identifier,
                        "provider_resource_id": getattr(
                            model, "provider_resource_id", None
                        ),
                        "provider_id": getattr(model, "provider_id", None),
                        "model_type": getattr(model, "model_type", None),
                        "metadata": getattr(model, "metadata", {}),
                    }
            return {"error": f"Model {model_to_check} not found"}
        except Exception as e:
            logger.error(f"Error fetching model info: {e}")
            return {"error": str(e)}

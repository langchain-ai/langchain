import asyncio
import json
import logging
from functools import partial
from typing import Any, AsyncIterator, Dict, List, Optional, cast

import requests
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
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)


class PaiEasChatEndpoint(BaseChatModel):
    """Eas LLM Service chat model API.

        To use, must have a deployed eas chat llm service on AliCloud. One can set the
    environment variable ``eas_service_url`` and ``eas_service_token`` set with your eas
    service url and service token.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import PaiEasChatEndpoint
            eas_chat_endpoint = PaiEasChatEndpoint(
                eas_service_url="your_service_url",
                eas_service_token="your_service_token"
            )
    """

    """PAI-EAS Service URL"""
    eas_service_url: str

    """PAI-EAS Service TOKEN"""
    eas_service_token: str

    """PAI-EAS Service Infer Params"""
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.1
    top_k: Optional[int] = 10
    do_sample: Optional[bool] = False
    use_cache: Optional[bool] = True
    stop_sequences: Optional[List[str]] = None

    """Enable stream chat mode."""
    streaming: bool = False

    """Key/value arguments to pass to the model. Reserved for future use"""
    model_kwargs: Optional[dict] = None

    version: Optional[str] = "2.0"

    timeout: Optional[int] = 5000

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["eas_service_url"] = get_from_dict_or_env(
            values, "eas_service_url", "EAS_SERVICE_URL"
        )
        values["eas_service_token"] = get_from_dict_or_env(
            values, "eas_service_token", "EAS_SERVICE_TOKEN"
        )

        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            "eas_service_url": self.eas_service_url,
            "eas_service_token": self.eas_service_token,
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "pai_eas_chat_endpoint"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": [],
            "do_sample": self.do_sample,
            "use_cache": self.use_cache,
        }

    def _invocation_params(
        self, stop_sequences: Optional[List[str]], **kwargs: Any
    ) -> dict:
        params = self._default_params
        if self.model_kwargs:
            params.update(self.model_kwargs)
        if self.stop_sequences is not None and stop_sequences is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop_sequences is not None:
            params["stop"] = self.stop_sequences
        else:
            params["stop"] = stop_sequences
        return {**params, **kwargs}

    def format_request_payload(
        self, messages: List[BaseMessage], **model_kwargs: Any
    ) -> dict:
        prompt: Dict[str, Any] = {}
        user_content: List[str] = []
        assistant_content: List[str] = []

        for message in messages:
            """Converts message to a dict according to role"""
            content = cast(str, message.content)
            if isinstance(message, HumanMessage):
                user_content = user_content + [content]
            elif isinstance(message, AIMessage):
                assistant_content = assistant_content + [content]
            elif isinstance(message, SystemMessage):
                prompt["system_prompt"] = content
            elif isinstance(message, ChatMessage) and message.role in [
                "user",
                "assistant",
                "system",
            ]:
                if message.role == "system":
                    prompt["system_prompt"] = content
                elif message.role == "user":
                    user_content = user_content + [content]
                elif message.role == "assistant":
                    assistant_content = assistant_content + [content]
            else:
                supported = ",".join([role for role in ["user", "assistant", "system"]])
                raise ValueError(
                    f"""Received unsupported role. 
                    Supported roles for the LLaMa Foundation Model: {supported}"""
                )
        prompt["prompt"] = user_content[len(user_content) - 1]
        history = [
            history_item
            for _, history_item in enumerate(zip(user_content[:-1], assistant_content))
        ]

        prompt["history"] = history

        return {**prompt, **model_kwargs}

    def _format_response_payload(
        self, output: bytes, stop_sequences: Optional[List[str]]
    ) -> str:
        """Formats response"""
        try:
            text = json.loads(output)["response"]
            if stop_sequences:
                text = enforce_stop_tokens(text, stop_sequences)
            return text
        except Exception as e:
            if isinstance(e, json.decoder.JSONDecodeError):
                return output.decode("utf-8")
            raise e

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        params = self._invocation_params(stop, **kwargs)

        request_payload = self.format_request_payload(messages, **params)
        response_payload = self._call_eas(request_payload)
        generated_text = self._format_response_payload(response_payload, params["stop"])

        if run_manager:
            run_manager.on_llm_new_token(generated_text)

        return generated_text

    def _call_eas(self, query_body: dict) -> Any:
        """Generate text from the eas service."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"{self.eas_service_token}",
        }

        # make request
        response = requests.post(
            self.eas_service_url, headers=headers, json=query_body, timeout=self.timeout
        )

        if response.status_code != 200:
            raise Exception(
                f"Request failed with status code {response.status_code}"
                f" and message {response.text}"
            )

        return response.text

    def _call_eas_stream(self, query_body: dict) -> Any:
        """Generate text from the eas service."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"{self.eas_service_token}",
        }

        # make request
        response = requests.post(
            self.eas_service_url, headers=headers, json=query_body, timeout=self.timeout
        )

        if response.status_code != 200:
            raise Exception(
                f"Request failed with status code {response.status_code}"
                f" and message {response.text}"
            )

        return response

    def _convert_chunk_to_message_message(
        self,
        chunk: str,
    ) -> AIMessageChunk:
        data = json.loads(chunk.encode("utf-8"))
        return AIMessageChunk(content=data.get("response", ""))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        params = self._invocation_params(stop, **kwargs)

        request_payload = self.format_request_payload(messages, **params)
        request_payload["use_stream_chat"] = True

        response = self._call_eas_stream(request_payload)
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"\0"
        ):
            if chunk:
                content = self._convert_chunk_to_message_message(chunk)

                # identify stop sequence in generated text, if any
                stop_seq_found: Optional[str] = None
                for stop_seq in params["stop"]:
                    if stop_seq in content.content:
                        stop_seq_found = stop_seq

                # identify text to yield
                text: Optional[str] = None
                if stop_seq_found:
                    content.content = content.content[
                        : content.content.index(stop_seq_found)
                    ]

                # yield text, if any
                if text:
                    if run_manager:
                        await run_manager.on_llm_new_token(cast(str, content.content))
                    yield ChatGenerationChunk(message=content)

                # break if stop sequence found
                if stop_seq_found:
                    break

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stream if stream is not None else self.streaming:
            generation: Optional[ChatGenerationChunk] = None
            async for chunk in self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                generation = chunk
            assert generation is not None
            return ChatResult(generations=[generation])

        func = partial(
            self._generate, messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

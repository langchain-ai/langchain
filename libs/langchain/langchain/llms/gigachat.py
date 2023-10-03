# flake8: noqa: I001
"""
GigaChatModel for GigaChat.
"""
from __future__ import annotations

import logging
from functools import cached_property
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import gigachat
from gigachat.models import ChatCompletion
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import BaseLLM
from langchain.schema.output import Generation, GenerationChunk, LLMResult

logger = logging.getLogger(__name__)


class GigaChat(BaseLLM):
    """`GigaChat` large language models API.

    To use, you should pass login and password to access GigaChat API or use token.

    Example:
        .. code-block:: python

            from langchain.llms import GigaChat
            giga = GigaChat(credentials=..., verify_ssl_certs=False)
    """

    base_url: Optional[str] = None
    """ Адрес относительно которого выполняются запросы """
    auth_url: Optional[str] = None
    credentials: Optional[str] = None
    """ Авторизационные данные """
    scope: Optional[str] = None

    access_token: Optional[str] = None
    model: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None

    timeout: Optional[float] = None
    verify_ssl_certs: Optional[bool] = None
    """ Check certificates for all requests """

    use_auth: Optional[bool] = None

    profanity: bool = True
    streaming: bool = False
    """ Whether to stream the results or not. """
    temperature: float = 0
    max_tokens: int = 0
    """ Maximum number of tokens to generate """

    @property
    def _llm_type(self) -> str:
        return "giga-chat-model"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "credentials": "GIGACHAT_CREDENTIALS",
            "access_token": "GIGACHAT_ACCESS_TOKEN",
            "password": "GIGACHAT_PASSWORD",
        }

    @property
    def lc_serializable(self) -> bool:
        return True

    @cached_property
    def _client(self) -> gigachat.GigaChat:
        return gigachat.GigaChat(
            base_url=self.base_url,
            auth_url=self.auth_url,
            credentials=self.credentials,
            scope=self.scope,
            access_token=self.access_token,
            model=self.model,
            user=self.user,
            password=self.password,
            timeout=self.timeout,
            verify_ssl_certs=self.verify_ssl_certs,
            use_auth=self.use_auth,
        )

    def _build_payload(self, messages: List[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": m} for m in messages],
            "profanity_check": self.profanity,
        }
        if self.temperature > 0:
            payload["temperature"] = self.temperature
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens

        if self.verbose:
            logger.warning("Giga request: %s", payload)

        return payload

    def _create_llm_result(self, response: ChatCompletion) -> LLMResult:
        generations = []
        for res in response.choices:
            finish_reason = res.finish_reason
            gen = Generation(
                text=res.message.content,
                generation_info={"finish_reason": finish_reason},
            )
            generations.append([gen])
            if finish_reason != "stop":
                logger.warning(
                    "Giga generation stopped with reason: %s",
                    finish_reason,
                )
            if self.verbose:
                logger.warning("Giga response: %s", res.message.content)
        token_usage = response.usage
        llm_output = {"token_usage": token_usage, "model_name": response.model}
        return LLMResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> LLMResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            generation: Optional[GenerationChunk] = None
            stream_iter = self._stream(
                prompts[0], stop=stop, run_manager=run_manager, **kwargs
            )
            for chunk in stream_iter:
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return LLMResult(generations=[[generation]])

        payload = self._build_payload(prompts)
        response = self._client.chat(payload)

        return self._create_llm_result(response)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> LLMResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            generation: Optional[GenerationChunk] = None
            stream_iter = self._astream(
                prompts[0], stop=stop, run_manager=run_manager, **kwargs
            )
            async for chunk in stream_iter:
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return LLMResult(generations=[[generation]])

        payload = self._build_payload(prompts)
        response = await self._client.achat(payload)

        return self._create_llm_result(response)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        payload = self._build_payload([prompt])

        for chunk in self._client.stream(payload):
            if chunk.choices:
                content = chunk.choices[0].delta.content
                yield GenerationChunk(text=content)
                if run_manager:
                    run_manager.on_llm_new_token(content)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        payload = self._build_payload([prompt])

        async for chunk in self._client.astream(payload):
            if chunk.choices:
                content = chunk.choices[0].delta.content
                yield GenerationChunk(text=content)
                if run_manager:
                    await run_manager.on_llm_new_token(content)

    def get_num_tokens(self, text: str) -> int:
        return round(len(text) / 4.6)

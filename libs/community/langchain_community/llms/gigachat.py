from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.load.serializable import Serializable
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.utils import pre_init
from langchain_core.utils.pydantic import get_fields

if TYPE_CHECKING:
    import gigachat
    import gigachat.models as gm

logger = logging.getLogger(__name__)


class _BaseGigaChat(Serializable):
    base_url: Optional[str] = None
    """ Base API URL """
    auth_url: Optional[str] = None
    """ Auth URL """
    credentials: Optional[str] = None
    """ Auth Token """
    scope: Optional[str] = None
    """ Permission scope for access token """

    access_token: Optional[str] = None
    """ Access token for GigaChat """

    model: Optional[str] = None
    """Model name to use."""
    user: Optional[str] = None
    """ Username for authenticate """
    password: Optional[str] = None
    """ Password for authenticate """

    timeout: Optional[float] = None
    """ Timeout for request """
    verify_ssl_certs: Optional[bool] = None
    """ Check certificates for all requests """

    ca_bundle_file: Optional[str] = None
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    key_file_password: Optional[str] = None
    # Support for connection to GigaChat through SSL certificates

    profanity: bool = True
    """ DEPRECATED: Check for profanity """
    profanity_check: Optional[bool] = None
    """ Check for profanity """
    streaming: bool = False
    """ Whether to stream the results or not. """
    temperature: Optional[float] = None
    """ What sampling temperature to use. """
    max_tokens: Optional[int] = None
    """ Maximum number of tokens to generate """
    use_api_for_tokens: bool = False
    """ Use GigaChat API for tokens count """
    verbose: bool = False
    """ Verbose logging """
    top_p: Optional[float] = None
    """ top_p value to use for nucleus sampling. Must be between 0.0 and 1.0 """
    repetition_penalty: Optional[float] = None
    """ The penalty applied to repeated tokens """
    update_interval: Optional[float] = None
    """ Minimum interval in seconds that elapses between sending tokens """

    @property
    def _llm_type(self) -> str:
        return "giga-chat-model"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "credentials": "GIGACHAT_CREDENTIALS",
            "access_token": "GIGACHAT_ACCESS_TOKEN",
            "password": "GIGACHAT_PASSWORD",
            "key_file_password": "GIGACHAT_KEY_FILE_PASSWORD",
        }

    @property
    def lc_serializable(self) -> bool:
        return True

    @cached_property
    def _client(self) -> gigachat.GigaChat:
        """Returns GigaChat API client"""
        import gigachat

        return gigachat.GigaChat(
            base_url=self.base_url,
            auth_url=self.auth_url,
            credentials=self.credentials,
            scope=self.scope,
            access_token=self.access_token,
            model=self.model,
            profanity_check=self.profanity_check,
            user=self.user,
            password=self.password,
            timeout=self.timeout,
            verify_ssl_certs=self.verify_ssl_certs,
            ca_bundle_file=self.ca_bundle_file,
            cert_file=self.cert_file,
            key_file=self.key_file,
            key_file_password=self.key_file_password,
            verbose=self.verbose,
        )

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate authenticate data in environment and python package is installed."""
        try:
            import gigachat  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import gigachat python package. "
                "Please install it with `pip install gigachat`."
            )
        fields = set(get_fields(cls).keys())
        diff = set(values.keys()) - fields
        if diff:
            logger.warning(f"Extra fields {diff} in GigaChat class")
        if "profanity" in fields and values.get("profanity") is False:
            logger.warning(
                "'profanity' field is deprecated. Use 'profanity_check' instead."
            )
            if values.get("profanity_check") is None:
                values["profanity_check"] = values.get("profanity")
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "temperature": self.temperature,
            "model": self.model,
            "profanity": self.profanity_check,
            "streaming": self.streaming,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }

    def tokens_count(
        self, input_: List[str], model: Optional[str] = None
    ) -> List[gm.TokensCount]:
        """Get tokens of string list"""
        return self._client.tokens_count(input_, model)

    async def atokens_count(
        self, input_: List[str], model: Optional[str] = None
    ) -> List[gm.TokensCount]:
        """Get tokens of strings list (async)"""
        return await self._client.atokens_count(input_, model)

    def get_models(self) -> gm.Models:
        """Get available models of Gigachat"""
        return self._client.get_models()

    async def aget_models(self) -> gm.Models:
        """Get available models of Gigachat (async)"""
        return await self._client.aget_models()

    def get_model(self, model: str) -> gm.Model:
        """Get info about model"""
        return self._client.get_model(model)

    async def aget_model(self, model: str) -> gm.Model:
        """Get info about model (async)"""
        return await self._client.aget_model(model)

    def get_num_tokens(self, text: str) -> int:
        """Count approximate number of tokens"""
        if self.use_api_for_tokens:
            return self.tokens_count([text])[0].tokens  # type: ignore
        else:
            return round(len(text) / 4.6)


class GigaChat(_BaseGigaChat, BaseLLM):
    """`GigaChat` large language models API.

    To use, you should pass login and password to access GigaChat API or use token.

    Example:
        .. code-block:: python

            from langchain_community.llms import GigaChat
            giga = GigaChat(credentials=..., scope=..., verify_ssl_certs=False)
    """

    payload_role: str = "user"

    def _build_payload(self, messages: List[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": [{"role": self.payload_role, "content": m} for m in messages],
        }
        if self.model:
            payload["model"] = self.model
        if self.profanity_check is not None:
            payload["profanity_check"] = self.profanity_check
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = self.repetition_penalty
        if self.update_interval is not None:
            payload["update_interval"] = self.update_interval

        if self.verbose:
            logger.info("Giga request: %s", payload)

        return payload

    def _create_llm_result(self, response: Any) -> LLMResult:
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
                logger.info("Giga response: %s", res.message.content)

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

    class Config:
        extra = "allow"

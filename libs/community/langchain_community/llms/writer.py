from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.utils import get_from_dict_or_env
from pydantic import ConfigDict, Field, SecretStr, model_validator


class Writer(LLM):
    """Writer large language models.

    To use, you should have the ``writer-sdk`` Python package installed, and the
    environment variable ``WRITER_API_KEY`` set with your API key.

    Example:
        .. code-block:: python

            from langchain_community.llms import Writer as WriterLLM
            from writerai import Writer, AsyncWriter

            client = Writer()
            async_client = AsyncWriter()

            chat = WriterLLM(
                client=client,
                async_client=async_client
            )
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:

    api_key: Optional[SecretStr] = Field(default=None)
    """Writer API key."""

    model_name: str = Field(default="palmyra-x-003-instruct", alias="model")
    """Model name to use."""

    max_tokens: Optional[int] = None
    """The maximum number of tokens that the model can generate in the response."""

    temperature: Optional[float] = 0.7
    """Controls the randomness of the model's outputs. Higher values lead to more 
    random outputs, while lower values make the model more deterministic."""

    top_p: Optional[float] = None
    """Used to control the nucleus sampling, where only the most probable tokens
     with a cumulative probability of top_p are considered for sampling, providing 
     a way to fine-tune the randomness of predictions."""

    stop: Optional[List[str]] = None
    """Specifies stopping conditions for the model's output generation. This can
     be an array of strings or a single string that the model will look for as a 
     signal to stop generating further tokens."""

    best_of: Optional[int] = None
    """Specifies the number of completions to generate and return the best one.
     Useful for generating multiple outputs and choosing the best based on some
      criteria."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Writer API."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
            "best_of": self.best_of,
            **self.model_kwargs,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model_name,
            **self._default_params,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "writer"

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validates that api key is passed and creates Writer clients."""
        try:
            from writerai import AsyncClient, Client
        except ImportError as e:
            raise ImportError(
                "Could not import writerai python package. "
                "Please install it with `pip install writerai`."
            ) from e

        if not values.get("client"):
            values.update(
                {
                    "client": Client(
                        api_key=get_from_dict_or_env(
                            values, "api_key", "WRITER_API_KEY"
                        )
                    )
                }
            )

        if not values.get("async_client"):
            values.update(
                {
                    "async_client": AsyncClient(
                        api_key=get_from_dict_or_env(
                            values, "api_key", "WRITER_API_KEY"
                        )
                    )
                }
            )

        if not (
            type(values.get("client")) is Client
            and type(values.get("async_client")) is AsyncClient
        ):
            raise ValueError(
                "'client' attribute must be with type 'Client' and "
                "'async_client' must be with type 'AsyncClient' from 'writerai' package"
            )

        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        params = {**self._identifying_params, **kwargs}
        if stop is not None:
            params.update({"stop": stop})
        text = self.client.completions.create(prompt=prompt, **params).choices[0].text
        return text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        params = {**self._identifying_params, **kwargs}
        if stop is not None:
            params.update({"stop": stop})
        response = await self.async_client.completions.create(prompt=prompt, **params)
        text = response.choices[0].text
        return text

    def _stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = {**self._identifying_params, **kwargs, "stream": True}
        if stop is not None:
            params.update({"stop": stop})
        response = self.client.completions.create(prompt=prompt, **params)
        for chunk in response:
            if run_manager:
                run_manager.on_llm_new_token(chunk.value)
            yield GenerationChunk(text=chunk.value)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        params = {**self._identifying_params, **kwargs, "stream": True}
        if stop is not None:
            params.update({"stop": stop})
        response = await self.async_client.completions.create(prompt=prompt, **params)
        async for chunk in response:
            if run_manager:
                await run_manager.on_llm_new_token(chunk.value)
            yield GenerationChunk(text=chunk.value)

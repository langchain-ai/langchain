"""Ollama large language models."""

from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Union,
)

import ollama
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Extra
from ollama import Options


class OllamaLLM(BaseLLM):
    """OllamaLLM large language models.

    Example:
        .. code-block:: python

            from langchain_ollama import OllamaLLM

            model = OllamaLLM()
            model.invoke("Come up with 10 names for a song about parrots")
    """

    model: str = "llama2"
    """Model name to use."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ollama-llm"

    def _create_generate_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[Union[Mapping[str, Any], str]]:
        options_data: Dict = {
            k: v
            for k, v in kwargs.items()
            if k not in ["keep_alive", "format"] and k in Options.__annotations__
        }
        options_data["stop"] = stop
        yield from ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            options=Options(**options_data),
            keep_alive=kwargs.get("keep_alive", None),
            format=kwargs.get("format", None),
        )

    def _stream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk = None
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = GenerationChunk(
                    text=stream_resp["response"] if "response" in stream_resp else "",
                    generation_info=dict(stream_resp)
                    if stream_resp.get("done") is True
                    else None,
                )
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            final_chunk = self._stream_with_aggregation(
                prompt,
                stop=stop,
                run_manager=run_manager,
                verbose=self.verbose,
                **kwargs,
            )
            generations.append([final_chunk])
        return LLMResult(generations=generations)  # type: ignore[arg-type]

    # TODO: Implement if OllamaLLM supports async generation. Otherwise
    # delete method.
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        raise NotImplementedError

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = GenerationChunk(
                    text=stream_resp["message"]["content"]
                    if "message" in stream_resp
                    else "",
                    generation_info=dict(stream_resp)
                    if stream_resp.get("done") is True
                    else None,
                )
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    # TODO: Implement if OllamaLLM supports async streaming. Otherwise delete
    # method.
    # async def _astream(
    #     self,
    #     prompt: str,
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> AsyncIterator[GenerationChunk]:
    #     raise NotImplementedError

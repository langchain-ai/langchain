from __future__ import annotations

import logging
from typing import Any, Iterator, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.llms.llamacpp import _LlamaCppCommon
from langchain.schema import ChatResult
from langchain.schema.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain.schema.output import ChatGeneration, ChatGenerationChunk, GenerationChunk
from langchain_core.schema.messages import messages_to_dict

logger = logging.getLogger(__name__)


class ChatLlamaCpp(BaseChatModel, _LlamaCppCommon):
    """llama.cpp chat model.

    To use, you should have the llama-cpp-python library installed, and provide the
    path to the Llama model as a named parameter to the constructor.
    Check out: https://github.com/abetlen/llama-cpp-python

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatLlamaCpp
            llm = ChatLlamaCpp(model_path="./path/to/model.gguf")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "llamacpp-chat"

    def _stream_with_aggregation(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk: Optional[GenerationChunk] = None
        for chunk in self._stream(messages, stop, **kwargs):
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    verbose=verbose,
                )
        if final_chunk is None:
            raise ValueError("No data received from llamacpp stream.")

        return final_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to LlamaCpp's generation endpoint.

        Args:
            messages: The list of base messages to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            Chat generations from the model

        Example:
            .. code-block:: python

                response = llamacpp([
                    HumanMessage(content="Tell me about the history of AI")
                ])
        """
        final_chunk = self._stream_with_aggregation(
            messages, stop=stop, run_manager=run_manager, verbose=self.verbose, **kwargs
        )
        chat_generation = ChatGeneration(
            message=AIMessage(content=final_chunk.text),
            generation_info=final_chunk.generation_info,
        )
        return ChatResult(generations=[chat_generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = {**self._get_parameters(stop), **kwargs}
        result = self.client.create_chat_completion(
            messages=messages_to_dict(messages), stream=True, **params
        )
        for part in result:
            logprobs = part["choices"][0].get("logprobs", None)
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=part["choices"][0]["text"]),
                generation_info={"logprobs": logprobs},
            )
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    token=chunk.text, verbose=self.verbose, log_probs=logprobs
                )

"""LiteLLM Router as LangChain Model."""
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
)

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_community.chat_models.litellm import (
    ChatLiteLLM,
    _convert_dict_to_message,
    _convert_delta_to_message_chunk,
)
from langchain_core.language_models.chat_models import (
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)

token_usage_key_name = "token_usage"
model_extra_key_name = "model_extra"


def get_llm_output(usage: Any, **params: Any) -> dict:
    """Get llm output from usage and params."""
    llm_output = {token_usage_key_name: usage}
    # copy over metadata (metadata came from router completion call)
    metadata = params["metadata"]
    for key in metadata:
        if key not in llm_output:
            # if token usage in metadata, prefer metadata's copy of it
            llm_output[key] = metadata[key]
    return llm_output


def _get_router(
    llm: Any,
):
    return llm.metadata["router"]


def _get_model_for_completion(router):
    # use first model name (aka: model group),
    # since we can only pass one to the router completion functions
    return router.model_list[0]["model_name"]


class ChatLiteLLMRouter(ChatLiteLLM):
    """LiteLLM Router as LangChain Model."""

    # use metadata here, so we can store arbitrary types
    # from https://github.com/langchain-ai/langchain/issues/12304#issuecomment-1826394746
    metadata: Dict[str, Any] = None

    @property
    def _llm_type(self) -> str:
        return "LiteLLMRouter"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        router = _get_router(self)
        self.model = _get_model_for_completion(router)
        params["model"] = self.model
        # add metadata so router can fill it below
        params.setdefault("metadata", {})
        response = router.completion(
            messages=message_dicts,
            **params,
        )
        return self._create_chat_result(response, **params)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        default_chunk_class = AIMessageChunk
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        router = _get_router(self)
        self.model = _get_model_for_completion(router)
        params["model"] = self.model
        # add metadata so router can fill it below
        params.setdefault("metadata", {})
        for chunk in router.completion(messages=message_dicts, **params):
            if len(chunk["choices"]) == 0:
                continue
            delta = chunk["choices"][0]["delta"]
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            yield ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, **params)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        default_chunk_class = AIMessageChunk
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        router = _get_router(self)
        self.model = _get_model_for_completion(router)
        params["model"] = self.model
        # add metadata so router can fill it below
        params.setdefault("metadata", {})
        async for chunk in await router.acompletion(messages=message_dicts, **params):
            if len(chunk["choices"]) == 0:
                continue
            delta = chunk["choices"][0]["delta"]
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            yield ChatGenerationChunk(message=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.content, **params)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        router = _get_router(self)
        self.model = _get_model_for_completion(router)
        params["model"] = self.model
        # add metadata so router can fill it below
        params.setdefault("metadata", {})
        response = await router.acompletion(
            messages=message_dicts,
            **params,
        )
        return self._create_chat_result(response, **params)

    # from
    # https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_models/openai.py
    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        combined: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            for output_key, output_value in output.items():
                if output_key == token_usage_key_name:
                    if output_key in combined:
                        # sum the usage values from different outputs
                        # to produce a set of totals
                        if hasattr(output_value, model_extra_key_name):
                            for usage_key, usage_value in output_value[model_extra_key_name].items():
                                if usage_key in overall_token_usage:
                                    overall_token_usage[usage_key] += usage_value
                                else:
                                    overall_token_usage[usage_key] = usage_value
                        combined[output_key] = overall_token_usage
                    else:
                        # set initial usage values
                        overall_token_usage = output_value
                        combined[output_key] = overall_token_usage
                elif output_key in combined:
                    # values for this key cannot be combined,
                    # so ignore subsequent values for it
                    pass
                else:
                    # set the initial value for this key
                    combined[output_key] = output_value
        return combined

    def _create_chat_result(self, response: Mapping[str, Any], **params: Any) -> ChatResult:
        from litellm.utils import Usage
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.get("finish_reason")),
            )
            generations.append(gen)
        token_usage = response.get("usage", Usage(prompt_tokens=0, total_tokens=0))
        llm_output = get_llm_output(token_usage, **params)
        return ChatResult(generations=generations, llm_output=llm_output)

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional

from requests.exceptions import HTTPError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import BaseLLM
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import Generation, LLMResult
from langchain.schema.output import GenerationChunk
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _stream_response_to_generation_chunk(
    stream_response: Dict[str, Any],
    length: int,
) -> GenerationChunk:
    """Convert a stream response to a generation chunk.

    As the low level API implement is different from openai and other llm.
    Stream response of Tongyi is not split into chunks, but all data generated before.
    For example, the answer 'Hi Pickle Rick! How can I assist you today?'
    Other llm will stream answer:
    'Hi Pickle',
    ' Rick!',
    ' How can I assist you today?'.

    Tongyi answer:
    'Hi Pickle',
    'Hi Pickle Rick!',
    'Hi Pickle Rick! How can I assist you today?'.

    As the GenerationChunk is implemented with chunks. Only return full_text[length:]
    for new chunk.
    """
    full_text = stream_response["output"]["text"]
    text = full_text[length:]
    finish_reason = stream_response["output"].get("finish_reason", None)

    return GenerationChunk(
        text=text,
        generation_info=dict(
            finish_reason=finish_reason,
        ),
    )


def _create_retry_decorator(
    llm: Tongyi, run_manager: Optional[CallbackManagerForLLMRun]
) -> Callable[[Any], Any]:
    def _before_sleep(retry_state: RetryCallState) -> None:
        if run_manager:
            run_manager.on_retry(retry_state)
        return None

    min_seconds = 1
    max_seconds = 4
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(HTTPError)),
        before_sleep=_before_sleep,
    )


def completion_with_retry(
    llm: Tongyi, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(**_kwargs: Any) -> Any:
        resp = llm.client.call(**_kwargs)
        if resp.status_code == 200:
            return resp
        elif resp.status_code in [400, 401]:
            raise ValueError(
                f"status_code: {resp.status_code} \n "
                f"code: {resp.code} \n message: {resp.message}"
            )
        else:
            raise HTTPError(
                f"HTTP error occurred: status_code: {resp.status_code} \n "
                f"code: {resp.code} \n message: {resp.message}"
            )

    return _completion_with_retry(**kwargs)


def stream_completion_with_retry(
    llm: Tongyi, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _stream_completion_with_retry(**_kwargs: Any) -> Any:
        return llm.client.call(**_kwargs)

    return _stream_completion_with_retry(**kwargs)


class Tongyi(BaseLLM):
    """Tongyi Qwen large language models.

    To use, you should have the ``dashscope`` python package installed,
    and set env ``DASHSCOPE_API_KEY`` with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.llms import Tongyi
            Tongyi = tongyi()
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"dashscope_api_key": "DASHSCOPE_API_KEY"}

    @property
    def lc_serializable(self) -> bool:
        return True

    client: Any  #: :meta private:
    model_name: str = Field(default="qwen-turbo", alias="model")

    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    top_p: float = 0.8
    """Total probability mass of tokens to consider at each step."""

    dashscope_api_key: Optional[str] = None
    """Dashscope api key provide by alicloud."""

    n: int = 1
    """How many completions to generate for each prompt."""

    streaming: bool = False
    """Whether to stream the results or not."""

    max_retries: int = 10
    """Maximum number of retries to make when generating."""

    prefix_messages: List = Field(default_factory=list)
    """Series of messages for Chat input."""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "tongyi"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        get_from_dict_or_env(values, "dashscope_api_key", "DASHSCOPE_API_KEY")
        try:
            import dashscope
        except ImportError:
            raise ImportError(
                "Could not import dashscope python package. "
                "Please install it with `pip install dashscope --upgrade`."
            )
        try:
            values["client"] = dashscope.Generation
        except AttributeError:
            raise ValueError(
                "`dashscope` has no `Generation` attribute, this is likely "
                "due to an old version of the dashscope package. Try upgrading it "
                "with `pip install --upgrade dashscope`."
            )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        normal_params = {
            "top_p": self.top_p,
        }

        return {**normal_params, **self.model_kwargs}

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        params: Dict[str, Any] = {
            **{"model": self.model_name},
            **self._default_params,
            **kwargs,
        }
        if self.streaming:
            if len(prompts) > 1:
                raise ValueError("Cannot stream results with multiple prompts.")
            params["stream"] = True

            # Mark current chunk total length
            length = 0
            for stream_resp in stream_completion_with_retry(
                self, prompt=prompts[0], **params
            ):
                full_text = stream_resp["output"]["text"]
                text = full_text[length:]
                generations.append(
                    [
                        Generation(
                            text=text,
                            generation_info=dict(
                                finish_reason=stream_resp["output"]["finish_reason"],
                            ),
                        )
                    ]
                )
                if run_manager:
                    run_manager.on_llm_new_token(text)
                length = len(stream_resp["output"]["text"])
        else:
            for prompt in prompts:
                completion = completion_with_retry(self, prompt=prompt, **params)
                generations.append(
                    [
                        Generation(
                            text=completion["output"]["text"],
                        )
                    ]
                )
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params: Dict[str, Any] = {
            **{"model": self.model_name},
            **self._default_params,
            **kwargs,
            "stream": True,
        }
        # Mark current chunk total length
        length = 0
        for stream_resp in stream_completion_with_retry(
            self, prompt=prompt, run_manager=run_manager, **params
        ):
            chunk = _stream_response_to_generation_chunk(stream_resp, length)
            length += len(chunk.text)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text)

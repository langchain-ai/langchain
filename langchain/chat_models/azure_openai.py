"""Azure OpenAI chat wrapper."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

from pydantic import root_validator

from langchain.chat_models.openai import (
    ChatOpenAI,
    acompletion_with_retry,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
)
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__file__)


def _create_chat_prompt(messages: List[BaseMessage]) -> str:
    """Create a prompt for Azure OpenAI using ChatML."""
    prompt = "\n".join([message.format_chatml() for message in messages])
    return prompt + "\n<|im_start|>assistant\n"


def _create_chat_result(response: Mapping[str, Any]) -> ChatResult:
    generations = []
    for res in response["choices"]:
        message = AIMessage(content=res["text"])
        gen = ChatGeneration(message=message)
        generations.append(gen)
    return ChatResult(generations=generations)


class AzureChatOpenAI(ChatOpenAI):
    """Wrapper around Azure OpenAI Chat large language models.

    To use, you should have the ``openai`` python package installed, and the
    following environment variables set:
    - ``OPENAI_API_TYPE``
    - ``OPENAI_API_KEY``
    - ``OPENAI_API_BASE``
    - ``OPENAI_API_VERSION``

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.chat_models import AzureChatOpenAI
            openai = AzureChatOpenAI(deployment_name="<your deployment name>")
    """

    deployment_name: str = ""
    stop: List[str] = ["<|im_end|>"]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values,
            "openai_api_key",
            "OPENAI_API_KEY",
        )
        openai_api_base = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
        )
        openai_api_version = get_from_dict_or_env(
            values,
            "openai_api_version",
            "OPENAI_API_VERSION",
        )
        openai_api_type = get_from_dict_or_env(
            values,
            "openai_api_type",
            "OPENAI_API_TYPE",
        )
        try:
            import openai

            openai.api_type = openai_api_type
            openai.api_base = openai_api_base
            openai.api_version = openai_api_version
            openai.api_key = openai_api_key
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        try:
            values["client"] = openai.Completion
        except AttributeError:
            raise ValueError(
                "`openai` has no `Completion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            **super()._default_params,
            "stop": self.stop,
        }

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        prompt, params = self._create_prompt(messages, stop)
        if self.streaming:
            inner_completion = ""
            params["stream"] = True
            for stream_resp in self.completion_with_retry(prompt=prompt, **params):
                token = stream_resp["choices"][0]["delta"].get("text", "")
                inner_completion += token
                self.callback_manager.on_llm_new_token(
                    token,
                    verbose=self.verbose,
                )
            message = AIMessage(content=inner_completion)
            return ChatResult(generations=[ChatGeneration(message=message)])
        response = self.completion_with_retry(prompt=prompt, **params)
        return _create_chat_result(response)

    def _create_prompt(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[str, Dict[str, Any]]:
        params: Dict[str, Any] = {
            **{"model": self.model_name, "engine": self.deployment_name},
            **self._default_params,
        }
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        prompt = _create_chat_prompt(messages)
        return prompt, params

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        prompt, params = self._create_prompt(messages, stop)
        if self.streaming:
            inner_completion = ""
            params["stream"] = True
            async for stream_resp in await acompletion_with_retry(
                self, prompt=prompt, **params
            ):
                token = stream_resp["choices"][0]["delta"].get("text", "")
                inner_completion += token
                if self.callback_manager.is_async:
                    await self.callback_manager.on_llm_new_token(
                        token,
                        verbose=self.verbose,
                    )
                else:
                    self.callback_manager.on_llm_new_token(
                        token,
                        verbose=self.verbose,
                    )
            message = AIMessage(content=inner_completion)
            return ChatResult(generations=[ChatGeneration(message=message)])
        else:
            response = await acompletion_with_retry(self, prompt=prompt, **params)
            return _create_chat_result(response)

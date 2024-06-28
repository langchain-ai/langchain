from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from langchain_community.llms.volcengine_maas import (
    VolcEngineMaasBase,
    VolcEngineMaasBaseV3,
)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {"role": "function", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    return message_dict


def convert_dict_to_message(_dict: Mapping[str, Any]) -> AIMessage:
    """Convert a dict to a message."""

    content = _dict.get("choice", {}).get("message", {}).get("content", "")
    return AIMessage(content=content)


def convert_dict_to_messageV3(response) -> AIMessage:
    """Convert a response object to a message."""
    if not hasattr(response, "choices") or not response.choices:
        raise ValueError("Invalid response format")

    content = response.choices[0].message.content if response.choices[0].message else ""
    return AIMessage(content=content)


class VolcEngineMaasChat(BaseChatModel, VolcEngineMaasBase):
    """Volc Engine Maas hosts a plethora of models.

    You can utilize these models through this class.

    To use, you should have the ``volcengine`` python package installed.
    and set access key and secret key by environment variable or direct pass those
    to this class.
    access key, secret key are required parameters which you could get help
    https://www.volcengine.com/docs/6291/65568

    In order to use them, it is necessary to install the 'volcengine' Python package.
    The access key and secret key must be set either via environment variables or
    passed directly to this class.
    access key and secret key are mandatory parameters for which assistance can be
    sought at https://www.volcengine.com/docs/6291/65568.

    The two methods are as follows:
    * Environment Variable
    Set the environment variables 'VOLC_ACCESSKEY' and 'VOLC_SECRETKEY' with your
    access key and secret key.

    * Pass Directly to Class
    Example:
        .. code-block:: python

            from langchain_community.llms import VolcEngineMaasLLM
            model = VolcEngineMaasChat(model="skylark-lite-public",
                                          volc_engine_maas_ak="your_ak",
                                          volc_engine_maas_sk="your_sk")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "volc-engine-maas-chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            **{"endpoint": self.endpoint, "model": self.model},
            **super()._identifying_params,
        }

    def _convert_prompt_msg_params(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        model_req = {
            "model": {
                "name": self.model,
            }
        }
        if self.model_version is not None:
            model_req["model"]["version"] = self.model_version
        return {
            **model_req,
            "messages": [_convert_message_to_dict(message) for message in messages],
            "parameters": {**self._default_params, **kwargs},
        }

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if stop is not None:
            kwargs["stop"] = stop
        params = self._convert_prompt_msg_params(messages, **kwargs)
        for res in self.client.stream_chat(params):
            if res:
                msg = convert_dict_to_message(res)
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=msg.content))
                if run_manager:
                    run_manager.on_llm_new_token(cast(str, msg.content), chunk=chunk)
                yield chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        completion = ""
        if self.streaming:
            for chunk in self._stream(messages, stop, run_manager, **kwargs):
                completion += chunk.text
        else:
            if stop is not None:
                kwargs["stop"] = stop
            params = self._convert_prompt_msg_params(messages, **kwargs)
            res = self.client.chat(params)
            msg = convert_dict_to_message(res)
            completion = cast(str, msg.content)

        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])


class VolcEngineMaasChatV3(BaseChatModel, VolcEngineMaasBaseV3):
    """Volc Engine Maas V3 hosts a plethora of models.

    You can utilize these models through this class.

    To use, you should have the ``volcengine-python-sdk`` python package installed.
    and set access key and secret key by environment variable or direct pass those
    to this class.
    access key, secret key are required parameters which you could get help
    https://www.volcengine.com/docs/82379/1263482

    In order to use them, it is necessary to install the 'volcengine-python-sdk'
    Python package. The access key and secret key must be set either via environment
    variables or passed directly to this class.
    access key and secret key are mandatory parameters for which assistance can be
    sought at https://www.volcengine.com/docs/82379/1263482.

    The two methods are as follows:
    * Environment Variable
    Set the environment variables 'VOLC_ACCESSKEY' and 'VOLC_SECRETKEY' with your
    access key and secret key.

    * Pass Directly to Class
    Example:
        .. code-block:: python

            from langchain_community.llms import VolcEngineMaasLLMV3
            model = VolcEngineMaasChatV3(model="ep-xxxxxxxxx-yyyy",
                                          volc_engine_maas_ak="your_ak",
                                          volc_engine_maas_sk="your_sk")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "volc-engine-maas-chat-v3"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            **super()._identifying_params,
        }

    def _convert_prompt_msg_params(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [_convert_message_to_dict(message) for message in messages],
            "stream": self.streaming,
        }

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if stop is not None:
            kwargs["stop"] = stop
        params = self._convert_prompt_msg_params(messages, **kwargs)
        for res in self.client.chat.completions.create(**params):
            if res:
                for choice in res.choices:
                    delta_content = choice.delta.content if choice.delta else ""
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=delta_content)
                    )
                    if run_manager:
                        run_manager.on_llm_new_token(
                            cast(str, delta_content), chunk=chunk
                        )
                    if stop and any(s in delta_content for s in stop):
                        # Trim the delta_content up to the stop word
                        delta_content = delta_content.split(
                            next(s for s in stop if s in delta_content)
                        )[0]
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=delta_content)
                        )
                        yield chunk
                        return
                    yield chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        completion = ""
        if self.streaming:
            for chunk in self._stream(messages, stop, run_manager, **kwargs):
                completion += chunk.message.content
                if stop and any(s in chunk.message.content for s in stop):
                    break
        else:
            if stop is not None:
                kwargs["stop"] = stop
            params = self._convert_prompt_msg_params(messages, **kwargs)
            res = self.client.chat.completions.create(**params)
            msg = convert_dict_to_messageV3(res)
            completion = cast(str, msg.content)
            if stop:
                for s in stop:
                    if s in completion:
                        completion = completion.split(s)[0]
                        break
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

from typing import Any, Dict, List, Mapping, Optional
from langchain_core.pydantic_v1 import root_validator
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from langchain.utils import get_from_dict_or_env
import logging
import time
import random

logger = logging.getLogger(__name__)


def predict(sensenova, messages, params):
    try:
        resp = sensenova.ChatCompletion.create(
            model=params.get('model_id', 'nova-ptc-xl-v1'),
            max_new_tokens=params.get('max_new_tokens', 1024),
            repetition_penalty=params.get('repetition_penalty', 1.05),
            stream=params.get('streaming', False),
            temperature=params.get('temperature', 0.1),
            top_p=params.get('top_p', 0.7),
            messages=messages
        )
    except Exception as e:
        logger.error('call sensenova api has exception: {}, retry...'.format(e))
        time.sleep(1 + random.random())
        return predict(sensenova, messages, params)
    return resp


async def predict_async(sensenova, messages, params):
    resp = await sensenova.ChatCompletion.acreate(
        model=params.get('model_id', 'nova-ptc-xl-v1'),
        max_new_tokens=params.get('max_new_tokens', 1024),
        repetition_penalty=params.get('repetition_penalty', 1.05),
        stream=params.get('streaming', False),
        temperature=params.get('temperature', 0.1),
        top_p=params.get('top_p', 0.7),
        messages=messages
    )
    return resp


def _convert_dict_to_message(_dict: dict) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict["content"])
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


class ChatSenseNova(BaseChatModel):
    max_tokens: int = 1024
    temperature: float = 0.1
    model_type: str = "SenseNova"
    model_id: Optional[str] = "nova-ptc-xl-v1"
    streaming: bool = False
    top_p: float = 0.7
    repetition_penalty: float = 1.05
    client: Any

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["sensenova_access_key_id"] = get_from_dict_or_env(
            values, "sensenova_access_key_id", "SENSENOVA_ACCESS_KEY_ID"
        )

        values["sensenova_secret_access_key"] = get_from_dict_or_env(
            values, "sensenova_secret_access_key", "SENSENOVA_SECRET_ACCESS_KEY"
        )

        model_id = get_from_dict_or_env(values, "model_id", "SENSENOVA_MODEL_ID")
        if model_id not in ["nova-ptc-xl-v1"]:
            raise ValueError('model_type must be in the range ["nova-ptc-xl-v1"]')

        values["model_id"] = model_id

        if values["temperature"] is not None and not 0 < values["temperature"] <= 2:
            raise ValueError("temperature must be in the range (0,2]")

        if values["temperature"] < 0.001:
            values["temperature"] = 0.001

        if values["max_tokens"] is not None and not 0 < values["max_tokens"] <= 2048:
            raise ValueError("max_token must be in the range [1,2048]")

        if values["top_p"] is not None and not 0 < values["top_p"] < 1:
            raise ValueError("top_p must be in the range (0,1)")

        if values["repetition_penalty"] is not None and not 0 < values["repetition_penalty"] <= 2:
            raise ValueError("repetition_penalty must be in the range (0,2]")

        try:
            import sensenova
        except ImportError:
            raise ImportError(
                "Could not import sensenova python package. "
                "Please install it with `pip install sensenova`."
            )

        sensenova.access_key_id = values["sensenova_access_key_id"]
        sensenova.secret_access_key = values["sensenova_secret_access_key"]
        values['client'] = sensenova

        return values

    def _llm_type(self) -> str:
        return self.model_type

    @property
    def _default_kwargs(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "streaming": self.streaming,
            "top_p": self.top_p,
        }

    def _create_message_dicts(
            self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        return [_convert_message_to_dict(m) for m in messages]

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = AIMessage(content=res["message"])
            gen = ChatGeneration(message=message)
            generations.append(gen)
        llm_output = {"token_usage": response["usage"], "model_name": self.model_id}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        message_dicts = self._create_message_dicts(messages, stop)
        response = predict(self.client, message_dicts, self._default_kwargs)
        if self.streaming:
            completion = ""
            usage = {}
            finish_reason = ""
            for part in response:
                choices = part['data']["choices"]
                for c_idx, c in enumerate(choices):
                    delta = c.get("delta")
                    if delta:
                        completion += delta
                        if run_manager:
                            run_manager.on_llm_new_token(
                                delta,
                            )
                    if c.get('finish_reason'):
                        finish_reason = c.get('finish_reason')
                usage = part['data']['usage']
            message = _convert_dict_to_message(
                {
                    "content": completion,
                    "role": "assistant",
                }
            )
            llm_output = {"token_usage": usage, "model_name": self.model_id, "finish_reason": finish_reason}
            return ChatResult(generations=[ChatGeneration(message=message)], llm_output=llm_output)

        response = response['data']
        return self._create_chat_result(response)

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        message_dicts = self._create_message_dicts(messages, stop)
        response = await predict_async(self.client, message_dicts, self._default_kwargs)
        if self.streaming:
            completion = ""
            usage = {}
            finish_reason = ""
            async for part in response:
                choices = part['data']["choices"]
                for c_idx, c in enumerate(choices):
                    delta = c.get("delta")
                    if delta:
                        completion += delta
                        if run_manager:
                            await run_manager.on_llm_new_token(
                                delta,
                            )
                    if c.get('finish_reason'):
                        finish_reason = c.get('finish_reason')
                usage = part['data']['usage']
            message = _convert_dict_to_message(
                {
                    "content": completion,
                    "role": "assistant",
                }
            )
            llm_output = {"token_usage": usage, "model_name": self.model_id, "finish_reason": finish_reason}
            return ChatResult(generations=[ChatGeneration(message=message)], llm_output=llm_output)

        response = response['data']
        return self._create_chat_result(response)

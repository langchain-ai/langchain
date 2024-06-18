import logging
import os
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

from ibm_watsonx_ai import Credentials  # type: ignore
from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
    convert_to_messages,
)
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

logger = logging.getLogger(__name__)


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("generated_text", ""))
    else:
        content = _dict.get("generated_text", "") or ""
        additional_kwargs: Dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        dict(make_invalid_tool_call(raw_tool_call, str(e)))
                    )
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            # If tool calls only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        try:
            tool_call_chunks = [
                {
                    "name": rtc["function"].get("name"),
                    "args": rtc["function"].get("arguments"),
                    "id": rtc.get("id"),
                    "index": rtc["index"],
                }
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass
    else:
        tool_call_chunks = []

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
        )
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)  # type: ignore


class ChatWatsonx(BaseChatModel):
    """
    IBM watsonx.ai large language chat models.

    To use, you should have ``langchain_ibm`` python package installed,
    and the environment variable ``WATSONX_APIKEY`` set with your API key, or pass
    it as a named parameter to the constructor.


    Example:
        .. code-block:: python

            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
            parameters = {
                GenTextParamsMetaNames.DECODING_METHOD: "sample",
                GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
                GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
                GenTextParamsMetaNames.TEMPERATURE: 0.5,
                GenTextParamsMetaNames.TOP_K: 50,
                GenTextParamsMetaNames.TOP_P: 1,
            }

            from langchain_ibm import ChatWatsonx
            watsonx_llm = ChatWatsonx(
                model_id="meta-llama/llama-3-70b-instruct",
                url="https://us-south.ml.cloud.ibm.com",
                apikey="*****",
                project_id="*****",
                params=parameters,
            )
    """

    model_id: str = ""
    """Type of model to use."""

    deployment_id: str = ""
    """Type of deployed model to use."""

    project_id: str = ""
    """ID of the Watson Studio project."""

    space_id: str = ""
    """ID of the Watson Studio space."""

    url: Optional[SecretStr] = None
    """Url to Watson Machine Learning or CPD instance"""

    apikey: Optional[SecretStr] = None
    """Apikey to Watson Machine Learning or CPD instance"""

    token: Optional[SecretStr] = None
    """Token to CPD instance"""

    password: Optional[SecretStr] = None
    """Password to CPD instance"""

    username: Optional[SecretStr] = None
    """Username to CPD instance"""

    instance_id: Optional[SecretStr] = None
    """Instance_id of CPD instance"""

    version: Optional[SecretStr] = None
    """Version of CPD instance"""

    params: Optional[dict] = None
    """Chat Model parameters to use during generate requests."""

    verify: Union[str, bool] = ""
    """User can pass as verify one of following:
        the path to a CA_BUNDLE file
        the path of directory with certificates of trusted CAs
        True - default path to truststore will be taken
        False - no verification will be made"""

    streaming: bool = False
    """ Whether to stream the results or not. """

    watsonx_model: ModelInference = Field(default=None, exclude=True)  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def _llm_type(self) -> str:
        return "watsonx-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example:
            {
                "url": "WATSONX_URL",
                "apikey": "WATSONX_APIKEY",
                "token": "WATSONX_TOKEN",
                "password": "WATSONX_PASSWORD",
                "username": "WATSONX_USERNAME",
                "instance_id": "WATSONX_INSTANCE_ID",
            }
        """
        return {
            "url": "WATSONX_URL",
            "apikey": "WATSONX_APIKEY",
            "token": "WATSONX_TOKEN",
            "password": "WATSONX_PASSWORD",
            "username": "WATSONX_USERNAME",
            "instance_id": "WATSONX_INSTANCE_ID",
        }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that credentials and python package exists in environment."""
        values["url"] = convert_to_secret_str(
            get_from_dict_or_env(values, "url", "WATSONX_URL")
        )
        if "cloud.ibm.com" in values.get("url", "").get_secret_value():
            values["apikey"] = convert_to_secret_str(
                get_from_dict_or_env(values, "apikey", "WATSONX_APIKEY")
            )
        else:
            if (
                not values["token"]
                and "WATSONX_TOKEN" not in os.environ
                and not values["password"]
                and "WATSONX_PASSWORD" not in os.environ
                and not values["apikey"]
                and "WATSONX_APIKEY" not in os.environ
            ):
                raise ValueError(
                    "Did not find 'token', 'password' or 'apikey',"
                    " please add an environment variable"
                    " `WATSONX_TOKEN`, 'WATSONX_PASSWORD' or 'WATSONX_APIKEY' "
                    "which contains it,"
                    " or pass 'token', 'password' or 'apikey'"
                    " as a named parameter."
                )
            elif values["token"] or "WATSONX_TOKEN" in os.environ:
                values["token"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "token", "WATSONX_TOKEN")
                )
            elif values["password"] or "WATSONX_PASSWORD" in os.environ:
                values["password"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "password", "WATSONX_PASSWORD")
                )
                values["username"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "username", "WATSONX_USERNAME")
                )
            elif values["apikey"] or "WATSONX_APIKEY" in os.environ:
                values["apikey"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "apikey", "WATSONX_APIKEY")
                )
                values["username"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "username", "WATSONX_USERNAME")
                )
            if not values["instance_id"] or "WATSONX_INSTANCE_ID" not in os.environ:
                values["instance_id"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "instance_id", "WATSONX_INSTANCE_ID")
                )
        credentials = Credentials(
            url=values["url"].get_secret_value() if values["url"] else None,
            api_key=values["apikey"].get_secret_value() if values["apikey"] else None,
            token=values["token"].get_secret_value() if values["token"] else None,
            password=values["password"].get_secret_value()
            if values["password"]
            else None,
            username=values["username"].get_secret_value()
            if values["username"]
            else None,
            instance_id=values["instance_id"].get_secret_value()
            if values["instance_id"]
            else None,
            version=values["version"].get_secret_value() if values["version"] else None,
            verify=values["verify"],
        )

        watsonx_chat = ModelInference(
            model_id=values["model_id"],
            deployment_id=values["deployment_id"],
            credentials=credentials,
            params=values["params"],
            project_id=values["project_id"],
            space_id=values["space_id"],
        )
        values["watsonx_model"] = watsonx_chat

        return values

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
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        chat_prompt = self._create_chat_prompt(message_dicts)

        response = self.watsonx_model.generate(
            prompt=chat_prompt, params=params, **kwargs
        )
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        chat_prompt = self._create_chat_prompt(message_dicts)

        for chunk in self.watsonx_model.generate_text_stream(
            prompt=chat_prompt, raw_response=True, params=params, **kwargs
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["results"]) == 0:
                continue
            choice = chunk["results"][0]

            chunk = AIMessageChunk(
                content=choice["generated_text"],
            )
            generation_info = {}
            if finish_reason := choice.get("stop_reason"):
                generation_info["finish_reason"] = finish_reason
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info or None
            )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk, logprobs=logprobs)

            yield chunk

    def _create_chat_prompt(self, messages: List[Dict[str, Any]]) -> str:
        prompt = ""

        if self.model_id in ["ibm/granite-13b-chat-v1", "ibm/granite-13b-chat-v2"]:
            for message in messages:
                if message["role"] == "system":
                    prompt += "<|system|>\n" + message["content"] + "\n\n"
                elif message["role"] == "assistant":
                    prompt += "<|assistant|>\n" + message["content"] + "\n\n"
                elif message["role"] == "function":
                    prompt += "<|function|>\n" + message["content"] + "\n\n"
                elif message["role"] == "tool":
                    prompt += "<|tool|>\n" + message["content"] + "\n\n"
                else:
                    prompt += "<|user|>:\n" + message["content"] + "\n\n"

            prompt += "<|assistant|>\n"

        elif self.model_id in [
            "meta-llama/llama-2-13b-chat",
            "meta-llama/llama-2-70b-chat",
        ]:
            for message in messages:
                if message["role"] == "system":
                    prompt += "[INST] <<SYS>>\n" + message["content"] + "<</SYS>>\n\n"
                elif message["role"] == "assistant":
                    prompt += message["content"] + "\n[INST]\n\n"
                else:
                    prompt += message["content"] + "\n[/INST]\n"

        else:
            prompt = ChatPromptValue(messages=convert_to_messages(messages)).to_string()

        return prompt

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = {**self.params} if self.params else {}
        if stop is not None:
            if params and "stop_sequences" in params:
                raise ValueError(
                    "`stop_sequences` found in both the input and default params."
                )
            params = (params or {}) | {"stop_sequences": stop}
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Union[dict]) -> ChatResult:
        generations = []
        sum_of_total_generated_tokens = 0
        sum_of_total_input_tokens = 0

        if response.get("error"):
            raise ValueError(response.get("error"))

        for res in response["results"]:
            message = _convert_dict_to_message(res)
            generation_info = dict(finish_reason=res.get("stop_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            if "generated_token_count" in res:
                sum_of_total_generated_tokens += res["generated_token_count"]
            if "input_token_count" in res:
                sum_of_total_input_tokens += res["input_token_count"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = {
            "generated_token_count": sum_of_total_generated_tokens,
            "input_token_count": sum_of_total_input_tokens,
        }
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_id,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

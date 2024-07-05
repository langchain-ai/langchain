import json
from typing import Optional, Dict, List, Tuple, Any, Mapping, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    SystemMessage,
    HumanMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env, get_pydantic_field_names
from langchain_community.utilities.requests import Requests
import io


class ChatStraico(BaseChatModel):
    model: Optional[str] = Field(default="openai/gpt-3.5-turbo-0125")
    """Model name to use."""
    straico_api_key: Optional[str] = Field(None, alias="api_key")
    """Automatically inferred from env var `STRAICO_API_KEY` if not provided."""
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to OpenAI completion API. Can be float, httpx.Timeout or 
        None."""
    max_retries: int = Field(default=2)
    """Maximum number of retries to make when generating."""
    # max_tokens: Optional[int] = None
    # """Maximum number of tokens to generate."""

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["straico_api_key"] = get_from_dict_or_env(
            values, "straico_api_key", "STRAICO_API_KEY"
        )
        return values

    def _generate(self, messages, stop=None, run_manager=None) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params}
        response = self._completion_with_retry(messages=message_dicts, params=params)
        print("RESPONSE", response)
        message = AIMessage(
            content=response["data"]["completion"]["choices"][0]["message"]["content"]
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> ChatResult:
        # Simulate asynchronous response generation
        responses = [f"Async response to: {msg.content}" for msg in messages]
        chat_generations = [
            ChatGeneration(message=AIMessage(content=response))
            for response in responses
        ]
        return ChatResult(generations=chat_generations)

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling PerplexityChat API."""
        return {
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            # "max_tokens": self.max_tokens,
        }

    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        else:
            raise TypeError(f"Got unknown type {message}")
        return message_dict

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = dict(self._invocation_params)
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    @property
    def _invocation_params(self) -> Mapping[str, Any]:
        """Get the parameters used to invoke the model."""
        straico_creds: Dict[str, Any] = {
            "api_key": self.straico_api_key,
            "url": "https://api.straico.com/v0/prompt/completion",
            "model": self.model,
        }
        return {**straico_creds, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "straicochat"

    def _completion_with_retry(
        self, messages: List[Dict[str, Any]], params: Dict[str, Any]
    ) -> Any:
        retries = params["max_retries"]
        auth_header = {"Authorization": f"Bearer {params['api_key']}"}
        for attempt in range(retries):
            try:
                request = Requests(headers=auth_header)
                # Create a payload dictionary
                payloadStruct = {"model": params["model"], "message": str(messages)}
                response = request.post(url=params["url"], data=payloadStruct)
                self._handle_status(response.status_code, response.text)
                response_data = json.loads(response._content)
                return response_data
            except Exception as e:
                if attempt == retries - 1:
                    raise e

    def _handle_status(self, status_code: int, text: str) -> None:
        if status_code >= 500:
            raise Exception(f"Straico Server: Error {status_code}")
        elif status_code >= 400:
            raise ValueError(f"Straico received an invalid payload: {text}")
        elif status_code != 201:  # straico returns 201 for success
            raise Exception(
                f"Straico returned an unexpected response with status {status_code}: {text}"
            )

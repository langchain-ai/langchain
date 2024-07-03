from typing import Optional, Dict, List, Tuple, Any, Mapping
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


class ChatStraico(BaseChatModel):
    model: str = "google/gemini-pro"

    straico_api_key: Optional[str] = Field(None, alias="api_key")

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["straico_api_key"] = get_from_dict_or_env(
            values, "straico_api_key", "STRAICO_API_KEY"
        )
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        try:
            values["client"] = openai.OpenAI(
                api_key=values["straico_api_key"],
                base_url="https://api.straico.com/v0/",
            )
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        return values

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.client.chat.completions.create(
            model=params["model"], messages=message_dicts
        )
        message = AIMessage(
            content=response.data["completion"]["choices"][0]["message"]["content"]
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
            # "request_timeout": self.request_timeout,
            # "max_tokens": self.max_tokens,
            # "stream": self.streaming,
            # **self.model_kwargs,
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
            "api_base": "https://api.straico.com/v0/prompt/completion",
            "model": self.model,
        }
        return {**straico_creds, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "straicochat"

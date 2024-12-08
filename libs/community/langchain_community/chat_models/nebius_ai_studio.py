"""Nebius AI Studio chat wrapper. Relies heavily on ChatOpenAI."""

from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, SecretStr

from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.utils.openai import is_openai_v1

DEFAULT_API_BASE = "https://api.studio.nebius.ai/v1"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"


class ChatNebiusAIStudio(ChatOpenAI):
    """Nebius AI Studio large language models.

    To use, you should have the ``openai`` python package installed and the
    environment variable ``NEBIUS_API_KEY`` set with your API key.
    Alternatively, you can use the nebius_api_key keyword argument.

    Any parameters that are valid to be passed to the `openai.create` call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import NebiusAIStudioChat
            chat = NebiusAIStudioChat(model_name="meta-llama/Meta-Llama-3.1-70B-Instruct")
    """

    nebius_api_base: str = Field(default=DEFAULT_API_BASE)
    nebius_api_key: SecretStr = Field(default=None, alias="api_key")
    model_name: str = Field(default=DEFAULT_MODEL, alias="model")

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "nebius-ai-studio-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"nebius_api_token": "NEBIUS_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["nebius_api_base"] = get_from_dict_or_env(
            values,
            "nebius_api_base",
            "NEBIUS_API_BASE",
            default=DEFAULT_API_BASE,
        )
        values["nebius_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "nebius_api_key", "NEBIUS_API_KEY")
        )
        values["model_name"] = get_from_dict_or_env(
            values,
            "model_name",
            "MODEL_NAME",
            default=DEFAULT_MODEL,
        )

        try:
            import openai

            if is_openai_v1():
                client_params = {
                    "api_key": values["nebius_api_key"].get_secret_value(),
                    "base_url": values["nebius_api_base"],
                }
                if not values.get("client"):
                    values["client"] = openai.OpenAI(**client_params).chat.completions
                if not values.get("async_client"):
                    values["async_client"] = openai.AsyncOpenAI(
                        **client_params
                    ).chat.completions
            else:
                values["openai_api_base"] = values["nebius_api_base"]
                values["openai_api_key"] = values["nebius_api_key"].get_secret_value()
                values["client"] = openai.ChatCompletion  # type: ignore[attr-defined]
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        return values

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Imitating bind_tool method from langchain_openai.ChatOpenAI"""

        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "any", "required"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                # 'any' is not natively supported by OpenAI API.
                # We support 'any' since other models use this instead of 'required'.
                if tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                tool_names = [
                    formatted_tool["function"]["name"]
                    for formatted_tool in formatted_tools
                ]
                if not any(
                    tool_name == tool_choice["function"]["name"]
                    for tool_name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

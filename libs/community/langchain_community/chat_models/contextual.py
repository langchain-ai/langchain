"""Contextual AI Generate Wrapper"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import Field, SecretStr, model_validator


class ChatContextual(BaseChatModel):
    """Contextual AI chat model.

    Please make sure that you have `contextual-client` Python package installed.

    You will need to provide your api key in the init param or set the environment
    variable CONTEXTUAL_AI_API_KEY.

    If you are using a custom `base_url` for Contextual AI, you will need to include
    it in the init param as well.

    Currently, only model `v1` is supported."""

    client: Any = Field(defualt=None, exclude=True)
    """Contextual AI Client"""

    api_key: Optional[SecretStr] = Field(default=None)
    """Contextual AI API key."""

    base_url: Optional[str] = Field(default=None)
    """Base URL of Contextual AI Application."""

    model_name: str = Field(default="v1", alias="model")
    """The name of the model. Defaults to `v1`."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            from contextual import ContextualAI
        except ImportError:
            raise ImportError(
                "Could not import contextual python package."
                "Please install via the command `pip install contextual-client`."
            )
        else:
            values["api_key"] = convert_to_secret_str(
                get_from_dict_or_env(values, "api_key", "CONTEXTUAL_AI_API_KEY")
            )
            base_url = values.get("base_url", None)
            values["client"] = ContextualAI(
                api_key=values["api_key"].get_secret_value(),
                base_url=base_url,
            )
        return values

    @property
    def _llm_type(self) -> str:
        """Used to uniquely identify the type of the model. Used for logging."""
        return "contextual-clm"

    def _convert_langchain_to_contextual(self, message: BaseMessage) -> dict:
        """Converts LangChain message to ContextualAI Message."""
        role = None

        if isinstance(message, ChatMessage):
            role = message.role
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unexpected message type: {type(message)}")

        contextual_message = {
            "role": role,
            "content": message.content,
        }

        return contextual_message

    def _get_contextual_params(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> Tuple[list[Any], List[str], Optional[str], bool]:
        """Prepares input and parameters for Contextual AI."""
        contextual_messages: List[Any] = [
            self._convert_langchain_to_contextual(message) for message in messages
        ]
        knowledge: List[str] = kwargs.get("knowledge", [""])
        system_prompt: Optional[str] = kwargs.get("system_message", None)
        avoid_commentary: bool = kwargs.get("avoid_commentary", False)

        return contextual_messages, knowledge, system_prompt, avoid_commentary

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        messages, knowledge, system_prompt, avoid_commentary = (
            self._get_contextual_params(
                messages,
                **kwargs,
            )
        )
        raw_message = self.client.generate.create(
            messages=messages,
            knowledge=knowledge,
            model=self.model_name,
            system_prompt=system_prompt,
            extra_body={
                "avoid_commentary": avoid_commentary,
            },
        )
        message = AIMessage(
            content=raw_message.to_dict().get("response"),
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

from typing import Any, Dict, List, Optional, Union

from aiohttp import ClientSession
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_community.utilities.requests import Requests


def _format_dappier_messages(
    messages: List[BaseMessage],
) -> List[Dict[str, Union[str, List[Union[str, Dict[Any, Any]]]]]]:
    formatted_messages = []

    for message in messages:
        if message.type == "human":
            formatted_messages.append({"role": "user", "content": message.content})
        elif message.type == "system":
            formatted_messages.append({"role": "system", "content": message.content})

    return formatted_messages


class ChatDappierAI(BaseChatModel):
    """`Dappier` chat large language models.

    `Dappier` is a platform enabling access to diverse, real-time data models.
    Enhance your AI applications with Dappier's pre-trained, LLM-ready data models
    and ensure accurate, current responses with reduced inaccuracies.

    To use one of our Dappier AI Data Models, you will need an API key.
    Please visit Dappier Platform (https://platform.dappier.com/) to log in
    and create an API key in your profile.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatDappierAI
            from langchain_core.messages import HumanMessage

            # Initialize `ChatDappierAI` with the desired configuration
            chat = ChatDappierAI(
                dappier_endpoint="https://api.dappier.com/app/datamodel/dm_01hpsxyfm2fwdt2zet9cg6fdxt",
                dappier_api_key="<YOUR_KEY>")

            # Create a list of messages to interact with the model
            messages = [HumanMessage(content="hello")]

            # Invoke the model with the provided messages
            chat.invoke(messages)


    you can find more details here : https://docs.dappier.com/introduction"""

    dappier_endpoint: str = "https://api.dappier.com/app/datamodelconversation"

    dappier_model: str = "dm_01hpsxyfm2fwdt2zet9cg6fdxt"

    dappier_api_key: Optional[SecretStr] = Field(None, description="Dappier API Token")

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        values["dappier_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "dappier_api_key", "DAPPIER_API_KEY")
        )
        return values

    @staticmethod
    def get_user_agent() -> str:
        from langchain_community import __version__

        return f"langchain/{__version__}"

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "dappier-realtimesearch-chat"

    @property
    def _api_key(self) -> str:
        if self.dappier_api_key:
            return self.dappier_api_key.get_secret_value()
        return ""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        url = f"{self.dappier_endpoint}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": self.get_user_agent(),
        }
        user_query = _format_dappier_messages(messages=messages)
        payload: Dict[str, Any] = {
            "model": self.dappier_model,
            "conversation": user_query,
        }

        request = Requests(headers=headers)
        response = request.post(url=url, data=payload)
        response.raise_for_status()

        data = response.json()

        message_response = data["message"]

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=message_response))]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        url = f"{self.dappier_endpoint}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": self.get_user_agent(),
        }
        user_query = _format_dappier_messages(messages=messages)
        payload: Dict[str, Any] = {
            "model": self.dappier_model,
            "conversation": user_query,
        }

        async with ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                message_response = data["message"]

                return ChatResult(
                    generations=[
                        ChatGeneration(message=AIMessage(content=message_response))
                    ]
                )

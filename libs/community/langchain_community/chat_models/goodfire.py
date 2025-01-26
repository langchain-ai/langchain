from typing import Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import Field, SecretStr, model_validator


def format_for_goodfire(messages: List[BaseMessage]) -> List[dict]:
    """
    Format messages for Goodfire by setting "role" based on the message type.
    """
    output = []
    for message in messages:
        if isinstance(message, HumanMessage):
            output.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            output.append({"role": "assistant", "content": message.content})
        elif isinstance(message, SystemMessage):
            output.append({"role": "system", "content": message.content})
        else:
            raise ValueError(f"Unknown message type: {type(message)}")
    return output


def format_for_langchain(message: dict) -> BaseMessage:
    """
    Format a Goodfire message for Langchain. This assumes that the message is an
    assistant message (AIMessage).
    """
    assert message["role"] == "assistant", (
        f"Expected role 'assistant', got {message['role']}"
    )
    return AIMessage(content=message["content"])


class ChatGoodfire(BaseChatModel):
    """Goodfire chat model."""

    goodfire_api_key: SecretStr = Field(default=SecretStr(""))
    sync_client: Any = Field(default=None)
    async_client: Any = Field(default=None)
    variant: Any  # Changed type hint since we can't import goodfire at module level

    @property
    def _llm_type(self) -> str:
        return "goodfire"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"goodfire_api_key": "GOODFIRE_API_KEY"}

    def __init__(
        self,
        model: str,  # Changed from SUPPORTED_MODELS since we can't import it
        goodfire_api_key: Optional[str] = None,
        variant: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize the Goodfire chat model.

        Args:
            model: The model to use, must be one of the supported models.
            goodfire_api_key: The API key to use. If None, will look for
                GOODFIRE_API_KEY env var.
            variant: Optional variant to use. If not provided, will be created
                from the model parameter.
        """
        try:
            import goodfire
        except ImportError as e:
            raise ImportError(
                "Could not import goodfire python package. "
                "Please install it with `pip install goodfire`."
            ) from e

        # Create variant first
        variant_instance = variant or goodfire.Variant(model)

        # Include variant in kwargs for parent initialization
        kwargs["variant"] = variant_instance

        # Initialize parent class
        super().__init__(**kwargs)

        # Initialize API key and clients if provided
        if goodfire_api_key:
            self.goodfire_api_key = SecretStr(goodfire_api_key)
            self.sync_client = goodfire.Client(api_key=goodfire_api_key)
            self.async_client = goodfire.AsyncClient(api_key=goodfire_api_key)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        try:
            import goodfire
        except ImportError as e:
            raise ImportError(
                "Could not import goodfire python package. "
                "Please install it with `pip install goodfire`."
            ) from e

        values["goodfire_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "goodfire_api_key",
                "GOODFIRE_API_KEY",
            )
        )

        # Initialize clients with the validated API key
        api_key = values["goodfire_api_key"].get_secret_value()
        values["sync_client"] = goodfire.Client(api_key=api_key)
        values["async_client"] = goodfire.AsyncClient(api_key=api_key)

        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response from Goodfire.
        """

        # If a model is provided, use it instead of the default variant
        if "model" in kwargs:
            model = kwargs.pop("model")
        else:
            model = self.variant

        goodfire_response = self.sync_client.chat.completions.create(
            messages=format_for_goodfire(messages),
            model=model,
            **kwargs,
        )

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=format_for_langchain(goodfire_response.choices[0].message)
                )
            ]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response from Goodfire.
        """

        # If a model is provided, use it instead of the default variant
        if "model" in kwargs:
            model = kwargs.pop("model")
        else:
            model = self.variant

        goodfire_response = await self.async_client.chat.completions.create(
            messages=format_for_goodfire(messages),
            model=model,
            **kwargs,
        )

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=format_for_langchain(goodfire_response.choices[0].message)
                )
            ]
        )

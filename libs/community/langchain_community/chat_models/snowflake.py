import json
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
    pre_init,
)
from langchain_core.utils.utils import _build_model_kwargs
from pydantic import Field, SecretStr, model_validator

SUPPORTED_ROLES: List[str] = [
    "system",
    "user",
    "assistant",
]


class ChatSnowflakeCortexError(Exception):
    """Error with Snowpark client."""


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {
        "content": message.content,
    }

    # populate role and additional message data
    if isinstance(message, ChatMessage) and message.role in SUPPORTED_ROLES:
        message_dict["role"] = message.role
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _truncate_at_stop_tokens(
    text: str,
    stop: Optional[List[str]],
) -> str:
    """Truncates text at the earliest stop token found."""
    if stop is None:
        return text

    for stop_token in stop:
        stop_token_idx = text.find(stop_token)
        if stop_token_idx != -1:
            text = text[:stop_token_idx]
    return text


class ChatSnowflakeCortex(BaseChatModel):
    """Snowflake Cortex based Chat model

    To use you must have the ``snowflake-snowpark-python`` Python package installed and
    either:

        1. environment variables set with your snowflake credentials or
        2. directly passed in as kwargs to the ChatSnowflakeCortex constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatSnowflakeCortex
            chat = ChatSnowflakeCortex()
    """

    _sp_session: Any = None
    """Snowpark session object."""

    model: str = "snowflake-arctic"
    """Snowflake cortex hosted LLM model name, defaulted to `snowflake-arctic`.
        Refer to docs for more options."""

    cortex_function: str = "complete"
    """Cortex function to use, defaulted to `complete`.
        Refer to docs for more options."""

    temperature: float = 0.7
    """Model temperature. Value should be >= 0 and <= 1.0"""

    max_tokens: Optional[int] = None
    """The maximum number of output tokens in the response."""

    top_p: Optional[float] = None
    """top_p adjusts the number of choices for each predicted tokens based on
        cumulative probabilities. Value should be ranging between 0.0 and 1.0. 
    """

    snowflake_username: Optional[str] = Field(default=None, alias="username")
    """Automatically inferred from env var `SNOWFLAKE_USERNAME` if not provided."""
    snowflake_password: Optional[SecretStr] = Field(default=None, alias="password")
    """Automatically inferred from env var `SNOWFLAKE_PASSWORD` if not provided."""
    snowflake_account: Optional[str] = Field(default=None, alias="account")
    """Automatically inferred from env var `SNOWFLAKE_ACCOUNT` if not provided."""
    snowflake_database: Optional[str] = Field(default=None, alias="database")
    """Automatically inferred from env var `SNOWFLAKE_DATABASE` if not provided."""
    snowflake_schema: Optional[str] = Field(default=None, alias="schema")
    """Automatically inferred from env var `SNOWFLAKE_SCHEMA` if not provided."""
    snowflake_warehouse: Optional[str] = Field(default=None, alias="warehouse")
    """Automatically inferred from env var `SNOWFLAKE_WAREHOUSE` if not provided."""
    snowflake_role: Optional[str] = Field(default=None, alias="role")
    """Automatically inferred from env var `SNOWFLAKE_ROLE` if not provided."""

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        values = _build_model_kwargs(values, all_required_field_names)
        return values

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            from snowflake.snowpark import Session
        except ImportError:
            raise ImportError(
                "`snowflake-snowpark-python` package not found, please install it with "
                "`pip install snowflake-snowpark-python`"
            )

        values["snowflake_username"] = get_from_dict_or_env(
            values, "snowflake_username", "SNOWFLAKE_USERNAME"
        )
        values["snowflake_password"] = convert_to_secret_str(
            get_from_dict_or_env(values, "snowflake_password", "SNOWFLAKE_PASSWORD")
        )
        values["snowflake_account"] = get_from_dict_or_env(
            values, "snowflake_account", "SNOWFLAKE_ACCOUNT"
        )
        values["snowflake_database"] = get_from_dict_or_env(
            values, "snowflake_database", "SNOWFLAKE_DATABASE"
        )
        values["snowflake_schema"] = get_from_dict_or_env(
            values, "snowflake_schema", "SNOWFLAKE_SCHEMA"
        )
        values["snowflake_warehouse"] = get_from_dict_or_env(
            values, "snowflake_warehouse", "SNOWFLAKE_WAREHOUSE"
        )
        values["snowflake_role"] = get_from_dict_or_env(
            values, "snowflake_role", "SNOWFLAKE_ROLE"
        )

        connection_params = {
            "account": values["snowflake_account"],
            "user": values["snowflake_username"],
            "password": values["snowflake_password"].get_secret_value(),
            "database": values["snowflake_database"],
            "schema": values["snowflake_schema"],
            "warehouse": values["snowflake_warehouse"],
            "role": values["snowflake_role"],
        }

        try:
            values["_sp_session"] = Session.builder.configs(connection_params).create()
        except Exception as e:
            raise ChatSnowflakeCortexError(f"Failed to create session: {e}")

        return values

    def __del__(self) -> None:
        if getattr(self, "_sp_session", None) is not None:
            self._sp_session.close()

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return f"snowflake-cortex-{self.model}"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        message_str = str(message_dicts)
        options = {"temperature": self.temperature}
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        options_str = str(options)
        sql_stmt = f"""
            select snowflake.cortex.{self.cortex_function}(
                '{self.model}'
                ,{message_str},{options_str}) as llm_response;"""

        try:
            l_rows = self._sp_session.sql(sql_stmt).collect()
        except Exception as e:
            raise ChatSnowflakeCortexError(
                f"Error while making request to Snowflake Cortex via Snowpark: {e}"
            )

        response = json.loads(l_rows[0]["LLM_RESPONSE"])
        ai_message_content = response["choices"][0]["messages"]

        content = _truncate_at_stop_tokens(ai_message_content, stop)
        message = AIMessage(
            content=content,
            response_metadata=response["usage"],
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

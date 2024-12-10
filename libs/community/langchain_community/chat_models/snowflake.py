import json
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
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

    # Populate role and additional message data
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

    To use the chat model, you must have the ``snowflake-snowpark-python`` Python
    package installed and either:

        1. environment variables set with your snowflake credentials or
        2. directly passed in as kwargs to the ChatSnowflakeCortex constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatSnowflakeCortex
            chat = ChatSnowflakeCortex()
    """

    # test_tools: Dict[str, Any] = Field(default_factory=dict)
    test_tools: Dict[str, Union[Dict[str, Any], Type, Callable, BaseTool]] = Field(
        default_factory=dict
    )

    session: Any = None
    """Snowpark session object."""

    model: str = "mistral-large"
    """Snowflake cortex hosted LLM model name, defaulted to `mistral-large`.
        Refer to docs for more options. Also note, not all models support 
        agentic workflows."""

    cortex_function: str = "complete"
    """Cortex function to use, defaulted to `complete`.
        Refer to docs for more options."""

    temperature: float = 0
    """Model temperature. Value should be >= 0 and <= 1.0"""

    max_tokens: Optional[int] = None
    """The maximum number of output tokens in the response."""

    top_p: Optional[float] = 0
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

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = "auto",
        **kwargs: Any,
    ) -> "ChatSnowflakeCortex":
        """Bind tool-like objects to this chat model, ensuring they conform to
        expected formats."""

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        # self.test_tools.update(formatted_tools)
        formatted_tools_dict = {
            tool["name"]: tool for tool in formatted_tools if "name" in tool
        }
        self.test_tools.update(formatted_tools_dict)

        return self

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        values = _build_model_kwargs(values, all_required_field_names)
        return values

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            from snowflake.snowpark import Session
        except ImportError:
            raise ImportError(
                """`snowflake-snowpark-python` package not found, please install:
                `pip install snowflake-snowpark-python`
                """
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
            "client_session_keep_alive": "True",
        }

        try:
            values["session"] = Session.builder.configs(connection_params).create()
        except Exception as e:
            raise ChatSnowflakeCortexError(f"Failed to create session: {e}")

        return values

    def __del__(self) -> None:
        if getattr(self, "session", None) is not None:
            self.session.close()

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

        # Check for tool invocation in the messages and prepare for tool use
        tool_output = None
        for message in messages:
            if (
                isinstance(message.content, dict)
                and isinstance(message, SystemMessage)
                and "invoke_tool" in message.content
            ):
                tool_info = json.loads(message.content.get("invoke_tool"))
                tool_name = tool_info.get("tool_name")
                if tool_name in self.test_tools:
                    tool_args = tool_info.get("args", {})
                    tool_output = self.test_tools[tool_name](**tool_args)
                    break

        # Prepare messages for SQL query
        if tool_output:
            message_dicts.append(
                {"tool_output": str(tool_output)}
            )  # Ensure tool_output is a string

        # JSON dump the message_dicts and options without additional escaping
        message_json = json.dumps(message_dicts)
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p if self.top_p is not None else 1.0,
            "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
        }
        options_json = json.dumps(options)  # JSON string of options

        # Form the SQL statement using JSON literals
        sql_stmt = f"""
            select snowflake.cortex.{self.cortex_function}(
                '{self.model}',
                parse_json('{message_json}'),
                parse_json('{options_json}')
            ) as llm_response;
        """

        try:
            # Use the Snowflake Cortex Complete function
            self.session.sql(
                f"USE WAREHOUSE {self.session.get_current_warehouse()};"
            ).collect()
            l_rows = self.session.sql(sql_stmt).collect()
        except Exception as e:
            raise ChatSnowflakeCortexError(
                f"Error while making request to Snowflake Cortex: {e}"
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

    def _stream_content(
        self, content: str, stop: Optional[List[str]]
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream the output of the model in chunks to return ChatGenerationChunk.
        """
        chunk_size = 50  # Define a reasonable chunk size for streaming
        truncated_content = _truncate_at_stop_tokens(content, stop)

        for i in range(0, len(truncated_content), chunk_size):
            chunk_content = truncated_content[i : i + chunk_size]

            # Create and yield a ChatGenerationChunk with partial content
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_content))

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model in chunks to return ChatGenerationChunk."""
        message_dicts = [_convert_message_to_dict(m) for m in messages]

        # Check for and potentially use a tool before streaming
        for message in messages:
            if (
                isinstance(message, str)
                and isinstance(message, SystemMessage)
                and "invoke_tool" in message.content
            ):
                tool_info = json.loads(message.content)
                tool_list = tool_info.get("invoke_tools", [])
                for tool in tool_list:
                    tool_name = tool.get("tool_name")
                    tool_args = tool.get("args", {})

                if tool_name in self.test_tools:
                    tool_args = tool_info.get("args", {})
                    tool_result = self.test_tools[tool_name](**tool_args)
                    additional_context = {"tool_output": tool_result}
                    message_dicts.append(
                        additional_context
                    )  # Append tool result to message dicts

        # JSON dump the message_dicts and options without additional escaping
        message_json = json.dumps(message_dicts)
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p if self.top_p is not None else 1.0,
            "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
            # "stream": True,
        }
        options_json = json.dumps(options)  # JSON string of options

        # Form the SQL statement using JSON literals
        sql_stmt = f"""
            select snowflake.cortex.{self.cortex_function}(
                '{self.model}',
                parse_json('{message_json}'),
                parse_json('{options_json}')
            ) as llm_stream_response;
        """

        try:
            # Use the Snowflake Cortex Complete function
            self.session.sql(
                f"USE WAREHOUSE {self.session.get_current_warehouse()};"
            ).collect()
            result = self.session.sql(sql_stmt).collect()

            # Iterate over the generator to yield streaming responses
            for row in result:
                response = json.loads(row["LLM_STREAM_RESPONSE"])
                ai_message_content = response["choices"][0]["messages"]

                # Stream response content in chunks
                for chunk in self._stream_content(ai_message_content, stop):
                    yield chunk

        except Exception as e:
            raise ChatSnowflakeCortexError(
                f"Error while making request to Snowflake Cortex stream: {e}"
            )

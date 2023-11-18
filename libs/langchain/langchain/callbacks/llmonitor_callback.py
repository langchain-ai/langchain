import importlib.metadata
import logging
import os
import traceback
import warnings
from contextvars import ContextVar
from typing import Any, Dict, List, Union, cast
from uuid import UUID

import requests
from packaging.version import parse

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://app.llmonitor.com"

user_ctx = ContextVar[Union[str, None]]("user_ctx", default=None)
user_props_ctx = ContextVar[Union[str, None]]("user_props_ctx", default=None)

PARAMS_TO_CAPTURE = [
    "temperature",
    "top_p",
    "top_k",
    "stop",
    "presence_penalty",
    "frequence_penalty",
    "seed",
    "function_call",
    "functions",
    "tools",
    "tool_choice",
    "response_format",
    "max_tokens",
    "logit_bias",
]


class UserContextManager:
    """Context manager for LLMonitor user context."""

    def __init__(self, user_id: str, user_props: Any = None) -> None:
        user_ctx.set(user_id)
        user_props_ctx.set(user_props)

    def __enter__(self) -> Any:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> Any:
        user_ctx.set(None)
        user_props_ctx.set(None)


def identify(user_id: str, user_props: Any = None) -> UserContextManager:
    """Builds an LLMonitor UserContextManager

    Parameters:
        - `user_id`: The user id.
        - `user_props`: The user properties.

    Returns:
        A context manager that sets the user context.
    """
    return UserContextManager(user_id, user_props)


def _serialize(obj: Any) -> Union[Dict[str, Any], List[Any], Any]:
    if hasattr(obj, "to_json"):
        return obj.to_json()

    if isinstance(obj, dict):
        return {key: _serialize(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_serialize(element) for element in obj]

    return obj


def _parse_input(raw_input: Any) -> Any:
    if not raw_input:
        return None

    # if it's an array of 1, just parse the first element
    if isinstance(raw_input, list) and len(raw_input) == 1:
        return _parse_input(raw_input[0])

    if not isinstance(raw_input, dict):
        return _serialize(raw_input)

    input_value = raw_input.get("input")
    inputs_value = raw_input.get("inputs")
    question_value = raw_input.get("question")
    query_value = raw_input.get("query")

    if input_value:
        return input_value
    if inputs_value:
        return inputs_value
    if question_value:
        return question_value
    if query_value:
        return query_value

    return _serialize(raw_input)


def _parse_output(raw_output: dict) -> Any:
    if not raw_output:
        return None

    if not isinstance(raw_output, dict):
        return _serialize(raw_output)

    text_value = raw_output.get("text")
    output_value = raw_output.get("output")
    output_text_value = raw_output.get("output_text")
    answer_value = raw_output.get("answer")
    result_value = raw_output.get("result")

    if text_value:
        return text_value
    if answer_value:
        return answer_value
    if output_value:
        return output_value
    if output_text_value:
        return output_text_value
    if result_value:
        return result_value

    return _serialize(raw_output)


def _parse_lc_role(
    role: str,
) -> str:
    if role == "human":
        return "user"
    else:
        return role


def _get_user_id(metadata: Any) -> Any:
    if user_ctx.get() is not None:
        return user_ctx.get()

    metadata = metadata or {}
    user_id = metadata.get("user_id")
    if user_id is None:
        user_id = metadata.get("userId")  # legacy, to delete in the future
    return user_id


def _get_user_props(metadata: Any) -> Any:
    if user_props_ctx.get() is not None:
        return user_props_ctx.get()

    metadata = metadata or {}
    return metadata.get("user_props", None)


def _parse_lc_message(message: BaseMessage) -> Dict[str, Any]:
    keys = ["function_call", "tool_calls", "tool_call_id", "name"]
    parsed = {"text": message.content, "role": _parse_lc_role(message.type)}
    parsed.update(
        {
            key: cast(Any, message.additional_kwargs.get(key))
            for key in keys
            if message.additional_kwargs.get(key) is not None
        }
    )
    return parsed


def _parse_lc_messages(messages: Union[List[BaseMessage], Any]) -> List[Dict[str, Any]]:
    return [_parse_lc_message(message) for message in messages]


class LLMonitorCallbackHandler(BaseCallbackHandler):
    """Callback Handler for LLMonitor`.

    #### Parameters:
        - `app_id`: The app id of the app you want to report to. Defaults to
        `None`, which means that `LLMONITOR_APP_ID` will be used.
        - `api_url`: The url of the LLMonitor API. Defaults to `None`,
        which means that either `LLMONITOR_API_URL` environment variable
        or `https://app.llmonitor.com` will be used.

    #### Raises:
        - `ValueError`: if `app_id` is not provided either as an
        argument or as an environment variable.
        - `ConnectionError`: if the connection to the API fails.


    #### Example:
    ```python
    from langchain.llms import OpenAI
    from langchain.callbacks import LLMonitorCallbackHandler

    llmonitor_callback = LLMonitorCallbackHandler()
    llm = OpenAI(callbacks=[llmonitor_callback],
                 metadata={"userId": "user-123"})
    llm.predict("Hello, how are you?")
    ```
    """

    __api_url: str
    __app_id: str
    __verbose: bool
    __llmonitor_version: str
    __has_valid_config: bool

    def __init__(
        self,
        app_id: Union[str, None] = None,
        api_url: Union[str, None] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        self.__has_valid_config = True

        try:
            import llmonitor

            self.__llmonitor_version = importlib.metadata.version("llmonitor")
            self.__track_event = llmonitor.track_event

        except ImportError:
            logger.warning(
                """[LLMonitor] To use the LLMonitor callback handler you need to 
                have the `llmonitor` Python package installed. Please install it 
                with `pip install llmonitor`"""
            )
            self.__has_valid_config = False
            return

        if parse(self.__llmonitor_version) < parse("0.0.32"):
            logger.warning(
                f"""[LLMonitor] The installed `llmonitor` version is 
                {self.__llmonitor_version} 
                but `LLMonitorCallbackHandler` requires at least version 0.0.32 
                upgrade `llmonitor` with `pip install --upgrade llmonitor`"""
            )
            self.__has_valid_config = False

        self.__has_valid_config = True

        self.__api_url = api_url or os.getenv("LLMONITOR_API_URL") or DEFAULT_API_URL
        self.__verbose = verbose or bool(os.getenv("LLMONITOR_VERBOSE"))

        _app_id = app_id or os.getenv("LLMONITOR_APP_ID")
        if _app_id is None:
            logger.warning(
                """[LLMonitor] app_id must be provided either as an argument or 
                as an environment variable"""
            )
            self.__has_valid_config = False
        else:
            self.__app_id = _app_id

        if self.__has_valid_config is False:
            return None

        try:
            res = requests.get(f"{self.__api_url}/api/app/{self.__app_id}")
            if not res.ok:
                raise ConnectionError()
        except Exception:
            logger.warning(
                f"""[LLMonitor] Could not connect to the LLMonitor API at 
                {self.__api_url}"""
            )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> None:
        if self.__has_valid_config is False:
            return
        try:
            user_id = _get_user_id(metadata)
            user_props = _get_user_props(metadata)

            params = kwargs.get("invocation_params", {})
            params.update(
                serialized.get("kwargs", {})
            )  # Sometimes, for example with ChatAnthropic, `invocation_params` is empty

            name = (
                params.get("model")
                or params.get("model_name")
                or params.get("model_id")
            )

            if not name and "anthropic" in params.get("_type"):
                name = "claude-2"

            extra = {
                param: params.get(param)
                for param in PARAMS_TO_CAPTURE
                if params.get(param) is not None
            }

            input = _parse_input(prompts)

            self.__track_event(
                "llm",
                "start",
                user_id=user_id,
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                name=name,
                input=input,
                tags=tags,
                extra=extra,
                metadata=metadata,
                user_props=user_props,
                app_id=self.__app_id,
            )
        except Exception as e:
            warnings.warn(f"[LLMonitor] An error occurred in on_llm_start: {e}")

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        if self.__has_valid_config is False:
            return

        try:
            user_id = _get_user_id(metadata)
            user_props = _get_user_props(metadata)

            params = kwargs.get("invocation_params", {})
            params.update(
                serialized.get("kwargs", {})
            )  # Sometimes, for example with ChatAnthropic, `invocation_params` is empty

            name = (
                params.get("model")
                or params.get("model_name")
                or params.get("model_id")
            )

            if not name and "anthropic" in params.get("_type"):
                name = "claude-2"

            extra = {
                param: params.get(param)
                for param in PARAMS_TO_CAPTURE
                if params.get(param) is not None
            }

            input = _parse_lc_messages(messages[0])

            self.__track_event(
                "llm",
                "start",
                user_id=user_id,
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                name=name,
                input=input,
                tags=tags,
                extra=extra,
                metadata=metadata,
                user_props=user_props,
                app_id=self.__app_id,
            )
        except Exception as e:
            logger.error(f"[LLMonitor] An error occurred in on_chat_model_start: {e}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> None:
        if self.__has_valid_config is False:
            return

        try:
            token_usage = (response.llm_output or {}).get("token_usage", {})

            parsed_output: Any = [
                _parse_lc_message(generation.message)
                if hasattr(generation, "message")
                else generation.text
                for generation in response.generations[0]
            ]

            # if it's an array of 1, just parse the first element
            if len(parsed_output) == 1:
                parsed_output = parsed_output[0]

            self.__track_event(
                "llm",
                "end",
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                output=parsed_output,
                token_usage={
                    "prompt": token_usage.get("prompt_tokens"),
                    "completion": token_usage.get("completion_tokens"),
                },
                app_id=self.__app_id,
            )
        except Exception as e:
            logger.error(f"[LLMonitor] An error occurred in on_llm_end: {e}")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> None:
        if self.__has_valid_config is False:
            return
        try:
            user_id = _get_user_id(metadata)
            user_props = _get_user_props(metadata)
            name = serialized.get("name")

            self.__track_event(
                "tool",
                "start",
                user_id=user_id,
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                name=name,
                input=input_str,
                tags=tags,
                metadata=metadata,
                user_props=user_props,
                app_id=self.__app_id,
            )
        except Exception as e:
            logger.error(f"[LLMonitor] An error occurred in on_tool_start: {e}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        **kwargs: Any,
    ) -> None:
        if self.__has_valid_config is False:
            return
        try:
            self.__track_event(
                "tool",
                "end",
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                output=output,
                app_id=self.__app_id,
            )
        except Exception as e:
            logger.error(f"[LLMonitor] An error occurred in on_tool_end: {e}")

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            name = serialized.get("id", [None, None, None, None])[3]
            type = "chain"
            metadata = metadata or {}

            agentName = metadata.get("agent_name")
            if agentName is None:
                agentName = metadata.get("agentName")

            if name == "AgentExecutor" or name == "PlanAndExecute":
                type = "agent"
            if agentName is not None:
                type = "agent"
                name = agentName
            if parent_run_id is not None:
                type = "chain"

            user_id = _get_user_id(metadata)
            user_props = _get_user_props(metadata)
            input = _parse_input(inputs)

            self.__track_event(
                type,
                "start",
                user_id=user_id,
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                name=name,
                input=input,
                tags=tags,
                metadata=metadata,
                user_props=user_props,
                app_id=self.__app_id,
            )
        except Exception as e:
            logger.error(f"[LLMonitor] An error occurred in on_chain_start: {e}")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            output = _parse_output(outputs)

            self.__track_event(
                "chain",
                "end",
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                output=output,
                app_id=self.__app_id,
            )
        except Exception as e:
            logger.error(f"[LLMonitor] An error occurred in on_chain_end: {e}")

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            name = action.tool
            input = _parse_input(action.tool_input)

            self.__track_event(
                "tool",
                "start",
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                name=name,
                input=input,
                app_id=self.__app_id,
            )
        except Exception as e:
            logger.error(f"[LLMonitor] An error occurred in on_agent_action: {e}")

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            output = _parse_output(finish.return_values)

            self.__track_event(
                "agent",
                "end",
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                output=output,
                app_id=self.__app_id,
            )
        except Exception as e:
            logger.error(f"[LLMonitor] An error occurred in on_agent_finish: {e}")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            self.__track_event(
                "chain",
                "error",
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                error={"message": str(error), "stack": traceback.format_exc()},
                app_id=self.__app_id,
            )
        except Exception as e:
            logger.error(f"[LLMonitor] An error occurred in on_chain_error: {e}")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            self.__track_event(
                "tool",
                "error",
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                error={"message": str(error), "stack": traceback.format_exc()},
                app_id=self.__app_id,
            )
        except Exception as e:
            logger.error(f"[LLMonitor] An error occurred in on_tool_error: {e}")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        if self.__has_valid_config is False:
            return
        try:
            self.__track_event(
                "llm",
                "error",
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                error={"message": str(error), "stack": traceback.format_exc()},
                app_id=self.__app_id,
            )
        except Exception as e:
            logger.error(f"[LLMonitor] An error occurred in on_llm_error: {e}")


__all__ = ["LLMonitorCallbackHandler", "identify"]

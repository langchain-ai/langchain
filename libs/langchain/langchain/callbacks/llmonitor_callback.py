import os
import traceback
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, List, Literal, Union
from uuid import UUID

import requests

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult

DEFAULT_API_URL = "https://app.llmonitor.com"

user_ctx = ContextVar[Union[str, None]]("user_ctx", default=None)
user_props_ctx = ContextVar[Union[str, None]]("user_props_ctx", default=None)


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
) -> Union[Literal["user", "ai", "system", "function"], None]:
    if role == "human":
        return "user"
    elif role == "ai":
        return "ai"
    elif role == "system":
        return "system"
    elif role == "function":
        return "function"
    else:
        return None


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
    return metadata.get("user_props")


def _parse_lc_message(message: BaseMessage) -> Dict[str, Any]:
    parsed = {"text": message.content, "role": _parse_lc_role(message.type)}

    function_call = (message.additional_kwargs or {}).get("function_call")

    if function_call is not None:
        parsed["functionCall"] = function_call

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

    def __init__(
        self,
        app_id: Union[str, None] = None,
        api_url: Union[str, None] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        self.__api_url = api_url or os.getenv("LLMONITOR_API_URL") or DEFAULT_API_URL

        self.__verbose = verbose or bool(os.getenv("LLMONITOR_VERBOSE"))

        _app_id = app_id or os.getenv("LLMONITOR_APP_ID")
        if _app_id is None:
            raise ValueError(
                """app_id must be provided either as an argument or as 
                an environment variable"""
            )
        self.__app_id = _app_id

        try:
            res = requests.get(f"{self.__api_url}/api/app/{self.__app_id}")
            if not res.ok:
                raise ConnectionError()
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to the LLMonitor API at {self.__api_url}"
            ) from e

    def __send_event(self, event: Dict[str, Any]) -> None:
        headers = {"Content-Type": "application/json"}

        event = {**event, "app": self.__app_id, "timestamp": str(datetime.utcnow())}

        if self.__verbose:
            print("llmonitor_callback", event)

        data = {"events": event}
        requests.post(headers=headers, url=f"{self.__api_url}/api/report", json=data)

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
        user_id = _get_user_id(metadata)
        user_props = _get_user_props(metadata)

        event = {
            "event": "start",
            "type": "llm",
            "userId": user_id,
            "runId": str(run_id),
            "parentRunId": str(parent_run_id) if parent_run_id else None,
            "input": _parse_input(prompts),
            "name": kwargs.get("invocation_params", {}).get("model_name"),
            "tags": tags,
            "metadata": metadata,
        }
        if user_props:
            event["userProps"] = user_props

        self.__send_event(event)

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
        user_id = _get_user_id(metadata)
        user_props = _get_user_props(metadata)

        event = {
            "event": "start",
            "type": "llm",
            "userId": user_id,
            "runId": str(run_id),
            "parentRunId": str(parent_run_id) if parent_run_id else None,
            "input": _parse_lc_messages(messages[0]),
            "name": kwargs.get("invocation_params", {}).get("model_name"),
            "tags": tags,
            "metadata": metadata,
        }
        if user_props:
            event["userProps"] = user_props

        self.__send_event(event)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> None:
        token_usage = (response.llm_output or {}).get("token_usage", {})

        parsed_output = [
            {
                "text": generation.text,
                "role": "ai",
                **(
                    {
                        "functionCall": generation.message.additional_kwargs[
                            "function_call"
                        ]
                    }
                    if hasattr(generation, "message")
                    and hasattr(generation.message, "additional_kwargs")
                    and "function_call" in generation.message.additional_kwargs
                    else {}
                ),
            }
            for generation in response.generations[0]
        ]

        event = {
            "event": "end",
            "type": "llm",
            "runId": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "output": parsed_output,
            "tokensUsage": {
                "prompt": token_usage.get("prompt_tokens"),
                "completion": token_usage.get("completion_tokens"),
            },
        }
        self.__send_event(event)

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
        user_id = _get_user_id(metadata)
        user_props = _get_user_props(metadata)

        event = {
            "event": "start",
            "type": "tool",
            "userId": user_id,
            "runId": str(run_id),
            "parentRunId": str(parent_run_id) if parent_run_id else None,
            "name": serialized.get("name"),
            "input": input_str,
            "tags": tags,
            "metadata": metadata,
        }
        if user_props:
            event["userProps"] = user_props

        self.__send_event(event)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        **kwargs: Any,
    ) -> None:
        event = {
            "event": "end",
            "type": "tool",
            "runId": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "output": output,
        }
        self.__send_event(event)

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
        name = serialized.get("id", [None, None, None, None])[3]
        type = "chain"
        metadata = metadata or {}

        agentName = metadata.get("agent_name")
        if agentName is None:
            agentName = metadata.get("agentName")

        if agentName is not None:
            type = "agent"
            name = agentName
        if name == "AgentExecutor" or name == "PlanAndExecute":
            type = "agent"

        if parent_run_id is not None:
            type = "chain"

        user_id = _get_user_id(metadata)
        user_props = _get_user_props(metadata)

        event = {
            "event": "start",
            "type": type,
            "userId": user_id,
            "runId": str(run_id),
            "parentRunId": str(parent_run_id) if parent_run_id else None,
            "input": _parse_input(inputs),
            "tags": tags,
            "metadata": metadata,
            "name": name,
        }
        if user_props:
            event["userProps"] = user_props

        self.__send_event(event)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        event = {
            "event": "end",
            "type": "chain",
            "runId": str(run_id),
            "output": _parse_output(outputs),
        }
        self.__send_event(event)

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        event = {
            "event": "start",
            "type": "tool",
            "runId": str(run_id),
            "parentRunId": str(parent_run_id) if parent_run_id else None,
            "name": action.tool,
            "input": _parse_input(action.tool_input),
        }
        self.__send_event(event)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        event = {
            "event": "end",
            "type": "agent",
            "runId": str(run_id),
            "parentRunId": str(parent_run_id) if parent_run_id else None,
            "output": _parse_output(finish.return_values),
        }
        self.__send_event(event)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        event = {
            "event": "error",
            "type": "chain",
            "runId": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "error": {"message": str(error), "stack": traceback.format_exc()},
        }
        self.__send_event(event)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        event = {
            "event": "error",
            "type": "tool",
            "runId": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "error": {"message": str(error), "stack": traceback.format_exc()},
        }
        self.__send_event(event)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        event = {
            "event": "error",
            "type": "llm",
            "runId": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "error": {"message": str(error), "stack": traceback.format_exc()},
        }
        self.__send_event(event)


__all__ = ["LLMonitorCallbackHandler", "identify"]

import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Literal, Union
from uuid import UUID

import requests

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult

DEFAULT_API_URL = "https://app.llmonitor.com"


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


def _serialize_lc_message(message: BaseMessage) -> Dict[str, Any]:
    return {"text": message.content, "role": _parse_lc_role(message.type)}


class LLMonitorCallbackHandler(BaseCallbackHandler):
    """Initializes the `LLMonitorCallbackHandler`.
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

    def __init__(
        self, app_id: Union[str, None] = None, api_url: Union[str, None] = None
    ) -> None:
        super().__init__()

        self.__api_url = api_url or os.getenv("LLMONITOR_API_URL") or DEFAULT_API_URL

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
        event = {
            "event": "start",
            "type": "llm",
            "userId": (metadata or {}).get("userId"),
            "runId": str(run_id),
            "parentRunId": str(parent_run_id) if parent_run_id else None,
            "input": prompts[0],
            "name": kwargs.get("invocation_params", {}).get("model_name"),
            "tags": tags,
            "metadata": metadata,
        }
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
        event = {
            "event": "start",
            "type": "llm",
            "userId": (metadata or {}).get("userId"),
            "runId": str(run_id),
            "parentRunId": str(parent_run_id) if parent_run_id else None,
            "input": [_serialize_lc_message(message[0]) for message in messages],
            "name": kwargs.get("invocation_params", {}).get("model_name"),
            "tags": tags,
            "metadata": metadata,
        }
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

        event = {
            "event": "end",
            "type": "llm",
            "runId": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "output": {"text": response.generations[0][0].text, "role": "ai"},
            "tokensUsage": {
                "prompt": token_usage.get("prompt_tokens", 0),
                "completion": token_usage.get("completion_tokens", 0),
            },
        }
        self.__send_event(event)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
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
        event = {
            "event": "start",
            "type": "tool",
            "userId": (metadata or {}).get("userId"),
            "runId": str(run_id),
            "parentRunId": str(parent_run_id) if parent_run_id else None,
            "name": serialized.get("name"),
            "input": input_str,
            "tags": tags,
            "metadata": metadata,
        }
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

        agentName = (metadata or {}).get("agentName")
        if agentName is not None:
            type = "agent"
            name = agentName
        if name == "AgentExecutor" or name == "PlanAndExecute":
            type = "agent"
        event = {
            "event": "start",
            "type": type,
            "userId": (metadata or {}).get("userId"),
            "runId": str(run_id),
            "parentRunId": str(parent_run_id) if parent_run_id else None,
            "input": inputs.get("input", inputs),
            "tags": tags,
            "metadata": metadata,
            "name": serialized.get("id", [None, None, None, None])[3],
        }

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
            "output": outputs.get("output", outputs),
        }
        self.__send_event(event)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
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
            "input": action.tool_input,
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
            "output": finish.return_values,
        }
        self.__send_event(event)


__all__ = ["LLMonitorCallbackHandler"]

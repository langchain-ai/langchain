from logging import INFO, Logger, getLevelName
from typing import Any, Dict, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from langchain.utils.input import get_colored_text


class LoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler that logs to the input Logger."""

    def __init__(
        self,
        logger: Logger,
        method_levels: Optional[Dict[str, int]] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Initialize.

        Args:
            logger: Logger being wrapped.
            method_levels: Optional overrides on the log level (default is
                logging.INFO) used within each method. For example, passing
                {"on_chain_start": logging.DEBUG} will lead to
                LoggingCallbackHandler.on_chain_start logging at a debug level.
            extra: Optional extra to pass to each log invocation.
        """
        self._logger = logger
        self._method_levels = method_levels or {}
        self._extra = extra

    def _log_text(
        self, method_name: str, text: str, color: Optional[str] = None
    ) -> None:
        """Log the input text to the contained logger."""
        if color:
            text = get_colored_text(text, color)
        log_level: int = self._method_levels.get(method_name, INFO)
        log_method_name = getLevelName(level=log_level).lower()
        getattr(self._logger, log_method_name)(text, extra=self._extra)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        self._log_text(
            "on_chain_start",
            text=f"\n\n\033[1m> Entering new {class_name} chain...\033[0m",
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self._log_text("on_chain_end", text="\n\033[1m> Finished chain.\033[0m")

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        self._log_text("on_agent_action", text=action.log, color=color)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        text = observation_prefix or ""
        text = f"{text}\n{output}" if text else output
        text = f"{text}\n{llm_prefix}" if llm_prefix else text
        self._log_text("on_tool_end", text=text, color=color)

    def on_text(
        self, text: str, color: Optional[str] = None, end: str = "", **kwargs: Any
    ) -> None:
        self._log_text("on_text", text=text, color=color)

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        self._log_text("on_agent_finish", text=finish.log, color=color)

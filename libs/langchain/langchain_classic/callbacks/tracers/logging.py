__all__ = ["LoggingCallbackHandler"]

import logging
from typing import Any
from uuid import UUID

from langchain_core.exceptions import TracerException
from langchain_core.tracers.stdout import FunctionCallbackHandler
from langchain_core.utils.input import get_bolded_text, get_colored_text
from typing_extensions import override


class LoggingCallbackHandler(FunctionCallbackHandler):
    """Tracer that logs via the input Logger."""

    name: str = "logging_callback_handler"

    def __init__(
        self,
        logger: logging.Logger,
        log_level: int = logging.INFO,
        extra: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LoggingCallbackHandler.

        Args:
            logger: the logger to use for logging
            log_level: the logging level (default: logging.INFO)
            extra: the extra context to log (default: None)
            **kwargs: additional keyword arguments.
        """
        log_method = getattr(logger, logging.getLevelName(level=log_level).lower())

        def callback(text: str) -> None:
            log_method(text, extra=extra)

        super().__init__(function=callback, **kwargs)

    @override
    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            crumbs_str = f"[{self.get_breadcrumbs(run=self._get_run(run_id=run_id))}] "
        except TracerException:
            crumbs_str = ""
        self.function_callback(
            f"{get_colored_text('[text]', color='blue')}"
            f" {get_bolded_text(f'{crumbs_str}New text:')}\n{text}",
        )

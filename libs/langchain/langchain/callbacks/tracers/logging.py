__all__ = ["LoggingCallbackHandler"]

import logging
from typing import Any, Optional
from uuid import UUID

from langchain_core.exceptions import TracerException
from langchain_core.tracers.stdout import FunctionCallbackHandler
from langchain_core.utils.input import get_bolded_text, get_colored_text


class LoggingCallbackHandler(FunctionCallbackHandler):
    """Tracer that logs via the input Logger."""

    name: str = "logging_callback_handler"

    def __init__(
        self,
        logger: logging.Logger,
        log_level: int = logging.INFO,
        extra: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        log_method = getattr(logger, logging.getLevelName(level=log_level).lower())

        def callback(text: str) -> None:
            log_method(text, extra=extra)

        super().__init__(function=callback, **kwargs)

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            crumbs_str = f"[{self.get_breadcrumbs(run=self._get_run(run_id=run_id))}] "
        except TracerException:
            crumbs_str = ""
        self.function_callback(
            f'{get_colored_text("[text]", color="blue")}'
            f' {get_bolded_text(f"{crumbs_str}New text:")}\n{text}'
        )

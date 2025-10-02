"""Callback Handler that writes to a file."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TextIO, cast

from typing_extensions import Self, override

from langchain_core._api import warn_deprecated
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.utils.input import print_text

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish


_GLOBAL_DEPRECATION_WARNED = False


class FileCallbackHandler(BaseCallbackHandler):
    """Callback Handler that writes to a file.

    This handler supports both context manager usage (recommended) and direct
    instantiation (deprecated) for backwards compatibility.

    Examples:
        Using as a context manager (recommended):

        .. code-block:: python

            with FileCallbackHandler("output.txt") as handler:
                # Use handler with your chain/agent
                chain.invoke(inputs, config={"callbacks": [handler]})

        Direct instantiation (deprecated):

        .. code-block:: python

            handler = FileCallbackHandler("output.txt")
            # File remains open until handler is garbage collected
            try:
                chain.invoke(inputs, config={"callbacks": [handler]})
            finally:
                handler.close()  # Explicit cleanup recommended

    Args:
        filename: The file path to write to.
        mode: The file open mode. Defaults to ``'a'`` (append).
        color: Default color for text output. Defaults to ``None``.

    .. note::
        When not used as a context manager, a deprecation warning will be issued
        on first use. The file will be opened immediately in ``__init__`` and closed
        in ``__del__`` or when ``close()`` is called explicitly.

    """

    def __init__(
        self, filename: str, mode: str = "a", color: Optional[str] = None
    ) -> None:
        """Initialize the file callback handler.

        Args:
            filename: Path to the output file.
            mode: File open mode (e.g., ``'w'``, ``'a'``, ``'x'``). Defaults to ``'a'``.
            color: Default text color for output. Defaults to ``None``.

        """
        self.filename = filename
        self.mode = mode
        self.color = color
        self._file_opened_in_context = False
        self.file: TextIO = cast(
            "TextIO",
            # Open the file in the specified mode with UTF-8 encoding.
            Path(self.filename).open(self.mode, encoding="utf-8"),  # noqa: SIM115
        )

    def __enter__(self) -> Self:
        """Enter the context manager.

        Returns:
            The FileCallbackHandler instance.

        .. note::
            The file is already opened in ``__init__``, so this just marks that
            the handler is being used as a context manager.

        """
        self._file_opened_in_context = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the context manager and close the file.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.

        """
        self.close()

    def __del__(self) -> None:
        """Destructor to cleanup when done."""
        self.close()

    def close(self) -> None:
        """Close the file if it's open.

        This method is safe to call multiple times and will only close
        the file if it's currently open.

        """
        if hasattr(self, "file") and self.file and not self.file.closed:
            self.file.close()

    def _write(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
    ) -> None:
        """Write text to the file with deprecation warning if needed.

        Args:
            text: The text to write to the file.
            color: Optional color for the text. Defaults to ``self.color``.
            end: String appended after the text. Defaults to ``""``.
            file: Optional file to write to. Defaults to ``self.file``.

        Raises:
            RuntimeError: If the file is closed or not available.

        """
        global _GLOBAL_DEPRECATION_WARNED  # noqa: PLW0603
        if not self._file_opened_in_context and not _GLOBAL_DEPRECATION_WARNED:
            warn_deprecated(
                since="0.3.67",
                pending=True,
                message=(
                    "Using FileCallbackHandler without a context manager is "
                    "deprecated. Use 'with FileCallbackHandler(...) as "
                    "handler:' instead."
                ),
            )
            _GLOBAL_DEPRECATION_WARNED = True

        if not hasattr(self, "file") or self.file is None or self.file.closed:
            msg = "File is not open. Use FileCallbackHandler as a context manager."
            raise RuntimeError(msg)

        print_text(text, file=self.file, color=color, end=end)

    @override
    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """Print that we are entering a chain.

        Args:
            serialized: The serialized chain information.
            inputs: The inputs to the chain.
            **kwargs: Additional keyword arguments that may contain ``'name'``.

        """
        name = (
            kwargs.get("name")
            or serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
            or "<unknown>"
        )
        self._write(f"\n\n> Entering new {name} chain...", end="\n")

    @override
    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Print that we finished a chain.

        Args:
            outputs: The outputs of the chain.
            **kwargs: Additional keyword arguments.

        """
        self._write("\n> Finished chain.", end="\n")

    @override
    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Handle agent action by writing the action log.

        Args:
            action: The agent action containing the log to write.
            color: Color override for this specific output. If ``None``, uses
                ``self.color``.
            **kwargs: Additional keyword arguments.

        """
        self._write(action.log, color=color or self.color)

    @override
    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool end by writing the output with optional prefixes.

        Args:
            output: The tool output to write.
            color: Color override for this specific output. If ``None``, uses
                ``self.color``.
            observation_prefix: Optional prefix to write before the output.
            llm_prefix: Optional prefix to write after the output.
            **kwargs: Additional keyword arguments.

        """
        if observation_prefix is not None:
            self._write(f"\n{observation_prefix}")
        self._write(output)
        if llm_prefix is not None:
            self._write(f"\n{llm_prefix}")

    @override
    def on_text(
        self, text: str, color: Optional[str] = None, end: str = "", **kwargs: Any
    ) -> None:
        """Handle text output.

        Args:
            text: The text to write.
            color: Color override for this specific output. If ``None``, uses
                ``self.color``.
            end: String appended after the text. Defaults to ``""``.
            **kwargs: Additional keyword arguments.

        """
        self._write(text, color=color or self.color, end=end)

    @override
    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Handle agent finish by writing the finish log.

        Args:
            finish: The agent finish object containing the log to write.
            color: Color override for this specific output. If ``None``, uses
                ``self.color``.
            **kwargs: Additional keyword arguments.

        """
        self._write(finish.log, color=color or self.color, end="\n")

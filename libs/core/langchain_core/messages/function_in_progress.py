from typing import Any, List, Literal

from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class FunctionInProgressMessage(BaseMessage):
    """Function in progress Message"""

    type: Literal["function_in_progress"] = "function_in_progress"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]


FunctionInProgressMessage.update_forward_refs()


class FunctionInProgressMessageChunk(FunctionInProgressMessage, BaseMessageChunk):
    """Function in Progress Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["FunctionInProgressMessage"] = "FunctionInProgressMessage"  # type: ignore[assignment]

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        # При суммировании чанков сообщений, пропускаем чанки, что функция выполнялась
        # для суммы сообщений эти чанки прогресса не так важны
        return other

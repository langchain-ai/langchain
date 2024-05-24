from typing import Any, Dict, List

from langchain_core.output_parsers.transform import BaseTransformOutputParser


class ChatCompletionsOutputParser(BaseTransformOutputParser[Dict[str, Any]]):
    """
    OutputParser that wraps the string output into a ChatCompletions
    structured format.
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "output_parser"]

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "openai_style"

    def parse(self, text: str) -> Dict[str, Any]:
        """Returns the input text wrapped in an OpenAI-like response structure."""
        try:
            from mlflow.models.rag_signatures import (
                ChainCompletionChoice,
                ChatCompletionResponse,
                Message,
            )
        except ImportError:
            raise ImportError(
                "mlflow 2.13.0 or higher is required to use this output parser. "
            )
        return ChatCompletionResponse(
            choices=[ChainCompletionChoice(message=Message(content=text))]
        ).asdict()


class StrObjOutputParser(BaseTransformOutputParser[Dict[str, Any]]):
    """OutputParser that wraps the string output into a structured format."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "output_parser"]

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "openai_style"

    def parse(self, text: str) -> Dict[str, Any]:
        """Returns the input text wrapped in a response structure with a string."""
        try:
            from mlflow.models.rag_signatures import StringResponse
        except ImportError:
            raise ImportError(
                "mlflow 2.13.0 or higher is required to use this output parser. "
            )
        return StringResponse(text=text).asdict()

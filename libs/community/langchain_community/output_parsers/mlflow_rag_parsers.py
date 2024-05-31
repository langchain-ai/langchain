from dataclasses import asdict
from typing import Any, Dict
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.messages import BaseMessage
from mlflow.models.rag_signatures import StringResponse, ChatCompletionResponse, ChainCompletionChoice, Message

class ChatCompletionsOutputParser(BaseTransformOutputParser[Dict[str, Any]]):
    """OutputParser that wraps the string output into a dictionary representation of an MLflow ChatCompletionResponse"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "mlflow_simplified_chat_completions"

    def parse(self, text: BaseMessage) -> Dict[str, Any]:
        return asdict(ChatCompletionResponse(
            choices=[ChainCompletionChoice(message=Message(content=text.content))]
        ))
    
class StringResponseOutputParser(BaseTransformOutputParser[Dict[str, Any]]):
    """OutputParser that wraps the string output into an dictionary representation of a MLflow StringResponse"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "mlflow_simplified_str_object"

    def parse(self, text: BaseMessage) -> Dict[str, Any]:
        return asdict(StringResponse(content=text.content))
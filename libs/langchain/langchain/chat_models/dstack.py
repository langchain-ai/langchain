from typing import Any, Dict, List, Optional, cast

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.llms.dstack import _BaseDstack
from langchain.pydantic_v1 import root_validator
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
)


class ChatDstack(_BaseDstack, BaseChatModel):
    """
    TODO
    """

    @root_validator()
    def require_tokenizer(cls, values: Dict) -> Dict:
        if values["tokenizer"] is None:
            raise ValueError(
                "Tokenizer is required for chat model."
                " Run `pip install --upgrade transformers`"
            )
        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self.tokenizer.apply_chat_template(
            _get_transformers_conversation(messages), tokenize=False
        )
        resp = self._tgi_generate(prompt, self._tgi_parameters(stop, **kwargs))
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=resp["generated_text"]))
            ]
        )

    # TODO async support


def _get_transformers_conversation(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    conversation = []
    for message in messages:
        content = cast(str, message.content)
        if isinstance(message, HumanMessage):
            conversation.append({"role": "user", "content": content})
        elif isinstance(message, AIMessage):
            conversation.append({"role": "assistant", "content": content})
        else:
            raise ValueError(f"Unsupported message type: {type(message).__name__}")
    return conversation

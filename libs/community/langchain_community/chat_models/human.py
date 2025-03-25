"""ChatModel wrapper which returns user input as the response.."""

from io import StringIO
from typing import Any, Callable, Dict, List, Mapping, Optional

import yaml
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    _message_from_dict,
    messages_to_dict,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from langchain_community.llms.utils import enforce_stop_tokens


def _display_messages(messages: List[BaseMessage]) -> None:
    dict_messages = messages_to_dict(messages)
    for message in dict_messages:
        yaml_string = yaml.dump(
            message,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=10000,
            line_break=None,
        )
        print("\n", "======= start of message =======", "\n\n")  # noqa: T201
        print(yaml_string)  # noqa: T201
        print("======= end of message =======", "\n\n")  # noqa: T201


def _collect_yaml_input(
    messages: List[BaseMessage], stop: Optional[List[str]] = None
) -> BaseMessage:
    """Collects and returns user input as a single string."""
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        if stop and any(seq in line for seq in stop):
            break
        lines.append(line)
    yaml_string = "\n".join(lines)

    # Try to parse the input string as YAML
    try:
        message = _message_from_dict(yaml.safe_load(StringIO(yaml_string)))
        if message is None:
            return HumanMessage(content="")
        if stop:
            if isinstance(message.content, str):
                message.content = enforce_stop_tokens(message.content, stop)
            else:
                raise ValueError("Cannot use when output is not a string.")
        return message
    except yaml.YAMLError:
        raise ValueError("Invalid YAML string entered.")
    except ValueError:
        raise ValueError("Invalid message entered.")


class HumanInputChatModel(BaseChatModel):
    """ChatModel which returns user input as the response."""

    input_func: Callable = Field(default_factory=lambda: _collect_yaml_input)
    message_func: Callable = Field(default_factory=lambda: _display_messages)
    separator: str = "\n"
    input_kwargs: Mapping[str, Any] = {}
    message_kwargs: Mapping[str, Any] = {}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "input_func": self.input_func.__name__,
            "message_func": self.message_func.__name__,
        }

    @property
    def _llm_type(self) -> str:
        """Returns the type of LLM."""
        return "human-input-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Displays the messages to the user and returns their input as a response.

        Args:
            messages (List[BaseMessage]): The messages to be displayed to the user.
            stop (Optional[List[str]]): A list of stop strings.
            run_manager (Optional[CallbackManagerForLLMRun]): Currently not used.

        Returns:
            ChatResult: The user's input as a response.
        """
        self.message_func(messages, **self.message_kwargs)
        user_input = self.input_func(messages, stop=stop, **self.input_kwargs)
        return ChatResult(generations=[ChatGeneration(message=user_input)])

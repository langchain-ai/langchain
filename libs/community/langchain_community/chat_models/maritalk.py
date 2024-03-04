from typing import Any, Dict, List, Optional, Union

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import Field


class ChatMaritalk(SimpleChatModel):
    """`MariTalk` Chat models API.

    This class allows interacting with the MariTalk chatbot API.
    To use it, you must provide an API key either through the constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatMaritalk
            chat = ChatMaritalk(api_key="your_api_key_here")
    """

    api_key: str
    """Your MariTalk API key."""

    temperature: float = Field(default=0.7, gt=0.0, lt=1.0)
    """Run inference with this temperature. 
    Must be in the closed interval [0.0, 1.0]."""

    max_tokens: int = Field(default=512, gt=0)
    """The maximum number of tokens to generate in the reply."""

    do_sample: bool = Field(default=True)
    """Whether or not to use sampling; use `True` to enable."""

    top_p: float = Field(default=0.95, gt=0.0, lt=1.0)
    """Nucleus sampling parameter controlling the size of 
    the probability mass considered for sampling."""

    system_message_workaround: bool = Field(default=True)
    """Whether to include a workaround for system messages 
    by adding them as a user message."""

    @property
    def _llm_type(self) -> str:
        """Identifies the LLM type as 'maritalk'."""
        return "maritalk"

    def parse_messages_for_model(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Union[str, List[Union[str, Dict[Any, Any]]]]]]:
        """
        Parses messages from LangChain's format to the format expected by
        the MariTalk API.

        Parameters:
            messages (List[BaseMessage]): A list of messages in LangChain
            format to be parsed.

        Returns:
            A list of messages formatted for the MariTalk API.
        """
        parsed_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                parsed_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                parsed_messages.append(
                    {"role": "assistant", "content": message.content}
                )
            elif isinstance(message, SystemMessage) and self.system_message_workaround:
                # Maritalk models do not understand system message.
                # #Instead we add these messages as user messages.
                parsed_messages.append({"role": "user", "content": message.content})
                parsed_messages.append({"role": "assistant", "content": "ok"})

        return parsed_messages

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Sends the parsed messages to the MariTalk API and returns the generated
        response or an error message.

        This method makes an HTTP POST request to the MariTalk API with the
        provided messages and other parameters.
        If the request is successful and the API returns a response,
        this method returns a string containing the answer.
        If the request is rate-limited or encounters another error,
        it returns a string with the error message.

        Parameters:
            messages (List[BaseMessage]): Messages to send to the model.
            stop (Optional[List[str]]): Tokens that will signal the model
                to stop generating further tokens.

        Returns:
            str: If the API call is successful, returns the answer.
                 If an error occurs (e.g., rate limiting), returns a string
                 describing the error.
        """
        try:
            url = "https://chat.maritaca.ai/api/chat/inference"
            headers = {"authorization": f"Key {self.api_key}"}
            stopping_tokens = stop if stop is not None else []

            parsed_messages = self.parse_messages_for_model(messages)

            data = {
                "messages": parsed_messages,
                "do_sample": self.do_sample,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stopping_tokens": stopping_tokens,
                **kwargs,
            }

            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 429:
                return "Rate limited, please try again soon"
            elif response.ok:
                return response.json().get("answer", "No answer found")

        except requests.exceptions.RequestException as e:
            return f"An error occurred: {str(e)}"

        # Fallback return statement, in case of unexpected code paths
        return "An unexpected error occurred"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
        Identifies the key parameters of the chat model for logging
        or tracking purposes.

        Returns:
            A dictionary of the key configuration parameters.
        """
        return {
            "system_message_workaround": self.system_message_workaround,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

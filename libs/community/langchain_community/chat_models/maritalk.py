from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import Field
from requests import Response
from requests.exceptions import HTTPError


class MaritalkHTTPError(HTTPError):
    def __init__(self, request_obj: Response) -> None:
        self.request_obj = request_obj
        try:
            response_json = request_obj.json()
            if "detail" in response_json:
                api_message = response_json["detail"]
            elif "message" in response_json:
                api_message = response_json["message"]
            else:
                api_message = response_json
        except Exception:
            api_message = request_obj.text

        self.message = api_message
        self.status_code = request_obj.status_code

    def __str__(self) -> str:
        status_code_meaning = HTTPStatus(self.status_code).phrase
        formatted_message = f"HTTP Error: {self.status_code} - {status_code_meaning}"
        formatted_message += f"\nDetail: {self.message}"
        return formatted_message


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

    model: str
    """Chose one of the available models: 
    - `sabia-2-medium`
    - `sabia-2-small`
    - `sabia-2-medium-2024-03-13`
    - `sabia-2-small-2024-03-13`
    - `maritalk-2024-01-08` (deprecated)"""

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
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"

            parsed_messages.append({"role": role, "content": message.content})
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
                "model": self.model,
                "do_sample": self.do_sample,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stopping_tokens": stopping_tokens,
                **kwargs,
            }

            response = requests.post(url, json=data, headers=headers)

            if response.ok:
                return response.json().get("answer", "No answer found")
            else:
                raise MaritalkHTTPError(response)

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
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

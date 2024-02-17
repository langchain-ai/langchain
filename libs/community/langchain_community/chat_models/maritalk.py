from typing import Any, List, Optional, Mapping
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import requests

class ChatMaritalk(SimpleChatModel):
    """`MariTalk` Chat models API.

    This class allows interacting with the MariTalk chatbot API. To use it, you must provide an API key either through the constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatMaritalk
            chat = ChatMaritalk(api_key="your_api_key_here")

    Attributes:
        api_key (str): Your MariTalk API key.
        temperature (float): Run inference with this temperature. Must be in the closed interval [0.0, 1.0].
        max_tokens (int): The maximum number of tokens to generate in the reply.
        do_sample (bool): Whether or not to use sampling; use `True` to enable.
        top_p (float): Nucleus sampling parameter controlling the size of the probability mass considered for sampling.
        system_message_workaround (bool): Whether to include a workaround for system messages by echoing them back.
    """

    def __init__(self, api_key: str, temperature: float = 0.7, max_tokens: int = 512, do_sample: bool = True, top_p: float = 0.95, system_message_workaround: bool = True):
        """
        Initializes the chat model with the necessary configuration to communicate with the MariTalk API.

        Parameters:
            api_key (str): The API key for authenticating requests to MariTalk.
            temperature (float): The temperature to use for generating responses. Controls randomness.
            max_tokens (int): The maximum number of tokens to generate for the response.
            do_sample (bool): Whether to use sampling for response generation.
            top_p (float): The nucleus sampling parameter, controlling the size of the probability mass considered for sampling.
            system_message_workaround (bool): Enables a workaround for handling system messages.
        """
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.do_sample = do_sample
        self.top_p = top_p
        self.system_message_workaround = system_message_workaround

    @property
    def _llm_type(self) -> str:
        """Identifies the LLM type as 'maritalk'."""
        return "maritalk"

    def parse_messages_for_model(self, messages: List[BaseMessage]):
        """
        Parses messages from LangChain's format to the format expected by the MariTalk API.

        Parameters:
            messages (List[BaseMessage]): A list of messages in LangChain format to be parsed.

        Returns:
            A list of messages formatted for the MariTalk API.
        """
        parsed_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                parsed_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                parsed_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage) and self.system_message_workaround:
                # Maritalk models do not understand system message. Instead we add these messages as user messages.
                parsed_messages.append({"role": "user", "content": message.content})
                parsed_messages.append({"role": "assistant", "content": "ok"})

        return parsed_messages

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Sends the parsed messages to the MariTalk API and returns the generated response.

        Parameters:
            messages (List[BaseMessage]): Messages to send to the model.
            stop (Optional[List[str]]): Tokens that will signal the model to stop generating further tokens.

        Returns:
            The generated response from the MariTalk API as a string.
        """
        url = "https://chat.maritaca.ai/api/chat/inference"
        headers = {"authorization": f"Key {self.api_key}"}

        parsed_messages = self.parse_messages_for_model(messages)

        data = {
            "messages": parsed_messages,
            "do_sample": self.do_sample,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 429:
            return "Rate limited, please try again soon"
        elif response.ok:
            answer = response.json()["answer"]
            return answer
        else:
            response.raise_for_status()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Identifies the key parameters of the chat model for logging or tracking purposes.

        Returns:
            A dictionary of the key configuration parameters.
        """
        return {
            "system_message_workaround": self.system_message_workaround,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens
        }

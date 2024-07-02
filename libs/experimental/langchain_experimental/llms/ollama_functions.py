from typing import (
    Any,
    Dict,
    List,
    Union,
    cast,
)

from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_experimental.llms.tool_calling_llm import (
    ToolCallingLLM,
    convert_to_tool_definition,
)


def convert_to_ollama_tool(tool: Any) -> Dict:
    """
    Converts a given tool object to an Ollama tool object.
    This function is provided here for backwards compatibility.
    """
    return convert_to_tool_definition(tool)


class OllamaFunctions(ToolCallingLLM, ChatOllama):
    """Function chat model that uses Ollama API."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _convert_messages_to_ollama_messages(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Union[str, List[str]]]]:
        ollama_messages: List = []
        for message in messages:
            role = ""
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage) or isinstance(message, ToolMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                raise ValueError("Received unsupported message type for Ollama.")

            content = ""
            images = []
            if isinstance(message.content, str):
                content = message.content
            else:
                for content_part in cast(List[Dict], message.content):
                    if content_part.get("type") == "text":
                        content += f"\n{content_part['text']}"
                    elif content_part.get("type") == "image_url":
                        if isinstance(content_part.get("image_url"), str):
                            image_url_components = content_part["image_url"].split(",")
                            # Support data:image/jpeg;base64,<image> format
                            # and base64 strings
                            if len(image_url_components) > 1:
                                images.append(image_url_components[1])
                            else:
                                images.append(image_url_components[0])
                        else:
                            raise ValueError(
                                "Only string image_url content parts are supported."
                            )
                    else:
                        raise ValueError(
                            "Unsupported message content type. "
                            "Must either have type 'text' or type 'image_url' "
                            "with a string 'image_url' field."
                        )

            ollama_messages.append(
                {
                    "role": role,
                    "content": content,
                    "images": images,
                }
            )

        return ollama_messages

    @property
    def _llm_type(self) -> str:
        return "ollama_functions"

import json
from typing import Any, Dict, List, Optional, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import SimpleChatModel
from langchain.llms.azureml_endpoint import (
    AzureMLBaseEndpoint,
    AzureMLEndpointApiType,
    ContentFormatterBase,
)


class LlamaContentFormatter(ContentFormatterBase):
    def __init__(self):
        raise TypeError(
            "`LlamaContentFormatter` is deprecated for chat models. Use "
            "`LlamaChatContentFormatter` instead."
        )


class LlamaChatContentFormatter(ContentFormatterBase):
    """Content formatter for `LLaMA`."""

    SUPPORTED_ROLES: List[str] = ["user", "assistant", "system"]

    @staticmethod
    def _convert_message_to_dict(message: BaseMessage) -> Dict:
        """Converts message to a dict according to role"""
        content = cast(str, message.content)
        if isinstance(message, HumanMessage):
            return {
                "role": "user",
                "content": ContentFormatterBase.escape_special_characters(content),
            }
        elif isinstance(message, AIMessage):
            return {
                "role": "assistant",
                "content": ContentFormatterBase.escape_special_characters(content),
            }
        elif isinstance(message, SystemMessage):
            return {
                "role": "system",
                "content": ContentFormatterBase.escape_special_characters(content),
            }
        elif (
            isinstance(message, ChatMessage)
            and message.role in LlamaChatContentFormatter.SUPPORTED_ROLES
        ):
            return {
                "role": message.role,
                "content": ContentFormatterBase.escape_special_characters(content),
            }
        else:
            supported = ",".join(
                [role for role in LlamaChatContentFormatter.SUPPORTED_ROLES]
            )
            raise ValueError(
                f"""Received unsupported role. 
                Supported roles for the LLaMa Foundation Model: {supported}"""
            )

    @property
    def supported_api_types(self) -> List[AzureMLEndpointApiType]:
        return [AzureMLEndpointApiType.realtime, AzureMLEndpointApiType.serverless]

    def format_request_payload(
        self, api_type: str, messages: List[BaseMessage], model_kwargs: Dict
    ) -> str:
        """Formats the request according to the chosen api"""
        chat_messages = [
            LlamaChatContentFormatter._convert_message_to_dict(message)
            for message in messages
        ]
        if api_type == AzureMLEndpointApiType.realtime:
            request_payload = json.dumps(
                {
                    "input_data": {
                        "input_string": chat_messages,
                        "parameters": model_kwargs,
                    }
                }
            )
        elif api_type == AzureMLEndpointApiType.serverless:
            request_payload = json.dumps({"messages": chat_messages, **model_kwargs})
        else:
            raise ValueError(
                f"`api_type` {api_type} is not supported by this formatter"
            )
        return str.encode(request_payload)

    def format_response_payload(self, api_type: str, output: bytes) -> str:
        """Formats response"""
        if api_type == AzureMLEndpointApiType.realtime:
            return json.loads(output)["output"]
        if api_type == AzureMLEndpointApiType.serverless:
            return json.loads(output)["choices"][0]["message"]["content"].strip()
        raise ValueError(f"`api_type` {api_type} is not supported by this formatter")


class AzureMLChatOnlineEndpoint(SimpleChatModel, AzureMLBaseEndpoint):
    """Azure ML Online Endpoint chat models.

    Example:
        .. code-block:: python
            azure_llm = AzureMLOnlineEndpoint(
                endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/score",
                endpoint_api_type=AzureMLApiType.realtime,
                endpoint_api_key="my-api-key",
                content_formatter=chat_content_formatter,
            )
    """  # noqa: E501

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "azureml_chat_endpoint"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to an AzureML Managed Online endpoint.
        Args:
            messages: The messages in the conversation with the chat model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                response = azureml_model("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}

        request_payload = self.content_formatter.format_request_payload(
            self.endpoint_api_type, messages, _model_kwargs
        )
        response_payload = self.http_client.call(request_payload, **kwargs)
        generated_text = self.content_formatter.format_response_payload(
            self.endpoint_api_type, response_payload
        )
        return generated_text

from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.anthropic import ChatAnthropicMessageConverter
from langchain.chat_models.bedrock import BaseBedrockChat
from langchain.schema.messages import AIMessage, BaseMessage
from langchain.schema.output import ChatGeneration, ChatResult


class BedrockChatAnthropic(BaseBedrockChat):
    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "bedrock-anthropic-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = ChatAnthropicMessageConverter().convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {**kwargs}
        if stop:
            params["stop_sequences"] = stop

        completion = self._prepare_input_and_invoke(prompt=prompt, stop=stop, **params)

        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

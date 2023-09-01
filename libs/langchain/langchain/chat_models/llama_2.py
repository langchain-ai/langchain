from typing import List, Any, Optional, Dict

from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatResult
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.output import ChatGeneration
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)

from langchain.pydantic_v1 import Field, root_validator
from transformers.pipelines import TextGenerationPipeline

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"


class ChatLlama2(BaseChatModel):
    pipeline: TextGenerationPipeline

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "llama-2-chat-hf"

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        if not hasattr(values["pipeline"], "task") or values["pipeline"].task != "text-generation":
            raise ValueError("The pipeline task should be 'text-generation'.")

        valid_models = (
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
        )

        if not hasattr(values["pipeline"], "model") or values["pipeline"].model.name_or_path not in valid_models:
            raise ValueError(f"The pipeline model name or path should be one of {valid_models}.")
        
        return values

    def _format_messages_as_text(self, messages: List[BaseMessage]) -> str:
        """ 
        # TODO: docstring
        https://huggingface.co/blog/llama2
        """
        prompt = ""

        for i, message in enumerate(messages):
            if isinstance(message, SystemMessage) and i != 0:
                raise ValueError("SystemMessage can only appear as the first message in the list.")
            elif isinstance(message, SystemMessage) and i == 0:
                prompt += f"<s>{B_INST} {B_SYS}\n{message.content}\n{E_SYS}\n\n"
            elif isinstance(message, HumanMessage) and i > 0:
                prompt += f"{message.content} {E_INST} "
            elif isinstance(message, HumanMessage) and i == 0:
                prompt += f"<s>{B_INST} {message.content} {E_INST} "
            elif isinstance(message, AIMessage):
                prompt += f"{message.content} </s><s>{B_INST} "
            elif isinstance(message, ChatMessage) and i == 0:
                prompt += f"<s>{B_INST} {message.role.capitalize()}: {message.content} {E_INST} "
            elif isinstance(message, ChatMessage) and i > 0:
                prompt += f"{message.role.capitalize()}: {message.content} {E_INST} "
        
        return prompt

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._format_messages_as_text(messages)

        # make sure that `return_full_text` is set to False
        # otherwise, pipeline will return prompt + generation
        kwargs["return_full_text"] = False
        kwargs["num_return_sequences"] = 1

        response = self.pipeline(prompt, **kwargs)[0]['generated_text']
        chat_generation = ChatGeneration(
            message=AIMessage(content=response),
        )
        return ChatResult(generations=[chat_generation])


# TODO:
# try adding stopping criteria
# tests for prompt generation
# streaming ?

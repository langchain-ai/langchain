import importlib.util
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import root_validator
from langchain.schema import ChatResult
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.output import ChatGeneration


class ChatHuggingFacePipeline(BaseChatModel, ABC):
    pipeline: Any

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "huggingface_pipeline_chat"

    @abstractmethod
    def format_messages_as_text(self, messages: List[BaseMessage]) -> str:
        """Method for parsing the list of LangChain Messages into string"""
        ...

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        if (
            not hasattr(values["pipeline"], "task")
            or values["pipeline"].task != "text-generation"
        ):
            raise ValueError("The pipeline task should be 'text-generation'.")

        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self.format_messages_as_text(messages)

        # make sure that `return_full_text` is set to False
        # otherwise, pipeline will return prompt + generation
        kwargs["return_full_text"] = False
        kwargs["num_return_sequences"] = 1

        if importlib.util.find_spec("torch") is not None:
            import torch

        if importlib.util.find_spec("transformers") is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList

        device = self.pipeline.device.type
        if device == "cuda":
            # in the multi-gpu case, stopping criteria tokens
            # need to be on the same device:
            device = f"{device}:{self.pipeline.device.index}"

        class CustomStoppingCriteria(StoppingCriteria):
            """
            A subclass of StoppingCriteria, used for defining custom stopping criteria
            for the generation process, apart from the standard End Of Sentence (EOS)
            token generation.

            This class allows for generation to be halted based on a list of specified
            token sequences, which might signify the end of a meaningful segment
            or passage within the generated text.
            """

            def __init__(
                self,
                stops: Optional[List[torch.Tensor]] = None,
                device: Union[torch.device, str, None] = None,
            ):
                """
                Args:
                    stops: A list of tensor sequences with individual,
                        tokenized stopping words.
                    device: The device (e.g., 'cpu', 'cuda', 'cuda:0')
                        on which to keep the stopping words tokens
                """
                super().__init__()
                stops = stops or []
                if device:
                    self.stops = [stop.to(device) for stop in stops]
                else:
                    self.stops = stops

            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
                **kwargs: Dict,
            ) -> bool:
                for stop_id in self.stops:
                    if (input_ids[0][-len(stop_id) :] == stop_id).all():
                        return True
                return False

        if stop:
            stopping_criteria_tokenized = [
                self.pipeline.tokenizer(
                    stopping_criterion, return_tensors="pt", add_special_tokens=False
                )["input_ids"]
                .squeeze()
                .to(device)
                for stopping_criterion in stop
            ]

            stopping_criteria = StoppingCriteriaList(
                [
                    CustomStoppingCriteria(
                        stops=stopping_criteria_tokenized, device=device
                    )
                ]
            )
        else:
            stopping_criteria = None

        response = self.pipeline(prompt, stopping_criteria=stopping_criteria, **kwargs)
        response = response[0]["generated_text"]
        chat_generation = ChatGeneration(
            message=AIMessage(content=response),
        )
        return ChatResult(generations=[chat_generation])


class ChatHFLlama2Pipeline(ChatHuggingFacePipeline):
    class InstructionTokens(Enum):
        def __str__(self) -> str:
            return self.value

        B_INST = "[INST]"
        E_INST = "[/INST]"

    class SystemTokens(Enum):
        def __str__(self) -> str:
            return self.value

        B_SYS = "<<SYS>>"
        E_SYS = "<</SYS>>"

    def format_messages_as_text(self, messages: List[BaseMessage]) -> str:
        """
        Transform List of Chat Messages to text following Meta's prompt guidelines.

        Prompt template with System Message:
        ```
        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s>
        ```

        Prompt template without System Message:
        ```
        <s>[INST] {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s>
        ```
        Source:
        https://github.com/facebookresearch/llama-recipes/blob/df77625e48c3994aef19702fb331215f7fb83494/docs/inference.md?plain=1#L124
        """
        prompt = ""

        for i, message in enumerate(messages):
            if isinstance(message, SystemMessage) and i != 0:
                raise ValueError(
                    "SystemMessage can only appear as the first message in the list."
                )
            elif isinstance(message, SystemMessage) and i == 0:
                prompt += (
                    f"<s>{self.InstructionTokens.B_INST} "
                    f"{self.SystemTokens.B_SYS}\n{message.content}\n"
                    f"{self.SystemTokens.E_SYS}\n\n"
                )
            elif isinstance(message, HumanMessage) and i > 0:
                prompt += f"{message.content} {self.InstructionTokens.E_INST} "
            elif isinstance(message, HumanMessage) and i == 0:
                prompt += (
                    f"<s>{self.InstructionTokens.B_INST} "
                    f"{message.content} {self.InstructionTokens.E_INST} "
                )
            elif isinstance(message, AIMessage):
                prompt += f"{message.content} </s><s>{self.InstructionTokens.B_INST} "
            else:
                raise ValueError(f"Unsupported Message type: {type(message)}")

        return prompt

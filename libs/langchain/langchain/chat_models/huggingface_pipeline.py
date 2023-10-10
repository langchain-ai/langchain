import importlib.util
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


class ChatHuggingFacePipeline(BaseChatModel):
    pipeline: Any

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "huggingface_pipeline_chat"

    @staticmethod
    def convert_lc_messages_to_hf_messages(
        messages: List[BaseMessage],
    ) -> List[Dict[str, str]]:
        """
        Method for converting the list of LangChain Messages into
        format required by Hugging Face.
        """
        output = []

        for message in messages:
            if isinstance(message, SystemMessage):
                output.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                output.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                output.append({"role": "assistant", "content": message.content})
            else:
                raise ValueError(
                    f"Unexpected message type: {type(message)}. "
                    "Expected one of [SystemMessage, HumanMessage, AIMessage]."
                )

        return output

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        if (
            not hasattr(values["pipeline"], "task")
            or values["pipeline"].task != "text-generation"
        ):
            raise ValueError("The pipeline task should be 'text-generation'.")

        if not hasattr(values["pipeline"], "apply_chat_template"):
            raise ValueError(
                "Your transformers module might be outdated. "
                "Please update it to ensure that tokenizer has the "
                "'apply_chat_template' method."
            )

        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        chat = self.convert_lc_messages_to_hf_messages(messages)
        prompt = self.pipeline.tokenizer.apply_chat_template(chat, tokenize=False)

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

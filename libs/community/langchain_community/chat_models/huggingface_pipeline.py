from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import Extra, Field

from langchain_community.adapters.openai import convert_message_to_dict
from langchain_community.llms.utils import enforce_stop_tokens

if TYPE_CHECKING:
    from transformers import Conversation

logger = logging.getLogger(__name__)


def _messages_to_conversation(messages: Sequence[BaseMessage]) -> Conversation:
    """Convert messages to transformers Conversation.

    Uses OpenAI message role conventions: AIMessage has role "assistant",
        HumanMessage has role "user".
    """
    from transformers import Conversation

    return Conversation(messages=[convert_message_to_dict(msg) for msg in messages])


class ChatHuggingFacePipeline(BaseChatModel):
    """HuggingFace Pipeline chat model API.

    To use, you should have the ``transformers`` python package installed.

    Only supports `conversational` task for now.

    Example using from_model_id:
        .. code-block:: python

            from langchain_community.chat_models import ChatHuggingFacePipeline

            chat = ChatHuggingFacePipeline.from_model_id(
                model_id="",
                pipeline_kwargs={"max_new_tokens": 10},
            )

    Example passing pipeline in directly:
        .. code-block:: python

            from langchain_community.chat_models import ChatHuggingFacePipeline
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

            model_id = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            pipe = pipeline(
                "conversational", model=model, tokenizer=tokenizer, max_new_tokens=10
            )
            chat = HuggingFacePipeline(pipeline=pipe)
    """

    pipeline: Any = Field(exclude=True)  #: :meta private:
    model_id: str
    """Model name to use."""
    model_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments passed to the model."""
    pipeline_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments passed to the pipeline."""
    messages_to_conversation: Callable = _messages_to_conversation

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        *,
        task: str = "conversational",
        device: Optional[int] = -1,
        device_map: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        pipeline_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> ChatHuggingFacePipeline:
        """Construct the pipeline object from model_id and task."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from transformers import pipeline as hf_pipeline
        except ImportError as e:
            raise ImportError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            ) from e

        _model_kwargs = model_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)

        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **_model_kwargs)
        except ImportError as e:
            raise ImportError(
                f"Could not load the {task} model due to missing dependencies."
            ) from e

        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = model.config.eos_token_id

        if (
            getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_loaded_in_8bit", False)
        ) and device is not None:
            logger.warning(
                f"Setting the `device` argument to None from {device} to avoid "
                "the error caused by attempting to move the model that was already "
                "loaded on the GPU using the Accelerate module to the same or "
                "another device."
            )
            device = None

        if device is not None and importlib.util.find_spec("torch") is not None:
            import torch

            cuda_device_count = torch.cuda.device_count()
            if device < -1 or (device >= cuda_device_count):
                raise ValueError(
                    f"Got device=={device}, "
                    f"device is required to be within [-1, {cuda_device_count})"
                )
            if device_map is not None and device < 0:
                device = None
            if device is not None and device < 0 and cuda_device_count > 0:
                logger.warning(
                    "Device has %d GPUs available. "
                    "Provide device={deviceId} to `from_model_id` to use available"
                    "GPUs for execution. deviceId is -1 (default) for CPU and "
                    "can be a positive integer associated with CUDA device id.",
                    cuda_device_count,
                )
        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }
        _pipeline_kwargs = pipeline_kwargs or {}
        pipeline = hf_pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            device=device,
            device_map=device_map,
            model_kwargs=_model_kwargs,
            **_pipeline_kwargs,
        )
        return cls(
            pipeline=pipeline,
            model_id=model_id,
            model_kwargs=_model_kwargs,
            pipeline_kwargs=_pipeline_kwargs,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
            "pipeline_kwargs": self.pipeline_kwargs,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        conversation = self.messages_to_conversation(messages)
        conversation = self.pipeline(conversation)
        generated_text = conversation.messages[-1]["content"]
        if stop:
            # Enforce stop tokens
            generated_text = enforce_stop_tokens(generated_text, stop)

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=generated_text))]
        )

    @property
    def _llm_type(self) -> str:
        return "chat_huggingface_pipeline"

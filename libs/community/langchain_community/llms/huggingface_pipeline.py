from __future__ import annotations

import importlib.util
import logging
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import Extra

from langchain_community.llms.utils import enforce_stop_tokens

DEFAULT_MODEL_ID = "gpt2"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text2text-generation", "text-generation", "summarization")
DEFAULT_BATCH_SIZE = 4

logger = logging.getLogger(__name__)


class HuggingFacePipeline(BaseLLM):
    """HuggingFace Pipeline API.

    To use, you should have the ``transformers`` python package installed.

    Only supports `text-generation`, `text2text-generation` and `summarization` for now.

    Example using from_model_id:
        .. code-block:: python

            from langchain_community.llms import HuggingFacePipeline
            hf = HuggingFacePipeline.from_model_id(
                model_id="gpt2",
                task="text-generation",
                pipeline_kwargs={"max_new_tokens": 10},
            )
    Example passing pipeline in directly:
        .. code-block:: python

            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_id = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
            )
            hf = HuggingFacePipeline(pipeline=pipe)
    """

    pipeline: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments passed to the model."""
    pipeline_kwargs: Optional[dict] = None
    """Keyword arguments passed to the pipeline."""
    batch_size: int = DEFAULT_BATCH_SIZE
    """Batch size to use when passing multiple documents to generate."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        task: str,
        device: Optional[int] = -1,
        device_map: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        pipeline_kwargs: Optional[dict] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs: Any,
    ) -> HuggingFacePipeline:
        """Construct the pipeline object from model_id and task."""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
            from transformers import pipeline as hf_pipeline

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)

        try:
            if task == "text-generation":
                model = AutoModelForCausalLM.from_pretrained(model_id, **_model_kwargs)
            elif task in ("text2text-generation", "summarization"):
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **_model_kwargs)
            else:
                raise ValueError(
                    f"Got invalid task {task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
        except ImportError as e:
            raise ValueError(
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
            batch_size=batch_size,
            model_kwargs=_model_kwargs,
            **_pipeline_kwargs,
        )
        if pipeline.task not in VALID_TASKS:
            raise ValueError(
                f"Got invalid task {pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        return cls(
            pipeline=pipeline,
            model_id=model_id,
            model_kwargs=_model_kwargs,
            pipeline_kwargs=_pipeline_kwargs,
            batch_size=batch_size,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
            "pipeline_kwargs": self.pipeline_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # List to hold all results
        text_generations: List[str] = []
        pipeline_kwargs = kwargs.get("pipeline_kwargs", {})

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]

            # Process batch of prompts
            responses = self.pipeline(batch_prompts, **pipeline_kwargs)

            # Process each response in the batch
            for j, response in enumerate(responses):
                if isinstance(response, list):
                    # if model returns multiple generations, pick the top one
                    response = response[0]

                if self.pipeline.task == "text-generation":
                    try:
                        from transformers.pipelines.text_generation import ReturnType

                        remove_prompt = (
                            self.pipeline._postprocess_params.get("return_type")
                            != ReturnType.NEW_TEXT
                        )
                    except Exception as e:
                        logger.warning(
                            f"Unable to extract pipeline return_type. "
                            f"Received error:\n\n{e}"
                        )
                        remove_prompt = True
                    if remove_prompt:
                        text = response["generated_text"][len(batch_prompts[j]) :]
                    else:
                        text = response["generated_text"]
                elif self.pipeline.task == "text2text-generation":
                    text = response["generated_text"]
                elif self.pipeline.task == "summarization":
                    text = response["summary_text"]
                else:
                    raise ValueError(
                        f"Got invalid task {self.pipeline.task}, "
                        f"currently only {VALID_TASKS} are supported"
                    )
                if stop:
                    # Enforce stop tokens
                    text = enforce_stop_tokens(text, stop)

                # Append the processed text to results
                text_generations.append(text)

        return LLMResult(
            generations=[[Generation(text=text)] for text in text_generations]
        )

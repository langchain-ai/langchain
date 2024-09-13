from __future__ import annotations

import importlib.util
import logging
from typing import Any, Iterator, List, Mapping, Optional

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from pydantic import ConfigDict

DEFAULT_MODEL_ID = "gpt2"
DEFAULT_TASK = "text-generation"
VALID_TASKS = (
    "text2text-generation",
    "text-generation",
    "summarization",
    "translation",
)
DEFAULT_BATCH_SIZE = 4

logger = logging.getLogger(__name__)


@deprecated(
    since="0.0.37",
    removal="1.0",
    alternative_import="langchain_huggingface.HuggingFacePipeline",
)
class HuggingFacePipeline(BaseLLM):
    """HuggingFace Pipeline API.

    To use, you should have the ``transformers`` python package installed.

    Only supports `text-generation`, `text2text-generation`, `summarization` and
    `translation`  for now.

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

    pipeline: Any = None  #: :meta private:
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments passed to the model."""
    pipeline_kwargs: Optional[dict] = None
    """Keyword arguments passed to the pipeline."""
    batch_size: int = DEFAULT_BATCH_SIZE
    """Batch size to use when passing multiple documents to generate."""

    model_config = ConfigDict(
        extra="forbid",
    )

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        task: str,
        backend: str = "default",
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
            raise ImportError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)

        try:
            if task == "text-generation":
                if backend == "openvino":
                    try:
                        from optimum.intel.openvino import OVModelForCausalLM

                    except ImportError:
                        raise ImportError(
                            "Could not import optimum-intel python package. "
                            "Please install it with: "
                            "pip install 'optimum[openvino,nncf]' "
                        )
                    try:
                        # use local model
                        model = OVModelForCausalLM.from_pretrained(
                            model_id, **_model_kwargs
                        )

                    except Exception:
                        # use remote model
                        model = OVModelForCausalLM.from_pretrained(
                            model_id, export=True, **_model_kwargs
                        )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, **_model_kwargs
                    )
            elif task in ("text2text-generation", "summarization", "translation"):
                if backend == "openvino":
                    try:
                        from optimum.intel.openvino import OVModelForSeq2SeqLM

                    except ImportError:
                        raise ImportError(
                            "Could not import optimum-intel python package. "
                            "Please install it with: "
                            "pip install 'optimum[openvino,nncf]' "
                        )
                    try:
                        # use local model
                        model = OVModelForSeq2SeqLM.from_pretrained(
                            model_id, **_model_kwargs
                        )

                    except Exception:
                        # use remote model
                        model = OVModelForSeq2SeqLM.from_pretrained(
                            model_id, export=True, **_model_kwargs
                        )
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_id, **_model_kwargs
                    )
            else:
                raise ValueError(
                    f"Got invalid task {task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
        except ImportError as e:
            raise ImportError(
                f"Could not load the {task} model due to missing dependencies."
            ) from e

        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = model.config.eos_token_id

        if (
            (
                getattr(model, "is_loaded_in_4bit", False)
                or getattr(model, "is_loaded_in_8bit", False)
            )
            and device is not None
            and backend == "default"
        ):
            logger.warning(
                f"Setting the `device` argument to None from {device} to avoid "
                "the error caused by attempting to move the model that was already "
                "loaded on the GPU using the Accelerate module to the same or "
                "another device."
            )
            device = None

        if (
            device is not None
            and importlib.util.find_spec("torch") is not None
            and backend == "default"
        ):
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
        if device is not None and device_map is not None and backend == "openvino":
            logger.warning("Please set device for OpenVINO through: `model_kwargs`")
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

        default_pipeline_kwargs = self.pipeline_kwargs if self.pipeline_kwargs else {}
        pipeline_kwargs = kwargs.get("pipeline_kwargs", default_pipeline_kwargs)

        skip_prompt = kwargs.get("skip_prompt", False)

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]

            # Process batch of prompts
            responses = self.pipeline(
                batch_prompts,
                **pipeline_kwargs,
            )

            # Process each response in the batch
            for j, response in enumerate(responses):
                if isinstance(response, list):
                    # if model returns multiple generations, pick the top one
                    response = response[0]

                if self.pipeline.task == "text-generation":
                    text = response["generated_text"]
                elif self.pipeline.task == "text2text-generation":
                    text = response["generated_text"]
                elif self.pipeline.task == "summarization":
                    text = response["summary_text"]
                elif self.pipeline.task in "translation":
                    text = response["translation_text"]
                else:
                    raise ValueError(
                        f"Got invalid task {self.pipeline.task}, "
                        f"currently only {VALID_TASKS} are supported"
                    )
                if skip_prompt:
                    text = text[len(batch_prompts[j]) :]
                # Append the processed text to results
                text_generations.append(text)

        return LLMResult(
            generations=[[Generation(text=text)] for text in text_generations]
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        from threading import Thread

        import torch
        from transformers import (
            StoppingCriteria,
            StoppingCriteriaList,
            TextIteratorStreamer,
        )

        pipeline_kwargs = kwargs.get("pipeline_kwargs", {})
        skip_prompt = kwargs.get("skip_prompt", True)

        if stop is not None:
            stop = self.pipeline.tokenizer.convert_tokens_to_ids(stop)
        stopping_ids_list = stop or []

        class StopOnTokens(StoppingCriteria):
            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
                **kwargs: Any,
            ) -> bool:
                for stop_id in stopping_ids_list:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        inputs = self.pipeline.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(
            self.pipeline.tokenizer,
            timeout=60.0,
            skip_prompt=skip_prompt,
            skip_special_tokens=True,
        )
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
            **pipeline_kwargs,
        )
        t1 = Thread(target=self.pipeline.model.generate, kwargs=generation_kwargs)
        t1.start()

        for char in streamer:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

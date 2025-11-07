from __future__ import annotations  # type: ignore[import-not-found]

import importlib.util
import logging
from collections.abc import Iterator, Mapping
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from pydantic import ConfigDict, model_validator

from langchain_huggingface.utils.import_utils import (
    IMPORT_ERROR,
    is_ipex_available,
    is_openvino_available,
    is_optimum_intel_available,
    is_optimum_intel_version,
)

DEFAULT_MODEL_ID = "gpt2"
DEFAULT_TASK = "text-generation"
VALID_TASKS = (
    "text2text-generation",
    "text-generation",
    "image-text-to-text",
    "summarization",
    "translation",
)
DEFAULT_BATCH_SIZE = 4
_MIN_OPTIMUM_VERSION = "1.21"


logger = logging.getLogger(__name__)


class HuggingFacePipeline(BaseLLM):
    """HuggingFace Pipeline API.

    To use, you should have the `transformers` python package installed.

    Only supports `text-generation`, `text2text-generation`, `image-text-to-text`,
    `summarization` and `translation`  for now.

    Example using from_model_id:
        ```python
        from langchain_huggingface import HuggingFacePipeline

        hf = HuggingFacePipeline.from_model_id(
            model_id="gpt2",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 10},
        )
        ```

    Example passing pipeline in directly:
        ```python
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        model_id = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=10,
        )
        hf = HuggingFacePipeline(pipeline=pipe)
        ```
    """

    pipeline: Any = None

    model_id: str | None = None
    """The model name. If not set explicitly by the user,
    it will be inferred from the provided pipeline (if available).
    If neither is provided, the DEFAULT_MODEL_ID will be used."""

    model_kwargs: dict | None = None
    """Keyword arguments passed to the model."""

    pipeline_kwargs: dict | None = None
    """Keyword arguments passed to the pipeline."""

    batch_size: int = DEFAULT_BATCH_SIZE
    """Batch size to use when passing multiple documents to generate."""

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def pre_init_validator(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Ensure model_id is set either by pipeline or user input."""
        if "model_id" not in values:
            if values.get("pipeline"):
                values["model_id"] = values["pipeline"].model.name_or_path
            else:
                values["model_id"] = DEFAULT_MODEL_ID
        return values

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        task: str,
        backend: str = "default",
        device: int | None = None,
        device_map: str | None = None,
        model_kwargs: dict | None = None,
        pipeline_kwargs: dict | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs: Any,
    ) -> HuggingFacePipeline:
        """Construct the pipeline object from model_id and task."""
        try:
            from transformers import (  # type: ignore[import]
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
            from transformers import pipeline as hf_pipeline  # type: ignore[import]

        except ImportError as e:
            msg = (
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )
            raise ValueError(msg) from e

        _model_kwargs = model_kwargs.copy() if model_kwargs else {}
        if device_map is not None:
            if device is not None:
                msg = (
                    "Both `device` and `device_map` are specified. "
                    "`device` will override `device_map`. "
                    "You will most likely encounter unexpected behavior."
                    "Please remove `device` and keep "
                    "`device_map`."
                )
                raise ValueError(msg)

            if "device_map" in _model_kwargs:
                msg = "`device_map` is already specified in `model_kwargs`."
                raise ValueError(msg)

            _model_kwargs["device_map"] = device_map
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)

        if backend in {"openvino", "ipex"}:
            if task not in VALID_TASKS:
                msg = (
                    f"Got invalid task {task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
                raise ValueError(msg)

            err_msg = f"Backend: {backend} {IMPORT_ERROR.format(f'optimum[{backend}]')}"
            if not is_optimum_intel_available():
                raise ImportError(err_msg)

            # TODO: upgrade _MIN_OPTIMUM_VERSION to 1.22 after release
            min_optimum_version = (
                "1.22"
                if backend == "ipex" and task != "text-generation"
                else _MIN_OPTIMUM_VERSION
            )
            if is_optimum_intel_version("<", min_optimum_version):
                msg = (
                    f"Backend: {backend} requires optimum-intel>="
                    f"{min_optimum_version}. You can install it with pip: "
                    "`pip install --upgrade --upgrade-strategy eager "
                    f"`optimum[{backend}]`."
                )
                raise ImportError(msg)

            if backend == "openvino":
                if not is_openvino_available():
                    raise ImportError(err_msg)

                from optimum.intel import (  # type: ignore[import]
                    OVModelForCausalLM,
                    OVModelForSeq2SeqLM,
                )

                model_cls = (
                    OVModelForCausalLM
                    if task == "text-generation"
                    else OVModelForSeq2SeqLM
                )
            else:
                if not is_ipex_available():
                    raise ImportError(err_msg)

                if task == "text-generation":
                    from optimum.intel import (
                        IPEXModelForCausalLM,  # type: ignore[import]
                    )

                    model_cls = IPEXModelForCausalLM
                else:
                    from optimum.intel import (
                        IPEXModelForSeq2SeqLM,  # type: ignore[import]
                    )

                    model_cls = IPEXModelForSeq2SeqLM

        else:
            model_cls = (
                AutoModelForCausalLM
                if task == "text-generation"
                else AutoModelForSeq2SeqLM
            )

        model = model_cls.from_pretrained(model_id, **_model_kwargs)

        if tokenizer.pad_token is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None and isinstance(
                model.config.eos_token_id, int
            ):
                tokenizer.pad_token_id = model.config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

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
                msg = (
                    f"Got device=={device}, "
                    f"device is required to be within [-1, {cuda_device_count})"
                )
                raise ValueError(msg)
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
        pipeline = hf_pipeline(  # type: ignore[call-overload]
            task=task,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            model_kwargs=_model_kwargs,
            **_pipeline_kwargs,
        )
        if pipeline.task not in VALID_TASKS:
            msg = (
                f"Got invalid task {pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
            raise ValueError(msg)
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
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        # List to hold all results
        text_generations: list[str] = []
        pipeline_kwargs = kwargs.get("pipeline_kwargs", {})
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

                if (
                    self.pipeline.task == "text-generation"
                    or self.pipeline.task == "text2text-generation"
                    or self.pipeline.task == "image-text-to-text"
                ):
                    text = response["generated_text"]
                elif self.pipeline.task == "summarization":
                    text = response["summary_text"]
                elif self.pipeline.task in "translation":
                    text = response["translation_text"]
                else:
                    msg = (
                        f"Got invalid task {self.pipeline.task}, "
                        f"currently only {VALID_TASKS} are supported"
                    )
                    raise ValueError(msg)
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
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
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
                return any(input_ids[0][-1] == stop_id for stop_id in stopping_ids_list)

        stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        streamer = TextIteratorStreamer(
            self.pipeline.tokenizer,
            timeout=60.0,
            skip_prompt=skip_prompt,
            skip_special_tokens=True,
        )
        generation_kwargs = dict(
            text_inputs=prompt,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
            **pipeline_kwargs,
        )
        t1 = Thread(target=self.pipeline, kwargs=generation_kwargs)
        t1.start()

        for char in streamer:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

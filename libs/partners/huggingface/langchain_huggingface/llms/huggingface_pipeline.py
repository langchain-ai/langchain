import logging
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.utils import pre_init
from pydantic import ConfigDict, Field, model_validator

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

    To use, you should have the ``transformers`` python package installed.
    The `pipeline` attribute is expected to receive a
    `transformers.Pipeline` object.

    Example using from_model_id:
        .. code-block:: python

            from langchain_huggingface import HuggingFacePipeline
            hf = HuggingFacePipeline.from_model_id(
                model_id="gpt2",
                task="text-generation",
                pipeline_kwargs={"max_new_tokens": 10},
            )

    Example passing pipeline in directly:
        .. code-block:: python

            from langchain_huggingface import HuggingFacePipeline
            from transformers import pipeline

            pipe = pipeline("text-generation", model="gpt2", max_new_tokens=10)
            hf = HuggingFacePipeline(pipeline=pipe)
    """

    pipeline: Any = Field(default=None, exclude=True)
    model_id: str = "gpt2"
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments passed to the model."""
    pipeline_kwargs: Optional[dict] = None
    """Keyword arguments passed to the pipeline."""
    batch_size: int = DEFAULT_BATCH_SIZE
    """Batch size to use when passing multiple documents to generate."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate that python package exists in environment."""
        try:
            from transformers import Pipeline

            if not isinstance(values.get("pipeline"), Pipeline):
                pass
        except ImportError:
            pass

        return values

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        task: str,
        backend: str = "default",
        device: Optional[int] = None,
        device_map: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        pipeline_kwargs: Optional[dict] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs: Any,
    ) -> "HuggingFacePipeline":
        """Construct the pipeline object from model_id and task."""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
            from transformers import pipeline as hf_pipeline
        except ImportError as e:
            msg = "Could not import transformers python package. "
            raise ImportError(msg) from e

        _model_kwargs = model_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)

        try:
            if task == "text-generation":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, **_model_kwargs
                )
            elif task in (
                "text2text-generation",
                "summarization",
                "translation",
            ) or task.startswith("translation_"):
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id, **_model_kwargs
                )
            else:
                msg = (
                    f"Got invalid task {task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
                raise ValueError(msg)
        except ImportError as e:
            msg = "Could not load the model. "
            raise ImportError(msg) from e

        if tokenizer.model_max_length > 1000000:
            tokenizer.model_max_length = 512

        _pipeline_kwargs = pipeline_kwargs or {}

        if backend == "ipex" and task != "text-generation":
            try:
                from optimum.intel.pipelines import pipeline as ipex_pipeline

                from langchain_huggingface.llms.utils import check_optimum_version

                check_optimum_version(_MIN_OPTIMUM_VERSION)
            except ImportError as e:
                msg = "Could not import optimum python package. "
                raise ImportError(msg) from e
            pipeline = ipex_pipeline(
                task=task,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                model_kwargs=_model_kwargs,
                **_pipeline_kwargs,
            )
        else:
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
        if pipeline.task not in VALID_TASKS and not pipeline.task.startswith(
            "translation_"
        ):
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
            **kwargs,
        )

    @pre_init
    def validate_pipeline(cls, values: dict) -> dict:
        """Validate that pipeline is a valid HuggingFace pipeline."""
        if values.get("pipeline") is not None:
            pipeline = values["pipeline"]
            if pipeline.task not in VALID_TASKS and not pipeline.task.startswith(
                "translation_"
            ):
                msg = (
                    f"Got invalid task {pipeline.task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
                raise ValueError(msg)
        return values

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
        skip_prompt = kwargs.get("skip_prompt", False)

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]

            # Process batch of prompts
            responses = self.pipeline(batch_prompts, **pipeline_kwargs)

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
                elif self.pipeline.task.startswith("translation"):
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
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        pipeline_kwargs = kwargs.get("pipeline_kwargs", {})
        skip_prompt = kwargs.get("skip_prompt", False)

        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(
            self.pipeline.tokenizer,
            timeout=60.0,
            skip_prompt=skip_prompt,
            skip_special_tokens=True,
        )
        pipeline_kwargs = {**pipeline_kwargs, "streamer": streamer}
        t = Thread(target=self.pipeline, args=[prompt], kwargs=pipeline_kwargs)
        t.start()
        for char in streamer:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"

    @property
    def _identifying_params(self) -> dict:
        """Get the identifying parameters."""
        return {
            **{"model_id": self.model_id},
            **{"model_kwargs": self.model_kwargs},
            **{"pipeline_kwargs": self.pipeline_kwargs},
        }

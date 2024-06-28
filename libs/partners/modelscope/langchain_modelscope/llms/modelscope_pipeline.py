from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult

DEFAULT_MODEL_ID = "qwen/Qwen2-0.5B-Instruct"
DEFAULT_TASK = "chat"
VALID_TASKS = (
    "chat",
    "text-generation",
)
DEFAULT_BATCH_SIZE = 4

logger = logging.getLogger(__name__)


class ModelScopePipeline(BaseLLM):
    """ModelScope Pipeline API.

    To use, you should have the ``modelscope[nlp]`` python package installed,
    you can install with ``pip install ms-swift "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html``.

    Only supports `chat` task for now.

    Example using from_model_id:
        .. code-block:: python

            from langchain_modelscope import ModelScopePipeline
            modelscope = ModelScopePipeline.from_model_id(
                model_id="qwen/Qwen2-0.5B-Instruct",
                task="chat",
                generate_kwargs={'do_sample': True, 'max_new_tokens': 128},
            )
    """

    pipeline: Any  #: :meta private:
    task: str = DEFAULT_TASK
    model_id: str = DEFAULT_MODEL_ID
    model_revision: Optional[str] = None
    generate_kwargs: Optional[Dict[Any, Any]] = None
    """Keyword arguments passed to the pipeline."""
    batch_size: int = DEFAULT_BATCH_SIZE
    """Batch size to use when passing multiple documents to generate."""

    @classmethod
    def from_model_id(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: Optional[str] = None,
        task: str = DEFAULT_TASK,
        device_map: Optional[str] = None,
        generate_kwargs: Optional[Dict[Any, Any]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs: Any,
    ) -> ModelScopePipeline:
        """Construct the pipeline object from model_id and task."""
        try:
            from modelscope import pipeline as PipelineBuilder  # type: ignore[import]
        except ImportError:
            raise ValueError(
                "Could not import modelscope python package. "
                "Please install it with `pip install ms-swift 'modelscope[nlp]' -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html`."
            )
        modelscope_pipeline = PipelineBuilder(
            task=task,
            model=model_id,
            model_revision=model_revision,
            device_map="auto" if device_map is None else device_map,
            llm_framework="swfit",
            llm_first=True,
            **kwargs,
        )
        return cls(
            pipeline=modelscope_pipeline,
            task=task,
            model_id=model_id,
            model_revision=model_revision,
            generate_kwargs=generate_kwargs,
            batch_size=batch_size,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "generate_kwargs": self.generate_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "modelscope_pipeline"

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        if self.generate_kwargs is not None:
            gen_cfg = {**self.generate_kwargs, **kwargs}
        else:
            gen_cfg = {**kwargs}

        for stream_output in self.pipeline.stream_generate(prompt, **gen_cfg):
            text = stream_output["text"]
            chunk = GenerationChunk(text=text)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # List to hold all results
        text_generations: List[str] = []
        if self.generate_kwargs is not None:
            gen_cfg = {**self.generate_kwargs, **kwargs}
        else:
            gen_cfg = {**kwargs}

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]

            # Process batch of prompts
            responses = self.pipeline(
                batch_prompts,
                **gen_cfg,
            )
            # Process each response in the batch
            for j, response in enumerate(responses):
                if isinstance(response, list):
                    # if model returns multiple generations, pick the top one
                    response = response[0]
                text = response["text"]
                # Append the processed text to results
                text_generations.append(text)

        return LLMResult(
            generations=[[Generation(text=text)] for text in text_generations]
        )

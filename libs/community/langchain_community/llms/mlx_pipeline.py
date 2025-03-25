from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import ConfigDict

DEFAULT_MODEL_ID = "mlx-community/quantized-gemma-2b"

logger = logging.getLogger(__name__)


class MLXPipeline(LLM):
    """MLX Pipeline API.

    To use, you should have the ``mlx-lm`` python package installed.

    Example using from_model_id:
        .. code-block:: python

            from langchain_community.llms import MLXPipeline
            pipe = MLXPipeline.from_model_id(
                model_id="mlx-community/quantized-gemma-2b",
                pipeline_kwargs={"max_tokens": 10, "temp": 0.7},
            )
    Example passing model and tokenizer in directly:
        .. code-block:: python

            from langchain_community.llms import MLXPipeline
            from mlx_lm import load
            model_id="mlx-community/quantized-gemma-2b"
            model, tokenizer = load(model_id)
            pipe = MLXPipeline(model=model, tokenizer=tokenizer)
    """

    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model: Any = None  #: :meta private:
    """Model."""
    tokenizer: Any = None  #: :meta private:
    """Tokenizer."""
    tokenizer_config: Optional[dict] = None
    """
        Configuration parameters specifically for the tokenizer.
        Defaults to an empty dictionary.
    """
    adapter_file: Optional[str] = None
    """
        Path to the adapter file. If provided, applies LoRA layers to the model.
        Defaults to None.
    """
    lazy: bool = False
    """
        If False eval the model parameters to make sure they are
        loaded in memory before returning, otherwise they will be loaded
        when needed. Default: ``False``
    """
    pipeline_kwargs: Optional[dict] = None
    """
    Keyword arguments passed to the pipeline. Defaults include:
        - temp (float): Temperature for generation, default is 0.0.
        - max_tokens (int): Maximum tokens to generate, default is 100.
        - verbose (bool): Whether to output verbose logging, default is False.
        - formatter (Optional[Callable]): A callable to format the output.
          Default is None.
        - repetition_penalty (Optional[float]): The penalty factor for
          repeated sequences, default is None.
        - repetition_context_size (Optional[int]): Size of the context
          for applying repetition penalty, default is None.
        - top_p (float): The cumulative probability threshold for
          top-p filtering, default is 1.0.

    """

    model_config = ConfigDict(
        extra="forbid",
    )

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        tokenizer_config: Optional[dict] = None,
        adapter_file: Optional[str] = None,
        lazy: bool = False,
        pipeline_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> MLXPipeline:
        """Construct the pipeline object from model_id and task."""
        try:
            from mlx_lm import load

        except ImportError:
            raise ImportError(
                "Could not import mlx_lm python package. "
                "Please install it with `pip install mlx_lm`."
            )

        tokenizer_config = tokenizer_config or {}
        if adapter_file:
            model, tokenizer = load(
                model_id, tokenizer_config, adapter_path=adapter_file, lazy=lazy
            )
        else:
            model, tokenizer = load(model_id, tokenizer_config, lazy=lazy)

        _pipeline_kwargs = pipeline_kwargs or {}
        return cls(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            tokenizer_config=tokenizer_config,
            adapter_file=adapter_file,
            lazy=lazy,
            pipeline_kwargs=_pipeline_kwargs,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "tokenizer_config": self.tokenizer_config,
            "adapter_file": self.adapter_file,
            "lazy": self.lazy,
            "pipeline_kwargs": self.pipeline_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "mlx_pipeline"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_logits_processors, make_sampler

        except ImportError:
            raise ImportError(
                "Could not import mlx_lm python package. "
                "Please install it with `pip install mlx_lm`."
            )

        pipeline_kwargs = kwargs.get("pipeline_kwargs", self.pipeline_kwargs)

        temp: float = pipeline_kwargs.get("temp", 0.0)
        max_tokens: int = pipeline_kwargs.get("max_tokens", 100)
        verbose: bool = pipeline_kwargs.get("verbose", False)
        formatter: Optional[Callable] = pipeline_kwargs.get("formatter", None)
        repetition_penalty: Optional[float] = pipeline_kwargs.get(
            "repetition_penalty", None
        )
        repetition_context_size: Optional[int] = pipeline_kwargs.get(
            "repetition_context_size", None
        )
        top_p: float = pipeline_kwargs.get("top_p", 1.0)
        min_p: float = pipeline_kwargs.get("min_p", 0.0)
        min_tokens_to_keep: int = pipeline_kwargs.get("min_tokens_to_keep", 1)

        sampler = make_sampler(temp, top_p, min_p, min_tokens_to_keep)
        logits_processors = make_logits_processors(
            None, repetition_penalty, repetition_context_size
        )

        return generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=verbose,
            formatter=formatter,
            sampler=sampler,
            logits_processors=logits_processors,
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        try:
            import mlx.core as mx
            from mlx_lm.sample_utils import make_logits_processors, make_sampler
            from mlx_lm.utils import generate_step

        except ImportError:
            raise ImportError(
                "Could not import mlx_lm python package. "
                "Please install it with `pip install mlx_lm`."
            )

        pipeline_kwargs = kwargs.get("pipeline_kwargs", self.pipeline_kwargs)

        temp: float = pipeline_kwargs.get("temp", 0.0)
        max_new_tokens: int = pipeline_kwargs.get("max_tokens", 100)
        repetition_penalty: Optional[float] = pipeline_kwargs.get(
            "repetition_penalty", None
        )
        repetition_context_size: Optional[int] = pipeline_kwargs.get(
            "repetition_context_size", None
        )
        top_p: float = pipeline_kwargs.get("top_p", 1.0)
        min_p: float = pipeline_kwargs.get("min_p", 0.0)
        min_tokens_to_keep: int = pipeline_kwargs.get("min_tokens_to_keep", 1)

        prompt = self.tokenizer.encode(prompt, return_tensors="np")

        prompt_tokens = mx.array(prompt[0])

        eos_token_id = self.tokenizer.eos_token_id
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        sampler = make_sampler(temp or 0.0, top_p, min_p, min_tokens_to_keep)

        logits_processors = make_logits_processors(
            None, repetition_penalty, repetition_context_size
        )

        for (token, prob), n in zip(
            generate_step(
                prompt=prompt_tokens,
                model=self.model,
                sampler=sampler,
                logits_processors=logits_processors,
            ),
            range(max_new_tokens),
        ):
            # identify text to yield
            text: Optional[str] = None
            detokenizer.add_token(token)
            detokenizer.finalize()
            text = detokenizer.last_segment

            # yield text, if any
            if text:
                chunk = GenerationChunk(text=text)
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)
                yield chunk

            # break if stop sequence found
            if token == eos_token_id or (stop is not None and text in stop):
                break

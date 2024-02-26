from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional, Iterator, Dict, Union, Sequence
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Extra

try:
    import mlx.nn as nn

except ImportError:
    raise ValueError(
        "Could not import mlx python package. "
        "Please install it with `pip install mlx`."
    )


from transformers import PreTrainedTokenizerBase




DEFAULT_MODEL_ID = "mlx-community/quantized-gemma-2b-it"

logger = logging.getLogger(__name__)

class MLXPipeline(LLM):
    """MLX Pipeline API.

    To use, you should have the ``mlx-lm`` python package installed.

    Example using from_model_id:
        .. code-block:: python

            from langchain_community.llms import MLXPipeline
            pipe = MLXPipeline.from_model_id(
                model_id="mlx-community/quantized-gemma-2b-it",
                pipeline_kwargs={"max_new_tokens": 10},
            )
    Example passing model and tokenizer in directly:
        .. code-block:: python

            from langchain_community.llms import MLXPipeline
            from mlx_lm import load
            model_id="mlx-community/quantized-gemma-2b-it"
            model, tokenizer = load(model_id)
            pipe = MLXPipeline(model=model, tokenizer=tokenizer)
    """

    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model: nn.Module
    """Model."""
    tokenizer: PreTrainedTokenizerBase
    """Tokenizer."""
    tokenizer_config: Optional[dict] = None
    """
        Configuration parameters specifically for the tokenizer.
        Defaults to an empty dictionary.
    """
    adapter_file:  Optional[dict] = None
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
    """Keyword arguments passed to the pipeline."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        tokenizer_config: Optional[dict] = None,
        adapter_file: str = None,
        lazy: bool = False,
        pipeline_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> MLXPipeline:
        """Construct the pipeline object from model_id and task."""
        try:
            from mlx_lm import load

        except ImportError:
            raise ValueError(
                "Could not import mlx_lm python package. "
                "Please install it with `pip install mlx_lm`."
            )


        tokenizer_config = tokenizer_config or {}
        model, tokenizer = load(model_id, tokenizer_config, adapter_file, lazy)
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
            "pipeline_kwargs": self.pipeline_kwargs
        }

    @property
    def _llm_type(self) -> str:
        return "mlx_pipeline"

    def _convert_prompt(self, prompt):
        """Convert langchain prompt to dict with keys `role` and `content`."""

        # Create a dictionary to store role and message pairs
        conversation_list = []

        # Process each line
        for line in prompt.split('\n'):
            # Split each line into role and message
            if ': ' in line:
                role, content = line.split(': ', 1)
            else:
                role, content = 'user', line

            role = role.lower()

            # Map roles
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"

            # Create a dictionary for the current line
            line_dict = {'content': content, 'role': role}

            # Append the dictionary to the list
            conversation_list.append(line_dict)

        return conversation_list

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        from mlx_lm import generate

        prompt = self._convert_prompt(prompt)

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="np",
                **kwargs
            )
        else:
            prompt = prompt[0]['content']

        pipeline_kwargs = kwargs.get("pipeline_kwargs", {})

        return generate(
            self.model, self.tokenizer,
            prompt=prompt,
            **pipeline_kwargs
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:

        import mlx.core as mx
        from mlx_lm.utils import generate_step

        try:
            import mlx.core as mx
            from mlx_lm.utils import generate_step

        except ImportError:
            raise ValueError(
                "Could not import mlx_lm python package. "
                "Please install it with `pip install mlx_lm`."
            )

        pipeline_kwargs = kwargs.get("pipeline_kwargs", {})

        temp: float =  pipeline_kwargs.get("temp", 0.0)
        max_new_tokens: int = pipeline_kwargs.get("max_new_tokens", 100)
        repetition_penalty: Optional[float] = pipeline_kwargs.get("repetition_penalty", None)
        repetition_context_size: Optional[float] = pipeline_kwargs.get("repetition_context_size", None)

        prompt = self._convert_prompt(prompt)

        eos_token_id = self.tokenizer.eos_token_id
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                return_tensors="np",
                **kwargs
            )
        else:
            prompt = self.tokenizer.encode(prompt[0]['content'], return_tensors="np")

        prompt_tokens = mx.array(prompt[0])

        for (token, prob), n in zip(
            generate_step(
                prompt_tokens,
                self.model,
                temp,
                repetition_penalty,
                repetition_context_size,
                **kwargs
            ),
            range(max_new_tokens)
        ):

            # identify text to yield
            text: Optional[str] = None
            text = self.tokenizer.decode(token.item())

            # yield text, if any
            if text:
                chunk = GenerationChunk(text=text)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)

            # break if stop sequence found
            if token == eos_token_id:
                break


from typing import Any, Callable, Dict, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.utils import pre_init
from pydantic import Field


class ExLlamaV2(LLM):
    """ExllamaV2 API.

    - working only with GPTQ models for now.
    - Lora models are not supported yet.

    To use, you should have the exllamav2 library installed, and provide the
    path to the Llama model as a named parameter to the constructor.
    Check out:

    Example:
        .. code-block:: python

            from langchain_community.llms import Exllamav2

            llm = Exllamav2(model_path="/path/to/llama/model")

    #TODO:
    - Add loras support
    - Add support for custom settings
    - Add support for custom stop sequences
    """

    client: Any = None
    model_path: str
    exllama_cache: Any = None
    config: Any = None
    generator: Any = None
    tokenizer: Any = None
    # If settings is None, it will be used as the default settings for the model.
    # All other parameters won't be used.
    settings: Any = None

    # Langchain parameters
    logfunc: Callable = print

    stop_sequences: List[str] = Field([])
    """Sequences that immediately will stop the generator."""

    max_new_tokens: int = Field(150)
    """Maximum number of tokens to generate."""

    streaming: bool = Field(True)
    """Whether to stream the results, token by token."""

    verbose: bool = Field(True)
    """Whether to print debug information."""

    # Generator parameters
    disallowed_tokens: Optional[List[int]] = Field(None)
    """List of tokens to disallow during generation."""

    @pre_init
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "Unable to import torch, please install with `pip install torch`."
            ) from e
        # check if cuda is available
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is not available. ExllamaV2 requires CUDA.")
        try:
            from exllamav2 import (
                ExLlamaV2,
                ExLlamaV2Cache,
                ExLlamaV2Config,
                ExLlamaV2Tokenizer,
            )
            from exllamav2.generator import (
                ExLlamaV2BaseGenerator,
                ExLlamaV2StreamingGenerator,
            )
        except ImportError:
            raise ImportError(
                "Could not import exllamav2 library. "
                "Please install the exllamav2 library with (cuda 12.1 is required)"
                "example : "
                "!python -m pip install https://github.com/turboderp/exllamav2/releases/download/v0.0.12/exllamav2-0.0.12+cu121-cp311-cp311-linux_x86_64.whl"
            )

        # Set logging function if verbose or set to empty lambda
        verbose = values["verbose"]
        if not verbose:
            values["logfunc"] = lambda *args, **kwargs: None
        logfunc = values["logfunc"]

        if values["settings"]:
            settings = values["settings"]
            logfunc(settings.__dict__)
        else:
            raise NotImplementedError(
                "settings is required. Custom settings are not supported yet."
            )

        config = ExLlamaV2Config()
        config.model_dir = values["model_path"]
        config.prepare()

        model = ExLlamaV2(config)

        exllama_cache = ExLlamaV2Cache(model, lazy=True)
        model.load_autosplit(exllama_cache)

        tokenizer = ExLlamaV2Tokenizer(config)
        if values["streaming"]:
            generator = ExLlamaV2StreamingGenerator(model, exllama_cache, tokenizer)
        else:
            generator = ExLlamaV2BaseGenerator(model, exllama_cache, tokenizer)

        # Configure the model and generator
        values["stop_sequences"] = [x.strip().lower() for x in values["stop_sequences"]]
        setattr(settings, "stop_sequences", values["stop_sequences"])
        logfunc(f"stop_sequences {values['stop_sequences']}")

        disallowed = values.get("disallowed_tokens")
        if disallowed:
            settings.disallow_tokens(tokenizer, disallowed)

        values["client"] = model
        values["generator"] = generator
        values["config"] = config
        values["tokenizer"] = tokenizer
        values["exllama_cache"] = exllama_cache

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "ExLlamaV2"

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""
        return self.generator.tokenizer.num_tokens(text)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        generator = self.generator

        if self.streaming:
            combined_text_output = ""
            for chunk in self._stream(
                prompt=prompt, stop=stop, run_manager=run_manager, kwargs=kwargs
            ):
                combined_text_output += str(chunk)
            return combined_text_output
        else:
            output = generator.generate_simple(
                prompt=prompt,
                gen_settings=self.settings,
                num_tokens=self.max_new_tokens,
            )
            # subtract subtext from output
            output = output[len(prompt) :]
            return output

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        input_ids = self.tokenizer.encode(prompt)
        self.generator.warmup()
        self.generator.set_stop_conditions([])
        self.generator.begin_stream(input_ids, self.settings)

        generated_tokens = 0

        while True:
            chunk, eos, _ = self.generator.stream()
            generated_tokens += 1

            if run_manager:
                run_manager.on_llm_new_token(
                    token=chunk,
                    verbose=self.verbose,
                )
            yield chunk
            if eos or generated_tokens == self.max_new_tokens:
                break

        return

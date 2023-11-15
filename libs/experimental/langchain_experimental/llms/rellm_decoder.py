"""Experimental implementation of RELLM wrapped LLM."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, cast

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.utils import enforce_stop_tokens

from langchain_experimental.pydantic_v1 import Field, root_validator

if TYPE_CHECKING:
    import rellm
    from regex import Pattern as RegexPattern
else:
    try:
        from regex import Pattern as RegexPattern
    except ImportError:
        pass


def import_rellm() -> rellm:
    """Lazily import rellm."""
    try:
        import rellm
    except ImportError:
        raise ImportError(
            "Could not import rellm python package. "
            "Please install it with `pip install rellm`."
        )
    return rellm


class RELLM(HuggingFacePipeline):
    """RELLM wrapped LLM using HuggingFace Pipeline API."""

    regex: RegexPattern = Field(..., description="The structured format to complete.")
    max_new_tokens: int = Field(
        default=200, description="Maximum number of new tokens to generate."
    )

    # TODO: move away from `root_validator` since it is deprecated in pydantic v2
    #       and causes mypy type-checking failures (hence the `type: ignore`)
    @root_validator  # type: ignore[call-overload]
    def check_rellm_installation(cls, values: dict) -> dict:
        import_rellm()
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        rellm = import_rellm()
        from transformers import Text2TextGenerationPipeline

        pipeline = cast(Text2TextGenerationPipeline, self.pipeline)

        text = rellm.complete_re(
            prompt,
            self.regex,
            tokenizer=pipeline.tokenizer,
            model=pipeline.model,
            max_new_tokens=self.max_new_tokens,
        )
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)
        return text

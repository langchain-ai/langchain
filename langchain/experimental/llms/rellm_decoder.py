"""Experimental implementation of RELLM wrapped LLM."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, cast

from pydantic import Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.utils import enforce_stop_tokens

if TYPE_CHECKING:
    from regex import Pattern as RegexPattern  # type: ignore
else:
    try:
        from regex import Pattern as RegexPattern  # type: ignore
    except ImportError:
        pass


class RELLM(HuggingFacePipeline):
    regex: RegexPattern = Field(..., description="The structured format to complete.")
    max_new_tokens: int = Field(
        default=200, description="Maximum number of new tokens to generate."
    )

    @root_validator
    def check_rellm_installation(cls, values: dict) -> dict:
        try:
            pass
        except ImportError:
            raise ValueError(
                "Could not import rellm python package. "
                "Please install it with `pip install rellm`."
            )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        from rellm import complete_re
        from transformers import Text2TextGenerationPipeline

        pipeline = cast(Text2TextGenerationPipeline, self.pipeline)

        text = complete_re(
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

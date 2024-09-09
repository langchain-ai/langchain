"""Experimental implementation of lm-format-enforcer wrapped LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from langchain.schema import LLMResult
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field

if TYPE_CHECKING:
    import lmformatenforcer


def import_lmformatenforcer() -> lmformatenforcer:
    """Lazily import of the lmformatenforcer package."""
    try:
        import lmformatenforcer
    except ImportError:
        raise ImportError(
            "Could not import lmformatenforcer python package. "
            "Please install it with `pip install lm-format-enforcer`."
        )
    return lmformatenforcer


class LMFormatEnforcer(HuggingFacePipeline):
    """LMFormatEnforcer wrapped LLM using HuggingFace Pipeline API.

    This pipeline is experimental and not yet stable.
    """

    json_schema: Optional[dict] = Field(
        description="The JSON Schema to complete.", default=None
    )
    regex: Optional[str] = Field(
        description="The regular expression to complete.", default=None
    )

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        lmformatenforcer = import_lmformatenforcer()
        import lmformatenforcer.integrations.transformers as hf_integration

        # We integrate lmformatenforcer by adding a prefix_allowed_tokens_fn.
        # It has to be done on each call, because the prefix function is stateful.
        if "prefix_allowed_tokens_fn" in self.pipeline._forward_params:
            raise ValueError(
                "prefix_allowed_tokens_fn param is forbidden with LMFormatEnforcer."
            )

        has_json_schema = self.json_schema is not None
        has_regex = self.regex is not None
        if has_json_schema == has_regex:
            raise ValueError(
                "You must specify exactly one of json_schema or a regex, but not both."
            )

        if has_json_schema:
            parser = lmformatenforcer.JsonSchemaParser(self.json_schema)
        else:
            parser = lmformatenforcer.RegexParser(self.regex)

        prefix_function = hf_integration.build_transformers_prefix_allowed_tokens_fn(
            self.pipeline.tokenizer, parser
        )
        self.pipeline._forward_params["prefix_allowed_tokens_fn"] = prefix_function

        result = super()._generate(
            prompts,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

        del self.pipeline._forward_params["prefix_allowed_tokens_fn"]
        return result

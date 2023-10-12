"""Experimental implementation of lm-format-enforcer wrapped LLM."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, cast

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers.pipelines import Text2TextGenerationPipeline

from langchain_experimental.pydantic_v1 import Field, root_validator

if TYPE_CHECKING:
    import lmformatenforcer


def import_lmformatenforcer() -> lmformatenforcer:
    """Lazily import lmformatenforcer."""
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

    @root_validator
    def check_lmformatenforcer_installation(cls, values: dict) -> dict:
        import_lmformatenforcer()
        return values

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
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

        lmformatenforcer = import_lmformatenforcer()
        if has_json_schema:
            parser = lmformatenforcer.JsonSchemaParser(self.json_schema)
        else:
            parser = lmformatenforcer.RegexParser(self.regex)

        pipeline = cast(Text2TextGenerationPipeline, self.pipeline)
        prefix_function = lmformatenforcer.build_transformers_prefix_allowed_tokens_fn(
            pipeline.tokenizer, parser
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

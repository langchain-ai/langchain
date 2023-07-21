"""Experimental implementation of jsonformer wrapped LLM."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, List, Optional, cast

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from pydantic import Field, root_validator

if TYPE_CHECKING:
    import jsonformer


def import_jsonformer() -> jsonformer:
    """Lazily import jsonformer."""
    try:
        import jsonformer
    except ImportError:
        raise ValueError(
            "Could not import jsonformer python package. "
            "Please install it with `pip install jsonformer`."
        )
    return jsonformer


class JsonFormer(HuggingFacePipeline):
    json_schema: dict = Field(..., description="The JSON Schema to complete.")
    max_new_tokens: int = Field(
        default=200, description="Maximum number of new tokens to generate."
    )
    debug: bool = Field(default=False, description="Debug mode.")

    @root_validator
    def check_jsonformer_installation(cls, values: dict) -> dict:
        import_jsonformer()
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        jsonformer = import_jsonformer()
        from transformers import Text2TextGenerationPipeline

        pipeline = cast(Text2TextGenerationPipeline, self.pipeline)

        model = jsonformer.Jsonformer(
            model=pipeline.model,
            tokenizer=pipeline.tokenizer,
            json_schema=self.json_schema,
            prompt=prompt,
            max_number_tokens=self.max_new_tokens,
            debug=self.debug,
        )
        text = model()
        return json.dumps(text)

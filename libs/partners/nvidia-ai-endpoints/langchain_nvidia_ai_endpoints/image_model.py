"""Embeddings Components Derived from NVEModel/Embeddings"""
import base64
from io import BytesIO
from PIL import Image

from typing import Any, List, Literal, Optional

from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import Field

from langchain_nvidia_ai_endpoints._common import _NVIDIAClient
from langchain_nvidia_ai_endpoints.callbacks import usage_callback_var

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableLambda

from langchain_nvidia_ai_endpoints import _common as nvidia_ai_endpoints
from langchain_nvidia_ai_endpoints._statics import MODEL_SPECS, Model


class ImageNVIDIA(_NVIDIAClient, LLM):
    """NVIDIA's AI Foundation Retriever Question-Answering Asymmetric Model."""

    negative_prompt: Optional[float] = Field(description="Sampling temperature in [0, 1]")
    sampler: Optional[float] = Field(description="Sampling strategy for process")
    guidance_scale: Optional[float] = Field(description="The scale of guidance")
    seed: Optional[int] = Field(description="The seed for deterministic results")

    @property
    def _llm_type(self) -> str:
        """Return type of NVIDIA AI Foundation Model Interface."""
        return "nvidia-image-model"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the Image Gen Model on the given prompt and input."""
        response = self.client.get_req(
            model_name=self.model,
            payload={
                "prompt": prompt,
                "negative_prompt": kwargs.get("negative_prompt", self.negative_prompt),
                "sampler": kwargs.get("sampler", self.sampler),
                "guidance_scale": kwargs.get("guidance_scale", self.guidance_scale),
                "seed": kwargs.get("seed", self.seed),
            },
        )
        response.raise_for_status()
        result = response.json()
        base64_str = result.get("b64_json")
        # output = Image.open(BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
        return base64_str


def ImageParser(**kwargs: Any) -> RunnableLambda[str, Image.Image]:
    return RunnableLambda(
        lambda x: Image.open(BytesIO(base64.decodebytes(bytes(x, "utf-8")))),
        **kwargs,
    )

"""HuggingFace Image Generation Wrapper."""
from __future__ import annotations

import io
from typing import Optional

from PIL import Image
from pydantic import BaseModel, Field, root_validator
from requests import Response
from tenacity import retry, retry_if_result, stop_after_attempt, wait_fixed

from langchain.requests import Requests
from langchain.tools.base import BaseTool


def is_503_error(response: Response) -> bool:
    return response.status_code == 503


class RunArgsSchema(BaseModel):
    """Schema for the RunArgs."""

    prompt: str = Field(..., description="Prompt to generate an image.")
    out_path: str = Field(..., description="Path to write the generated image to.")


DEFAULT_INFERENCE_URL = "https://api-inference.huggingface.co/models/"


class HuggingFaceImageGenerationTool(BaseTool):
    """Image Generation Wrapper."""

    name = "huggingface_image_generation"
    description = (
        "Generate an image using a valid image generation model from HuggingFace's API."
    )
    requests_wrapper: Requests
    model_id: str
    """The id of the model to use, such as 'CompVis/stable-diffusion-v1-4'."""
    """Requests wrapper to use containing the authorization headers."""
    url_base: str = DEFAULT_INFERENCE_URL

    @root_validator
    def _validate_authorization_present(cls, values: dict) -> dict:
        requests: Requests = values["requests_wrapper"]
        headers = requests.headers or {}
        if headers.get("Authorization") is None:
            raise ValueError(
                "Error: Authorization token required for the requests wrapper of"
                " the HuggingFaceImageGenerationTool."
            )
        return values

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(20),
        retry=retry_if_result(is_503_error),
    )
    def _request_huggingface_image(
        self,
        prompt: str,
    ) -> Response:
        """Generate an image using Huggingface's API."""
        api_url = self.url_base + self.model_id
        response = self.requests_wrapper.post(
            api_url,
            data={
                "inputs": prompt,
            },
        )
        return response

    def _run(self, prompt: str, out_path: str) -> str:
        """Generate an image using Stable Diffusion using HuggingFace's API."""
        response = self._request_huggingface_image(prompt=prompt)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            image.save(out_path)

            return f"Saved to disk: {out_path}"
        else:
            return f"Failed to generate image. Error: {str(response.content)}"

    async def _arun(self, prompt: str, out_path: str, model_id: str) -> str:
        raise NotImplementedError

    @classmethod
    def from_api_key(
        cls, huggingface_api_key: str, model_id: str, url_base: Optional[str] = None
    ) -> HuggingFaceImageGenerationTool:
        """Create a HuggingFaceImageGenerationTool from an API key."""
        requests = Requests(headers={"Authorization": f"Bearer {huggingface_api_key}"})
        url_base = url_base or DEFAULT_INFERENCE_URL
        return cls(requests_wrapper=requests, model_id=model_id, url_base=url_base)

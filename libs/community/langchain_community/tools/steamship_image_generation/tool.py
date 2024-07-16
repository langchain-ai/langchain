"""This tool allows agents to generate images using Steamship.

Steamship offers access to different third party image generation APIs
using a single API key.

Today the following models are supported:
- Dall-E
- Stable Diffusion

To use this tool, you must first set as environment variables:
    STEAMSHIP_API_KEY
```
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_community.tools import BaseTool
from langchain_community.tools.steamship_image_generation.utils import make_image_public

if TYPE_CHECKING:
    from steamship import Steamship


class ModelName(str, Enum):
    """Supported Image Models for generation."""

    DALL_E = "dall-e"
    STABLE_DIFFUSION = "stable-diffusion"


SUPPORTED_IMAGE_SIZES = {
    ModelName.DALL_E: ("256x256", "512x512", "1024x1024"),
    ModelName.STABLE_DIFFUSION: ("512x512", "768x768"),
}


class SteamshipImageGenerationTool(BaseTool):
    """Tool used to generate images from a text-prompt."""

    model_name: ModelName
    size: Optional[str] = "512x512"
    steamship: Steamship
    return_urls: Optional[bool] = False

    name: str = "generate_image"
    description: str = (
        "Useful for when you need to generate an image."
        "Input: A detailed text-2-image prompt describing an image"
        "Output: the UUID of a generated image"
    )

    @root_validator(pre=True)
    def validate_size(cls, values: Dict) -> Dict:
        if "size" in values:
            size = values["size"]
            model_name = values["model_name"]
            if size not in SUPPORTED_IMAGE_SIZES[model_name]:
                raise RuntimeError(f"size {size} is not supported by {model_name}")

        return values

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        steamship_api_key = get_from_dict_or_env(
            values, "steamship_api_key", "STEAMSHIP_API_KEY"
        )

        try:
            from steamship import Steamship
        except ImportError:
            raise ImportError(
                "steamship is not installed. "
                "Please install it with `pip install steamship`"
            )

        steamship = Steamship(
            api_key=steamship_api_key,
        )
        values["steamship"] = steamship
        if "steamship_api_key" in values:
            del values["steamship_api_key"]

        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        image_generator = self.steamship.use_plugin(
            plugin_handle=self.model_name.value, config={"n": 1, "size": self.size}
        )

        task = image_generator.generate(text=query, append_output_to_file=True)
        task.wait()
        blocks = task.output.blocks
        if len(blocks) > 0:
            if self.return_urls:
                return make_image_public(self.steamship, blocks[0])
            else:
                return blocks[0].id

        raise RuntimeError(f"[{self.name}] Tool unable to generate image!")

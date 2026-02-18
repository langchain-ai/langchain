"""Utility wrapper for ModelsLab API."""
from __future__ import annotations

import asyncio
import base64
import time
from typing import Any, Dict, List, Optional, Union

import requests
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from langchain_core.pydantic_v1 import BaseSettings, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


class ModelsLabAPIWrapper(BaseSettings):
    """Wrapper for ModelsLab API.
    
    To use, you should have the ``MODELSLAB_API_KEY`` environment variable set
    with your API key, or pass it as a named parameter to the constructor.
    
    Example:
        .. code-block:: python
        
            from langchain_community.utilities import ModelsLabAPIWrapper
            modelslab = ModelsLabAPIWrapper(modelslab_api_key="your-api-key")
            result = modelslab.run("A beautiful mountain landscape")
    """

    modelslab_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """ModelsLab API key. Get from https://modelslab.com/dashboard/api-keys"""
    
    model: str = "flux"
    """Model to use for image generation. Options: flux, stable-diffusion-xl, playground-v2.5"""
    
    size: str = "1024x1024"
    """Image size. Options: 512x512, 768x768, 1024x1024, 1024x768, 768x1024"""
    
    samples: int = Field(default=1, le=4)
    """Number of images to generate (1-4)"""
    
    num_inference_steps: int = Field(default=30, ge=1, le=50)
    """Number of inference steps for generation quality"""
    
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    """How closely to follow the prompt"""
    
    safety_checker: bool = True
    """Whether to enable content safety filtering"""
    
    enhance_prompt: bool = False
    """Whether to automatically enhance the prompt"""
    
    negative_prompt: Optional[str] = None
    """Negative prompt describing what to avoid in the image"""
    
    seed: Optional[int] = None
    """Seed for reproducible results"""
    
    base_url: str = "https://modelslab.com/api/v6"
    """ModelsLab API base URL"""
    
    model_config = ConfigDict(extra="forbid")

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        modelslab_api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "modelslab_api_key", "MODELSLAB_API_KEY")
        )
        values["modelslab_api_key"] = modelslab_api_key
        return values

    def _get_request_payload(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Build request payload for ModelsLab API."""
        payload = {
            "key": self.modelslab_api_key.get_secret_value() if self.modelslab_api_key else "",
            "model_id": kwargs.get("model_id", self.model),
            "prompt": prompt,
            "width": int(self.size.split("x")[0]),
            "height": int(self.size.split("x")[1]),
            "samples": kwargs.get("samples", self.samples),
            "num_inference_steps": kwargs.get("num_inference_steps", self.num_inference_steps),
            "guidance_scale": kwargs.get("guidance_scale", self.guidance_scale),
            "safety_checker": "yes" if kwargs.get("safety_checker", self.safety_checker) else "no",
            "enhance_prompt": kwargs.get("enhance_prompt", self.enhance_prompt),
        }
        
        # Optional parameters
        if self.negative_prompt or kwargs.get("negative_prompt"):
            payload["negative_prompt"] = kwargs.get("negative_prompt", self.negative_prompt)
            
        if self.seed or kwargs.get("seed"):
            payload["seed"] = kwargs.get("seed", self.seed)
            
        # Handle image-to-image parameters
        if "init_image" in kwargs:
            payload["init_image"] = kwargs["init_image"]
            payload["strength"] = kwargs.get("strength", 0.7)
            
        # Handle inpainting parameters  
        if "mask_image" in kwargs:
            payload["mask_image"] = kwargs["mask_image"]
            
        return payload

    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to ModelsLab API."""
        url = f"{self.base_url}/{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        return response.json()

    def _poll_for_result(self, request_id: str, max_wait: int = 300) -> Dict[str, Any]:
        """Poll for async operation result."""
        poll_url = f"{self.base_url}/image_editing/fetch/{request_id}"
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = requests.get(poll_url)
            response.raise_for_status()
            
            result = response.json()
            status = result.get("status")
            
            if status == "success":
                return result
            elif status == "error":
                raise RuntimeError(f"ModelsLab generation failed: {result.get('message', 'Unknown error')}")
            elif status == "processing":
                time.sleep(2)  # Wait before next poll
                continue
            else:
                raise RuntimeError(f"Unknown status: {status}")
                
        raise TimeoutError("ModelsLab generation timed out")

    def _process_images(self, result: Dict[str, Any]) -> List[str]:
        """Process image results and return URLs or base64 data."""
        output = result.get("output", [])
        if not output:
            raise ValueError("No images generated")
            
        processed_images = []
        for image in output:
            if isinstance(image, str):
                if image.startswith("http"):
                    # Image URL - return as is
                    processed_images.append(image)
                else:
                    # Base64 encoded image
                    processed_images.append(f"data:image/png;base64,{image}")
            else:
                processed_images.append(str(image))
                
        return processed_images

    def run(self, prompt: str, **kwargs: Any) -> str:
        """Generate an image from text prompt.
        
        Args:
            prompt: Text description of the image to generate
            **kwargs: Additional parameters to override defaults
            
        Returns:
            String containing image URL or base64 data
            
        Raises:
            ValueError: If generation fails or no images returned
            RuntimeError: If API returns error
            TimeoutError: If async operation times out
        """
        try:
            # Build request payload
            payload = self._get_request_payload(prompt, **kwargs)
            
            # Determine endpoint based on model type
            model_id = payload.get("model_id", self.model)
            if "init_image" in payload and "mask_image" in payload:
                endpoint = "image_editing/inpaint"
            elif "init_image" in payload:
                endpoint = "images/img2img"
            else:
                endpoint = "images/text2img"
                
            # Make API request
            result = self._make_request(endpoint, payload)
            
            # Handle async operations
            if result.get("status") == "processing":
                request_id = result.get("id")
                if not request_id:
                    raise ValueError("No request ID returned for async operation")
                result = self._poll_for_result(request_id)
            
            # Process and return images
            images = self._process_images(result)
            
            # Return first image URL/data
            if images:
                return images[0]
            else:
                raise ValueError("No images generated successfully")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"ModelsLab API request failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"ModelsLab image generation failed: {str(e)}")

    def run_multiple(self, prompt: str, **kwargs: Any) -> List[str]:
        """Generate multiple images from text prompt.
        
        Args:
            prompt: Text description of the image to generate
            **kwargs: Additional parameters (samples will be used for count)
            
        Returns:
            List of image URLs or base64 data strings
        """
        try:
            # Force multiple samples
            kwargs["samples"] = min(kwargs.get("samples", self.samples), 4)
            
            # Build request payload
            payload = self._get_request_payload(prompt, **kwargs)
            
            # Use text2img endpoint for multiple images
            endpoint = "images/text2img"
            
            # Make API request
            result = self._make_request(endpoint, payload)
            
            # Handle async operations
            if result.get("status") == "processing":
                request_id = result.get("id")
                if not request_id:
                    raise ValueError("No request ID returned for async operation")
                result = self._poll_for_result(request_id)
                
            # Process and return all images
            return self._process_images(result)
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"ModelsLab API request failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"ModelsLab image generation failed: {str(e)}")

    def run_img2img(self, prompt: str, init_image: str, strength: float = 0.7, **kwargs: Any) -> str:
        """Transform an existing image using AI.
        
        Args:
            prompt: Text description of desired transformation
            init_image: Base64 encoded input image or image URL
            strength: How much to change the original image (0.0-1.0)
            **kwargs: Additional parameters
            
        Returns:
            String containing transformed image URL or base64 data
        """
        kwargs.update({
            "init_image": init_image,
            "strength": strength
        })
        return self.run(prompt, **kwargs)

    def run_inpaint(self, prompt: str, init_image: str, mask_image: str, **kwargs: Any) -> str:
        """Edit specific regions of an image using a mask.
        
        Args:
            prompt: Text description of what to paint in masked areas
            init_image: Base64 encoded input image
            mask_image: Base64 encoded mask (white areas will be inpainted)
            **kwargs: Additional parameters
            
        Returns:
            String containing edited image URL or base64 data
        """
        kwargs.update({
            "init_image": init_image,
            "mask_image": mask_image
        })
        return self.run(prompt, **kwargs)
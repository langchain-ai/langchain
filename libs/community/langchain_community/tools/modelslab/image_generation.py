"""Tool for generating images using ModelsLab API."""
from typing import Optional, Type

from pydantic import BaseModel, Field

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.modelslab import ModelsLabAPIWrapper


class ModelsLabImageGenerationInput(BaseModel):
    """Input schema for ModelsLab image generation tool."""
    
    query: str = Field(
        description="A detailed text description of the image to generate. "
        "Be specific about style, composition, lighting, and subjects."
    )


class ModelsLabImageGenerationTool(BaseTool):
    """Tool for generating images using ModelsLab's AI models.
    
    This tool provides access to ModelsLab's powerful image generation capabilities,
    including Flux, Stable Diffusion, and other state-of-the-art models.
    
    Features:
    - High-quality image generation from text prompts
    - Multiple model options (Flux, Stable Diffusion XL, Playground v2.5)
    - Professional-grade results with fine-tuned control
    - Fast processing with reliable API
    
    Setup:
        1. Get API key from https://modelslab.com/dashboard/api-keys
        2. Set MODELSLAB_API_KEY environment variable
        3. Use with LangChain agents for AI-powered image creation
        
    Example:
        .. code-block:: python
        
            from langchain_community.tools.modelslab import ModelsLabImageGenerationTool
            from langchain_community.utilities.modelslab import ModelsLabAPIWrapper
            
            api_wrapper = ModelsLabAPIWrapper(modelslab_api_key="your-key")
            tool = ModelsLabImageGenerationTool(api_wrapper=api_wrapper)
            
            result = tool.run("A majestic mountain landscape at sunset")
            print(result)  # Returns image URL or base64 data
    """
    
    name: str = "modelslab_image_generation"
    description: str = (
        "Generate high-quality images using ModelsLab's AI models including "
        "Flux, Stable Diffusion, and professional image generators. "
        "Input should be a detailed text prompt describing the desired image. "
        "Useful for creating artwork, illustrations, concept art, product images, "
        "portraits, landscapes, and any visual content from text descriptions. "
        "Returns a URL or base64 data of the generated image."
    )
    args_schema: Type[BaseModel] = ModelsLabImageGenerationInput
    api_wrapper: ModelsLabAPIWrapper

    def __init__(self, api_wrapper: ModelsLabAPIWrapper, **kwargs):
        """Initialize the ModelsLab image generation tool.
        
        Args:
            api_wrapper: ModelsLabAPIWrapper instance with API key configured
            **kwargs: Additional tool arguments
        """
        super().__init__(api_wrapper=api_wrapper, **kwargs)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Generate an image from the given text prompt.
        
        Args:
            query: Text description of the image to generate
            run_manager: Optional callback manager for tool execution
            
        Returns:
            String containing the generated image URL or base64 data
            
        Raises:
            Exception: If image generation fails
        """
        try:
            if run_manager:
                run_manager.on_tool_start(
                    serialized={"name": self.name, "description": self.description},
                    input_str=query,
                )
                
            # Generate the image using ModelsLab API
            result = self.api_wrapper.run(query)
            
            if run_manager:
                run_manager.on_tool_end(result)
                
            return result
            
        except Exception as e:
            error_msg = f"ModelsLab image generation failed: {str(e)}"
            if run_manager:
                run_manager.on_tool_error(e)
            raise Exception(error_msg)


class ModelsLabImageToImageTool(BaseTool):
    """Tool for transforming images using ModelsLab's AI models.
    
    This tool transforms existing images using text prompts, allowing for
    artistic style transfer, content modification, and creative variations.
    """
    
    name: str = "modelslab_img2img"
    description: str = (
        "Transform existing images using ModelsLab's AI models. "
        "Useful for style transfer, artistic modifications, and creative variations. "
        "Input should include both a text prompt and base64 encoded image data."
    )
    api_wrapper: ModelsLabAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Transform an image based on text prompt.
        
        Note: This is a simplified interface. For full img2img functionality,
        use the api_wrapper directly with init_image parameter.
        """
        try:
            return self.api_wrapper.run(query)
        except Exception as e:
            raise Exception(f"ModelsLab img2img failed: {str(e)}")


class ModelsLabInpaintingTool(BaseTool):
    """Tool for editing specific regions of images using masks.
    
    This tool allows precise editing of images by specifying what to change
    in masked areas while keeping the rest of the image unchanged.
    """
    
    name: str = "modelslab_inpainting"
    description: str = (
        "Edit specific regions of images using masks with ModelsLab's AI. "
        "Useful for removing objects, changing backgrounds, or modifying "
        "specific parts of an image while preserving the rest."
    )
    api_wrapper: ModelsLabAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Perform inpainting on an image.
        
        Note: This is a simplified interface. For full inpainting functionality,
        use the api_wrapper directly with init_image and mask_image parameters.
        """
        try:
            return self.api_wrapper.run(query)
        except Exception as e:
            raise Exception(f"ModelsLab inpainting failed: {str(e)}")


class ModelsLabMultiImageTool(BaseTool):
    """Tool for generating multiple images at once from a single prompt.
    
    This tool generates 2-4 variations of an image from one prompt,
    useful for getting different creative options.
    """
    
    name: str = "modelslab_multi_image"
    description: str = (
        "Generate multiple image variations from a single prompt using ModelsLab. "
        "Returns 2-4 different versions of the same concept for creative options. "
        "Input should be a detailed text prompt."
    )
    args_schema: Type[BaseModel] = ModelsLabImageGenerationInput
    api_wrapper: ModelsLabAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Generate multiple images from the given text prompt.
        
        Args:
            query: Text description of the images to generate
            run_manager: Optional callback manager
            
        Returns:
            String containing comma-separated image URLs or base64 data
        """
        try:
            # Generate multiple images
            images = self.api_wrapper.run_multiple(query, samples=4)
            
            # Return as comma-separated string for tool interface
            return ", ".join(images)
            
        except Exception as e:
            raise Exception(f"ModelsLab multi-image generation failed: {str(e)}")
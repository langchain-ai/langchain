from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import root_validator

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.azure_cognitive_services.utils import detect_file_src_type
from langchain.tools.base import BaseTool
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class AzureCogsImageAnalysisTool(BaseTool):
    """Tool that queries the Azure Cognitive Services Image Analysis API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/image-analysis-client-library-40
    """

    azure_cogs_key: str = ""  #: :meta private:
    azure_cogs_endpoint: str = ""  #: :meta private:
    vision_service: Any  #: :meta private:
    analysis_options: Any  #: :meta private:

    name: str = "azure_cognitive_services_image_analysis"
    description: str = (
        "A wrapper around Azure Cognitive Services Image Analysis. "
        "Useful for when you need to analyze images. "
        "Input should be a url to an image."
    )

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        azure_cogs_key = get_from_dict_or_env(
            values, "azure_cogs_key", "AZURE_COGS_KEY"
        )

        azure_cogs_endpoint = get_from_dict_or_env(
            values, "azure_cogs_endpoint", "AZURE_COGS_ENDPOINT"
        )

        try:
            import azure.ai.vision as sdk

            values["vision_service"] = sdk.VisionServiceOptions(
                endpoint=azure_cogs_endpoint, key=azure_cogs_key
            )

            values["analysis_options"] = sdk.ImageAnalysisOptions()
            values["analysis_options"].features = (
                sdk.ImageAnalysisFeature.CAPTION
                | sdk.ImageAnalysisFeature.OBJECTS
                | sdk.ImageAnalysisFeature.TAGS
                | sdk.ImageAnalysisFeature.TEXT
            )
        except ImportError:
            raise ImportError(
                "azure-ai-vision is not installed. "
                "Run `pip install azure-ai-vision` to install."
            )

        return values

    def _image_analysis(self, image_path: str) -> Dict:
        try:
            import azure.ai.vision as sdk
        except ImportError:
            pass

        image_src_type = detect_file_src_type(image_path)
        if image_src_type == "local":
            vision_source = sdk.VisionSource(filename=image_path)
        elif image_src_type == "remote":
            vision_source = sdk.VisionSource(url=image_path)
        else:
            raise ValueError(f"Invalid image path: {image_path}")

        image_analyzer = sdk.ImageAnalyzer(
            self.vision_service, vision_source, self.analysis_options
        )
        result = image_analyzer.analyze()

        res_dict = {}
        if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:
            if result.caption is not None:
                res_dict["caption"] = result.caption.content

            if result.objects is not None:
                res_dict["objects"] = [obj.name for obj in result.objects]

            if result.tags is not None:
                res_dict["tags"] = [tag.name for tag in result.tags]

            if result.text is not None:
                res_dict["text"] = [line.content for line in result.text.lines]

        else:
            error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
            raise RuntimeError(
                f"Image analysis failed.\n"
                f"Reason: {error_details.reason}\n"
                f"Details: {error_details.message}"
            )

        return res_dict

    def _format_image_analysis_result(self, image_analysis_result: Dict) -> str:
        formatted_result = []
        if "caption" in image_analysis_result:
            formatted_result.append("Caption: " + image_analysis_result["caption"])

        if (
            "objects" in image_analysis_result
            and len(image_analysis_result["objects"]) > 0
        ):
            formatted_result.append(
                "Objects: " + ", ".join(image_analysis_result["objects"])
            )

        if "tags" in image_analysis_result and len(image_analysis_result["tags"]) > 0:
            formatted_result.append("Tags: " + ", ".join(image_analysis_result["tags"]))

        if "text" in image_analysis_result and len(image_analysis_result["text"]) > 0:
            formatted_result.append("Text: " + ", ".join(image_analysis_result["text"]))

        return "\n".join(formatted_result)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            image_analysis_result = self._image_analysis(query)
            if not image_analysis_result:
                return "No good image analysis result was found"

            return self._format_image_analysis_result(image_analysis_result)
        except Exception as e:
            raise RuntimeError(f"Error while running AzureCogsImageAnalysisTool: {e}")

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator

from langchain_community.tools.azure_ai_services.utils import (
    detect_file_src_type,
)

logger = logging.getLogger(__name__)


class AzureAiServicesImageAnalysisTool(BaseTool):  # type: ignore[override]
    """Tool that queries the Azure AI Services Image Analysis API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/azure/ai-services/computer-vision/quickstarts-sdk/image-analysis-client-library-40

    Attributes:
    azure_ai_services_key (Optional[str]): The API key for Azure AI Services.
    azure_ai_services_endpoint (Optional[str]): The endpoint URL for Azure AI Services.
    visual_features Any: The visual features to analyze in the image, can be set as
        either strings or azure.ai.vision.imageanalysis.models.VisualFeatures.
        (e.g. 'TAGS', VisualFeatures.CAPTION).
    image_analysis_client (Any): The client for interacting
        with Azure AI Services Image Analysis.
    name (str): The name of the tool.
    description (str): A description of the tool,
        including its purpose and expected input.
    """

    azure_ai_services_key: Optional[str] = None  #: :meta private:
    azure_ai_services_endpoint: Optional[str] = None  #: :meta private:
    visual_features: Any = None
    image_analysis_client: Any = None  #: :meta private:

    name: str = "azure_ai_services_image_analysis"
    description: str = (
        "A wrapper around Azure AI Services Image Analysis. "
        "Useful for when you need to analyze images. "
        "Input must be a url string or path string to an image."
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        azure_ai_services_key = get_from_dict_or_env(
            values, "azure_ai_services_key", "AZURE_AI_SERVICES_KEY"
        )

        azure_ai_services_endpoint = get_from_dict_or_env(
            values, "azure_ai_services_endpoint", "AZURE_AI_SERVICES_ENDPOINT"
        )

        """Validate that azure-ai-vision-imageanalysis is installed."""
        try:
            from azure.ai.vision.imageanalysis import ImageAnalysisClient
            from azure.ai.vision.imageanalysis.models import VisualFeatures
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError(
                "azure-ai-vision-imageanalysis is not installed. "
                "Run `pip install azure-ai-vision-imageanalysis` to install. "
            )

        """Validate Azure AI Vision Image Analysis client can be initialized."""
        try:
            values["image_analysis_client"] = ImageAnalysisClient(
                endpoint=azure_ai_services_endpoint,
                credential=AzureKeyCredential(azure_ai_services_key),
            )
        except Exception as e:
            raise RuntimeError(
                f"Initialization of Azure AI Vision Image Analysis client failed: {e}"
            )

        visual_features = values.get(
            "visual_features",
            [
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.CAPTION,
                VisualFeatures.READ,
            ],
        )
        values["visual_features"] = visual_features
        return values

    def _image_analysis(self, image_path: str) -> Dict:
        try:
            from azure.ai.vision.imageanalysis import ImageAnalysisClient
        except ImportError:
            pass

        self.image_analysis_client: ImageAnalysisClient

        image_src_type = detect_file_src_type(image_path)
        if image_src_type == "local":
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            result = self.image_analysis_client.analyze(
                image_data=image_data,
                visual_features=self.visual_features,
            )
        elif image_src_type == "remote":
            result = self.image_analysis_client.analyze_from_url(
                image_url=image_path,
                visual_features=self.visual_features,
            )
        else:
            raise ValueError(f"Invalid image path: {image_path}")

        res_dict = {}
        if result:
            if result.caption is not None:
                res_dict["caption"] = result.caption.text

            if result.objects is not None:
                res_dict["objects"] = [obj.tags[0].name for obj in result.objects.list]

            if result.tags is not None:
                res_dict["tags"] = [tag.name for tag in result.tags.list]

            if result.read is not None and len(result.read.blocks) > 0:
                res_dict["text"] = [line.text for line in result.read.blocks[0].lines]

            if result.dense_captions is not None and len(result.dense_captions) > 0:
                res_dict["dense_captions"] = [
                    str(dc) for dc in result.dense_captions.list
                ]

            if result.smart_crops is not None and len(result.smart_crops) > 0:
                res_dict["smart_crops"] = [str(sc) for sc in result.smart_crops.list]

            if result.people is not None and len(result.people) > 0:
                res_dict["people"] = [str(p) for p in result.people.list]

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

        if "dense_captions" in image_analysis_result:
            formatted_result.append(
                "Dense Captions: " + ", ".join(image_analysis_result["dense_captions"])
            )

        if "smart_crops" in image_analysis_result:
            formatted_result.append(
                "Smart Crops: " + ", ".join(image_analysis_result["smart_crops"])
            )

        if "people" in image_analysis_result:
            formatted_result.append(
                "People: " + ", ".join(image_analysis_result["people"])
            )

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
            raise RuntimeError(f"Error while running AzureAiImageAnalysisTool: {e}")

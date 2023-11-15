from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import root_validator
from langchain.tools.azure_cognitive_services.utils import detect_file_src_type
from langchain.tools.base import BaseTool
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class AzureCogsTextAnalyticsHealthTool(BaseTool):
    """Tool that queries the Azure Cognitive Services Text Analytics for Health API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/ai-services/language-service/text-analytics-for-health/quickstart?tabs=windows&pivots=programming-language-python
    """

    azure_cogs_key: str = ""  #: :meta private:
    azure_cogs_endpoint: str = ""  #: :meta private:
    # vision_service: Any  #: :meta private:
    # analysis_options: Any  #: :meta private:

    name: str = "azure_cognitive_services_text_analyics_health"
    description: str = (
        "A wrapper around Azure Cognitive Services Text Analytics for Health. "
        "Useful for when you need to analyze text in healthcare data. "
        "Input should be text."
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
            import azure.ai.textanalytics as sdk

            values["text_analytics_client"] = sdk.TextAnalyticsClient(
                endpoint=azure_cogs_endpoint, key=azure_cogs_key
            )

            # values["analysis_options"] = sdk.ImageAnalysisOptions()
            # values["analysis_options"].features = (
            #     sdk.ImageAnalysisFeature.CAPTION
            #     | sdk.ImageAnalysisFeature.OBJECTS
            #     | sdk.ImageAnalysisFeature.TAGS
            #     | sdk.ImageAnalysisFeature.TEXT
            # )
        except ImportError:
            raise ImportError(
                "azure-ai-textanalytics is not installed. "
                "Run `pip install azure-ai-textanalytics` to install."
            )

        return values

    def _text_analysis(self, text: str) -> Dict:
        poller = self.text_analytics_client.begin_analyze_healthcare_entities([{"id": "1", "language": "en", "text": text}])

        result = poller.result()
        res_dict = {}

        doc = result.next()

        if doc.content is not None:
            res_dict["entities"] = [ f"{x.text} ({x.category}) found at offset {str(x.offset)}" for x in result.entities ]

        return res_dict

    def _format_text_analysis_result(self, text_analysis_result: Dict) -> str:
        formatted_result = []
        if "entities" in text_analysis_result:
            formatted_result.append(
                f"Entities: {', '.join(text_analysis_result['entities'])}".replace("\n", " ")
            )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            text_analysis_result = self._text_analysis(query)

            return self._format_text_analysis_result(text_analysis_result)
        except Exception as e:
            raise RuntimeError(f"Error while running AzureCogsTextAnalyticsHealthTool: {e}")

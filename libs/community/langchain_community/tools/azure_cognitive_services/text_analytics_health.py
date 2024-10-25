from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator

logger = logging.getLogger(__name__)


class AzureCogsTextAnalyticsHealthTool(BaseTool):  # type: ignore[override]
    """Tool that queries the Azure Cognitive Services Text Analytics for Health API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/ai-services/language-service/text-analytics-for-health/quickstart?tabs=windows&pivots=programming-language-python
    """

    azure_cogs_key: str = ""  #: :meta private:
    azure_cogs_endpoint: str = ""  #: :meta private:
    text_analytics_client: Any  #: :meta private:

    name: str = "azure_cognitive_services_text_analyics_health"
    description: str = (
        "A wrapper around Azure Cognitive Services Text Analytics for Health. "
        "Useful for when you need to identify entities in healthcare data. "
        "Input should be text."
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        azure_cogs_key = get_from_dict_or_env(
            values, "azure_cogs_key", "AZURE_COGS_KEY"
        )

        azure_cogs_endpoint = get_from_dict_or_env(
            values, "azure_cogs_endpoint", "AZURE_COGS_ENDPOINT"
        )

        try:
            import azure.ai.textanalytics as sdk
            from azure.core.credentials import AzureKeyCredential

            values["text_analytics_client"] = sdk.TextAnalyticsClient(
                endpoint=azure_cogs_endpoint,
                credential=AzureKeyCredential(azure_cogs_key),
            )

        except ImportError:
            raise ImportError(
                "azure-ai-textanalytics is not installed. "
                "Run `pip install azure-ai-textanalytics` to install."
            )

        return values

    def _text_analysis(self, text: str) -> Dict:
        poller = self.text_analytics_client.begin_analyze_healthcare_entities(
            [{"id": "1", "language": "en", "text": text}]
        )

        result = poller.result()

        res_dict = {}

        docs = [doc for doc in result if not doc.is_error]

        if docs is not None:
            res_dict["entities"] = [
                f"{x.text} is a healthcare entity of type {x.category}"
                for y in docs
                for x in y.entities
            ]

        return res_dict

    def _format_text_analysis_result(self, text_analysis_result: Dict) -> str:
        formatted_result = []
        if "entities" in text_analysis_result:
            formatted_result.append(
                f"""The text contains the following healthcare entities: {
                        ', '.join(text_analysis_result['entities'])
                }""".replace("\n", " ")
            )

        return "\n".join(formatted_result)

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
            raise RuntimeError(
                f"Error while running AzureCogsTextAnalyticsHealthTool: {e}"
            )

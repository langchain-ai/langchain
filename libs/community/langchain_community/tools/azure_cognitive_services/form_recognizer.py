from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env

from langchain_community.tools.azure_cognitive_services.utils import (
    detect_file_src_type,
)

logger = logging.getLogger(__name__)


class AzureCogsFormRecognizerTool(BaseTool):
    """Tool that queries the Azure Cognitive Services Form Recognizer API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/applied-ai-services/form-recognizer/quickstarts/get-started-sdks-rest-api?view=form-recog-3.0.0&pivots=programming-language-python
    """

    azure_cogs_key: str = ""  #: :meta private:
    azure_cogs_endpoint: str = ""  #: :meta private:
    doc_analysis_client: Any  #: :meta private:

    name: str = "azure_cognitive_services_form_recognizer"
    description: str = (
        "A wrapper around Azure Cognitive Services Form Recognizer. "
        "Useful for when you need to "
        "extract text, tables, and key-value pairs from documents. "
        "Input should be a url to a document."
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
            from azure.ai.formrecognizer import DocumentAnalysisClient
            from azure.core.credentials import AzureKeyCredential

            values["doc_analysis_client"] = DocumentAnalysisClient(
                endpoint=azure_cogs_endpoint,
                credential=AzureKeyCredential(azure_cogs_key),
            )

        except ImportError:
            raise ImportError(
                "azure-ai-formrecognizer is not installed. "
                "Run `pip install azure-ai-formrecognizer` to install."
            )

        return values

    def _parse_tables(self, tables: List[Any]) -> List[Any]:
        result = []
        for table in tables:
            rc, cc = table.row_count, table.column_count
            _table = [["" for _ in range(cc)] for _ in range(rc)]
            for cell in table.cells:
                _table[cell.row_index][cell.column_index] = cell.content
            result.append(_table)
        return result

    def _parse_kv_pairs(self, kv_pairs: List[Any]) -> List[Any]:
        result = []
        for kv_pair in kv_pairs:
            key = kv_pair.key.content if kv_pair.key else ""
            value = kv_pair.value.content if kv_pair.value else ""
            result.append((key, value))
        return result

    def _document_analysis(self, document_path: str) -> Dict:
        document_src_type = detect_file_src_type(document_path)
        if document_src_type == "local":
            with open(document_path, "rb") as document:
                poller = self.doc_analysis_client.begin_analyze_document(
                    "prebuilt-document", document
                )
        elif document_src_type == "remote":
            poller = self.doc_analysis_client.begin_analyze_document_from_url(
                "prebuilt-document", document_path
            )
        else:
            raise ValueError(f"Invalid document path: {document_path}")

        result = poller.result()
        res_dict = {}

        if result.content is not None:
            res_dict["content"] = result.content

        if result.tables is not None:
            res_dict["tables"] = self._parse_tables(result.tables)

        if result.key_value_pairs is not None:
            res_dict["key_value_pairs"] = self._parse_kv_pairs(result.key_value_pairs)

        return res_dict

    def _format_document_analysis_result(self, document_analysis_result: Dict) -> str:
        formatted_result = []
        if "content" in document_analysis_result:
            formatted_result.append(
                f"Content: {document_analysis_result['content']}".replace("\n", " ")
            )

        if "tables" in document_analysis_result:
            for i, table in enumerate(document_analysis_result["tables"]):
                formatted_result.append(f"Table {i}: {table}".replace("\n", " "))

        if "key_value_pairs" in document_analysis_result:
            for kv_pair in document_analysis_result["key_value_pairs"]:
                formatted_result.append(
                    f"{kv_pair[0]}: {kv_pair[1]}".replace("\n", " ")
                )

        return "\n".join(formatted_result)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            document_analysis_result = self._document_analysis(query)
            if not document_analysis_result:
                return "No good document analysis result was found"

            return self._format_document_analysis_result(document_analysis_result)
        except Exception as e:
            raise RuntimeError(f"Error while running AzureCogsFormRecognizerTool: {e}")

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.edenai.edenai_base_tool import EdenaiTool

logger = logging.getLogger(__name__)


class EdenAiParsingInvoice(EdenaiTool):
    """Tool that queries the Eden AI Invoice parsing API.

    for api reference check edenai documentation:
    https://docs.edenai.co/reference/ocr_invoice_parser_create.

    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """

    edenai_api_key: Optional[str] = None

    name = "edenai_invoice_parsing"

    description = (
        "A wrapper around edenai Services invoice parsing. "
        """Useful for when you have to extract information from 
        an image it enables to take invoices 
        in a variety of formats and returns the data in contains
        (items, prices, addresses, vendor name, etc.)
        in a structured format to automate the invoice processing """
        "Input should be the string url of the document to parse."
    )

    language: Optional[str] = None
    """
    language of the image passed to the model.
    """

    params: Optional[Dict[str, Any]] = Field(default_factory=dict)

    feature = "ocr"
    subfeature = "invoice_parser"

    def _parse_json(
        self, extracted_data: dict, formatted_list: list, level: int = 0
    ) -> None:
        for section, subsections in extracted_data.items():
            indentation = "  " * level
            if isinstance(subsections, str):
                subsections = subsections.replace("\n", ",")
                formatted_list.append(f"{indentation}{section} : {subsections}")

            elif isinstance(subsections, list):
                formatted_list.append(f"{indentation}{section} : ")
                self._list_handling(subsections, formatted_list, level + 1)

            elif isinstance(subsections, dict):
                formatted_list.append(f"{indentation}{section} : ")
                self._parse_json(subsections, formatted_list, level + 1)

    def _list_handling(
        self, subsection_list: list, formatted_list: list, level: int
    ) -> None:
        for list_item in subsection_list:
            if isinstance(list_item, dict):
                self._parse_json(list_item, formatted_list, level)

            elif isinstance(list_item, list):
                self._list_handling(list_item, formatted_list, level + 1)

            else:
                formatted_list.append(f"{'  ' * level}{list_item}")

    def _format_invoice_parsing(self, json_data: list) -> str:
        formatted_list: list = []

        if len(json_data) == 1:
            self._parse_json(json_data[0]["extracted_data"][0], formatted_list)
        else:
            for entry in json_data:
                if entry.get("provider") == "eden-ai":
                    self._parse_json(entry["extracted_data"][0], formatted_list)

        return "\n".join(formatted_list)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            if self.params is None:
                query_params = {
                    "file_url": query,
                    "language": self.language,
                    "attributes_as_list": False,
                }
            else:
                query_params = {
                    "file_url": query,
                    "language": self.language,
                    **self.params,
                    "attributes_as_list": False,
                }

            image_analysis_result = self._call_eden_ai(query_params)
            image_analysis_dict = image_analysis_result.json()
            return self._format_invoice_parsing(image_analysis_dict)

        except Exception as e:
            raise RuntimeError(f"Error while running EdenAiExplicitText: {e}")

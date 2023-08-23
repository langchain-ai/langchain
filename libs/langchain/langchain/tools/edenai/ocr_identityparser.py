from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.edenai.edenai_base_tool import EdenaiTool

logger = logging.getLogger(__name__)


class EdenAiParsingIDTool(EdenaiTool):
    """Tool that queries the Eden AI  Identity parsing API.

    for api reference check edenai documentation:
    https://docs.edenai.co/reference/ocr_identity_parser_create.

    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """

    edenai_api_key: Optional[str] = None

    name = "edenai_identity_parsing"

    description = (
        "A wrapper around edenai Services Identity parsing. "
        "Useful for when you have to extract information from an ID Document "
        "Input should be the string url of the document to parse."
    )

    feature = "ocr"
    subfeature = "identity_parser"

    language: Optional[str] = None
    """
    language of the text passed to the model.
    """

    params: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def _format_id_parsing(self, json_data: list) -> str:
        formatted_list: list = []

        if len(json_data) == 1:
            self._parse_json_multilevel(
                json_data[0]["extracted_data"][0], formatted_list
            )
        else:
            for entry in json_data:
                if entry.get("provider") == "eden-ai":
                    self._parse_json_multilevel(
                        entry["extracted_data"][0], formatted_list
                    )

        return "\n".join(formatted_list)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
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
        try:
            image_analysis_dict = image_analysis_result.json()
            return self._format_id_parsing(image_analysis_dict)

        except Exception as e:
            raise RuntimeError(f"Error while running EdenAiExplicitText: {e}")

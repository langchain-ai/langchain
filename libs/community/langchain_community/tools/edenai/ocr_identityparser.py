from __future__ import annotations

import logging
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field, HttpUrl

from langchain_community.tools.edenai.edenai_base_tool import EdenaiTool

logger = logging.getLogger(__name__)


class IDParsingInput(BaseModel):
    query: HttpUrl = Field(description="url of the document to parse")


class EdenAiParsingIDTool(EdenaiTool):
    """Tool that queries the Eden AI  Identity parsing API.

    for api reference check edenai documentation:
    https://docs.edenai.co/reference/ocr_identity_parser_create.

    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """

    name: str = "edenai_identity_parsing"

    description: str = (
        "A wrapper around edenai Services Identity parsing. "
        "Useful for when you have to extract information from an ID Document "
        "Input should be the string url of the document to parse."
    )
    args_schema: Type[BaseModel] = IDParsingInput

    feature: str = "ocr"
    subfeature: str = "identity_parser"

    language: Optional[str] = None
    """
    language of the text passed to the model.
    """

    def _parse_response(self, response: list) -> str:
        formatted_list: list = []

        if len(response) == 1:
            self._parse_json_multilevel(
                response[0]["extracted_data"][0], formatted_list
            )
        else:
            for entry in response:
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
        query_params = {
            "file_url": query,
            "language": self.language,
            "attributes_as_list": False,
        }

        return self._call_eden_ai(query_params)

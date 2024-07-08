from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env

from langchain_community.utilities.box import BoxAPIWrapper, BoxAuthType

logger = logging.getLogger(__name__)

class BoxFileSearchInput(BaseModel):
    """Input for the BoxFileSearch tool."""

    query: str = Field(description="search query to look up")

class BoxFileSearchTool(BaseTool):
    """Tool that performs natural language search on Box Content Cloud
    """

    box_developer_token: str = ""  #: :meta private:

    name: str = "box_file_search"
    description: str = (
        "A wrapper for performing plain language search for files in Box. "
        "This tool will return a List[str] containing file ids. "
        "These file ids can be used with other tools"
    )
    args_schema: Type[BaseModel] = BoxFileSearchInput

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        box_developer_token = get_from_dict_or_env(
            values, "box_developer_token", "BOX_DEVELOPER_TOKEN"
        )

        box = BoxAPIWrapper(
            auth_type="token",
            box_developer_token=box_developer_token
        )

        values["box"] = box

        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Document]:
        """Use the tool."""
        try:
            return self.box.get_documents_by_search(query)
        except Exception as e:
            repr(f"Error while running BoxFileSearchTool: {e}")

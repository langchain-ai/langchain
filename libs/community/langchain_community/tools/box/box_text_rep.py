from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, List

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env

from langchain_community.utilities.box import BoxAPIWrapper, BoxAuthType

logger = logging.getLogger(__name__)


class BoxTextRepTool(BaseTool):
    """
        Tool that retrieves a text representation of any file that has one.
    """

    box_developer_token: str = ""  #: :meta private:
    box_file_id: str

    name: str = "box_text_rep"
    description: str = (
        "A wrapper for Box to retrieve the text representation of a file. "
        "set query equal to a string equal to the file Id you wish to "
        "download the text representation for."
    )

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        box_developer_token = get_from_dict_or_env(
            values, "box_developer_token", "BOX_DEVELOPER_TOKEN"
        )

        if not values.get("box_file_id"):
            raise ValueError("Box AI requires a file_id.")
        
        box_file_id = values.get("box_file_id")

        box = BoxAPIWrapper(
            auth_type="token",
            box_developer_token=box_developer_token,
            box_file_id=box_file_id
        )

        values["box"] = box

        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            return self.box.get_text_representation(query)
        except Exception as e:
            raise RuntimeError(
                f"Error while running BoxTextRepTool: {e}"
            )

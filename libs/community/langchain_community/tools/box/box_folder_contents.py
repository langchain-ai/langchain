from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env

from langchain_community.utilities.box import BoxAPIWrapper, BoxAuthType

logger = logging.getLogger(__name__)


class BoxFolderContentsTool(BaseTool):
    """List all files in a folder in Box
    """

    box_developer_token: str = ""  #: :meta private:

    name: str = "box_folder_contents"
    description: str = (
        "A wrapper for listing all the files in a Box folder. "
        "This tool will return a Iterator[str,str] containing file names and ids. "
        "Query should be set to a string containing the folder id. "
        "These file ids can be used with other tools."
    )

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        box_developer_token = get_from_dict_or_env(
            values, "box_developer_token", "BOX_DEVELOPER_TOKEN"
        )

        box = BoxAPIWrapper(
            auth_type=BoxAuthType.TOKEN,
            box_developer_token=box_developer_token
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
            return self.box.get_folder_information(query)
        except Exception as e:
            raise RuntimeError(
                f"Error while running BoxFolderContentsTool: {e}"
            )

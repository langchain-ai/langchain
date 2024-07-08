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


class BoxAIAskTool(BaseTool):
    """
        Tool that calls Box AI to ask a question on a document(s) and adds the answer to your Documents.
        This is only available to Box customers on an Enterprise+ plan or higher.
    """

    box_developer_token: str = ""  #: :meta private:
    box_file_ids: List[str] = None

    name: str = "box_ai_ask"
    description: str = (
        "A wrapper for asking Box AI a question about one or more documents. "
        "The response is in the form of List[Document] to add to your document list."
    )

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        box_developer_token = get_from_dict_or_env(
            values, "box_developer_token", "BOX_DEVELOPER_TOKEN"
        )

        if not values.get("box_file_ids"):
            raise ValueError("Box AI requires List[str] with file_ids.")

        box = BoxAPIWrapper(
            auth_type=BoxAuthType.TOKEN,
            box_developer_token=box_developer_token,
            box_file_ids=values.get("box_file_ids")
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
            return self.box.get_documents_by_box_ai_ask(query, True)
        except Exception as e:
            raise RuntimeError(
                f"Error while running BoxAIAskTool: {e}"
            )

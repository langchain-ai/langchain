from abc import abstractmethod
from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool


class AudioTool(BaseTool):
    """
    The base tool for audio tools.
    """

    @abstractmethod
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        pass

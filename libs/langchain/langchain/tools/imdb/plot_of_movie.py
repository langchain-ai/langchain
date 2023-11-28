from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool


class IMDBPlotOfMovie(IMDbBaseTool):
    """Tool to find plot of a movie given its name."""

    name: str = "PlotOfMovie"
    description: str = (
        """Use this tool to retrieve a summary of the plot of a movie, 
        given its IMDB movie ID."""
    )

    def _run(self, 
             id: str, 
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        res_movie = self.client.get_movie(id)

        return res_movie['plot']

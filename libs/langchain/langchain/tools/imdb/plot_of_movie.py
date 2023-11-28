import json
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool


class IMDBPlotOfMovie(IMDbBaseTool):
    """Tool to find plot of a movie given its name."""

    name: str = "PlotOfMovie"
    description: str = """Use this tool to retrieve a summary of the plot of a movie, 
        given its IMDB movie ID."""

    def _run(
        self, id: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        from imdb import IMDbError

        try:
            res_movie = self.client.get_movie(id)
        except IMDbError:
            return (
                "The movie could not be found. "
                "Please make sure to give a movie ID instead of the movie's name."
            )

        return json.dumps(res_movie["plot"])

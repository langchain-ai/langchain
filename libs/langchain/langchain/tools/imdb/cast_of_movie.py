import json
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool
from langchain.tools.imdb.utils import people_to_dicts


class IMDBCastOfMovie(IMDbBaseTool):
    """Tool to find cast of a movie given its name."""

    name: str = "CastOfMovie"
    description: str = (
        "Use this tool to retrieve a list of cast members for a movie, given "
        "its IMBD movie ID."
    )

    def _run(
        self, id: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        res_movie = self.client.get_movie(id)

        return json.dumps(people_to_dicts(res_movie["cast"]))

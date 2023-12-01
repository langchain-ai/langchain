import json
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool
from langchain.tools.imdb.utils import people_to_dicts


class IMDbCastOfMovie(IMDbBaseTool):
    """Tool to find cast of a movie given its name."""

    name: str = "imdb_cast_of_movie"
    description: str = (
        "Use this tool to retrieve a list of cast members for a movie, given "
        "its IMBD movie ID."
    )

    def _run(
        self, id: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        from imdb import IMDbError

        try:
            res_movie = self.client.get_movie(id)
        except IMDbError:
            return (
                "The movie could not be found. "
                "Please make sure to give a movie ID instead of a movie name."
            )

        return json.dumps(people_to_dicts(res_movie["cast"]))

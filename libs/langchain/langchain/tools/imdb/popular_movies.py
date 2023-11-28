from typing import Optional
import json

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool
from langchain.tools.imdb.utils import movies_to_dicts


class IMDbPopularMovies(IMDbBaseTool):
    """Tool that gives a list of the most popular movies."""

    name: str = "imdb_popular_movies"
    description: str = (
        "A wrapper around IMDb. "
        "Useful for getting the current most popular movies. "
        "Input is ignored. Output is a JSON array containing the movies."
    )

    def _run(
        self, input, run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        movies = self.client.get_popular100_movies()
        return json.dumps(movies_to_dicts(movies[:20]))
import json
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool


class IMDbSearchMovie(IMDbBaseTool):
    """Tool that searches movies with a given title."""

    name: str = "imdb_search_movie"
    description: str = (
        "Searches IMDb for a movie with the given title and returns a "
        "JSON array containing the search results, sorted by relevance. "
        "Each entry in the array contains the movie title and its ID."
    )

    def _run(
        self,
        title: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        movies = self.client.search_movie(title)
        movies = [{"title": m.get("title"), "id": m.getID()} for m in movies]
        return json.dumps(movies)

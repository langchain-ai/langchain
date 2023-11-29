import json
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool


class IMDbSearchMovieKeyword(IMDbBaseTool):
    """Tool that searches movies with key word"""

    name: str = "imdb_search_movie_keyword"
    description: str = (
        "Searches IMDb for a movie with the given key word returns a "
        "JSON array containing the search results, sorted by relevance. "
        "Each entry in the array contains the movie title and its ID."
    )

    def _run(
        self,
        keyword: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Searches IMDB for a movie that match a given keyword
        returns a JSON array containing the search results, sorted by relevance.
        Each entry in the array contains the movie title and its ID.
        """
        keyword = keyword.replace(" ", "-")
        m = self.client.get_keyword(keyword)
        if len(m) == 0:
            keylist = self.client.search_keyword(keyword)
            if keylist:
                keylist_str = ", ".join(keylist)
                return (
                    "No movies found for this keyword. Try these keywords instead: "
                    + keylist_str
                )
            else:
                return "No movies found for this keyword."

        movies = list(map(lambda m: {"title": m.get("title"), "id": m.getID()}, m[:3]))
        return json.dumps(movies)

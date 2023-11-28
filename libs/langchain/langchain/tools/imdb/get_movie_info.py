from typing import Optional
import json

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool
from langchain.tools.imdb.utils import companies_to_strs


class IMDbGetMovieInfo(IMDbBaseTool):
    """Tool that fetches miscellaneous info about a movie."""

    name: str = "imdb_get_movie_info"
    description: str = (
        "A wrapper around IMDb. "
        "Useful for when you need to know any of the following "
        "information about a movie: "
        "title, year of release, genre, runtime (in minutes), country, "
        "language, box office data, rating, "
        "production companies, or distributors. "
        "Input should be a movie ID number. "
        "Output is a JSON object containing the results."
    )

    def _run(
        self,
        id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from imdb import IMDbError
        try:
            movie = self.client.get_movie(id)
        except IMDbError:
            return (
                "The movie could not be found. "
                "Please make sure to give a movie ID instead of the movie title."
            )
        
        keys = [
            'title',
            'year',
            'genres',
            'runtimes',
            'countries',
            'languages',
            'box office',
            'rating',
        ]
        info = {
            key: movie.get(key)
            for key in keys
            if key in movie
        }

        prod_companies = movie.get('production companies')
        if prod_companies:
            info['production companies'] = companies_to_strs(prod_companies)
        distributors = movie.get('distributors')
        if distributors:
            info['distributors'] = companies_to_strs(distributors)

        return json.dumps(info)
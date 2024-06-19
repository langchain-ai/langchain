from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

from neo4j_semantic_ollama.utils import get_candidates, get_user_id, graph

recommendation_query_db_history = """
  MERGE (u:User {userId:$user_id})
  WITH u
  // get recommendation candidates
  OPTIONAL MATCH (u)-[r1:RATED]->()<-[r2:RATED]-()-[r3:RATED]->(recommendation)
  WHERE r1.rating > 3.5 AND r2.rating > 3.5 AND r3.rating > 3.5
        AND NOT EXISTS {(u)-[:RATED]->(recommendation)}
  // rank and limit recommendations
  WITH u, recommendation, count(*) AS count
  ORDER BY count DESC LIMIT 3
RETURN 'title:' + recommendation.title + '\nactors:' +
apoc.text.join([(recommendation)<-[:ACTED_IN]-(a) | a.name], ',') +
'\ngenre:' + apoc.text.join([(recommendation)-[:IN_GENRE]->(a) | a.name], ',')
AS movie
"""

recommendation_query_genre = """
MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name:$genre})
// filter out already seen movies by the user
WHERE NOT EXISTS {
  (m)<-[:RATED]-(:User {userId:$user_id})
}
// rank and limit recommendations
WITH m AS recommendation
ORDER BY recommendation.imdbRating DESC LIMIT 3
RETURN 'title:' + recommendation.title + '\nactors:' +
apoc.text.join([(recommendation)<-[:ACTED_IN]-(a) | a.name], ',') +
'\ngenre:' + apoc.text.join([(recommendation)-[:IN_GENRE]->(a) | a.name], ',')
AS movie
"""


def recommendation_query_movie(genre: bool) -> str:
    return f"""
MATCH (m1:Movie)<-[r1:RATED]-()-[r2:RATED]->(m2:Movie)
WHERE r1.rating > 3.5 AND r2.rating > 3.5 and m1.title IN $movieTitles
// filter out already seen movies by the user
AND NOT EXISTS {{
  (m2)<-[:RATED]-(:User {{userId:$user_id}})
}}
{'AND EXISTS {(m2)-[:IN_GENRE]->(:Genre {name:$genre})}' if genre else ''}
// rank and limit recommendations
WITH m2 AS recommendation, count(*) AS count
ORDER BY count DESC LIMIT 3
RETURN 'title:' + recommendation.title + '\nactors:' +
apoc.text.join([(recommendation)<-[:ACTED_IN]-(a) | a.name], ',') +
'\ngenre:' + apoc.text.join([(recommendation)-[:IN_GENRE]->(a) | a.name], ',')
AS movie
"""


nl = "\n"


def recommend_movie(movie: Optional[str] = None, genre: Optional[str] = None) -> str:
    """
    Recommends movies based on user's history and preference
    for a specific movie and/or genre.
    Returns:
        str: A string containing a list of recommended movies, or an error message.
    """
    user_id = get_user_id()
    params = {"user_id": user_id, "genre": genre}
    if not movie and not genre:
        # Try to recommend a movie based on the information in the db
        response = graph.query(recommendation_query_db_history, params)
        try:
            return (
                'Recommended movies are: '
                f'{f"###Movie {nl}".join([el["movie"] for el in response])}'
            )
        except Exception:
            return "Can you tell us about some of the movies you liked?"
    if not movie and genre:
        # Recommend top voted movies in the genre the user haven't seen before
        response = graph.query(recommendation_query_genre, params)
        try:
            return (
                'Recommended movies are: '
                f'{f"###Movie {nl}".join([el["movie"] for el in response])}'
            )
        except Exception:
            return "Something went wrong"

    candidates = get_candidates(movie, "movie")
    if not candidates:
        return "The movie you mentioned wasn't found in the database"
    params["movieTitles"] = [el["candidate"] for el in candidates]
    query = recommendation_query_movie(bool(genre))
    response = graph.query(query, params)
    try:
        return (
            'Recommended movies are: '
            f'{f"###Movie {nl}".join([el["movie"] for el in response])}'
        )
    except Exception:
        return "Something went wrong"


all_genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


class RecommenderInput(BaseModel):
    movie: Optional[str] = Field(description="movie used for recommendation")
    genre: Optional[str] = Field(
        description=(
            "genre used for recommendation. Available options are:" f"{all_genres}"
        )
    )


class RecommenderTool(BaseTool):
    name = "Recommender"
    description = "useful for when you need to recommend a movie"
    args_schema: Type[BaseModel] = RecommenderInput

    def _run(
        self,
        movie: Optional[str] = None,
        genre: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return recommend_movie(movie, genre)

    async def _arun(
        self,
        movie: Optional[str] = None,
        genre: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return recommend_movie(movie, genre)

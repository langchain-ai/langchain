from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

from neo4j_semantic_layer.utils import get_candidates, get_user_id, graph

store_rating_query = """
MERGE (u:User {userId:$user_id}) 
WITH u 
UNWIND $candidates as row
MATCH (m:Movie {title: row.candidate})
MERGE (u)-[r:RATED]->(m)
SET r.rating = toFloat($rating)
RETURN distinct 'Noted' AS response
"""


def store_movie_rating(movie: str, rating: int):
    user_id = get_user_id()
    candidates = get_candidates(movie, "movie")
    if not candidates:
        return "This movie is not in our database"
    response = graph.query(
        store_rating_query,
        params={"user_id": user_id, "candidates": candidates, "rating": rating},
    )
    try:
        return response[0]["response"]
    except Exception as e:
        print(e)  # noqa: T201
        return "Something went wrong"


class MemoryInput(BaseModel):
    movie: str = Field(description="movie the user liked")
    rating: int = Field(
        description=(
            "Rating from 1 to 5, where one represents heavy dislike "
            "and 5 represent the user loved the movie"
        )
    )


class MemoryTool(BaseTool):
    name = "Memory"
    description = "useful for memorizing which movies the user liked"
    args_schema: Type[BaseModel] = MemoryInput

    def _run(
        self,
        movie: str,
        rating: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return store_movie_rating(movie, rating)

    async def _arun(
        self,
        movie: str,
        rating: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return store_movie_rating(movie, rating)

"""Aggregation pipeline components used in Atlas Full-Text, Vector, and Hybrid Search.

"""
from typing import Any, Dict, List, TypeVar

MongoDBDocumentType = TypeVar("MongoDBDocumentType", bound=Dict[str, Any])


def text_search_stage(
    query: str, search_field, index_name: str, operator: str = "phrase", **kwargs: Any
) -> Dict[str, Any]:
    """Full-Text search.

    Args:
        query: Input text to search for
        search_field: Field in Collection that will be searched
        index_name: Atlas Search Index name
        operator: A number of operators are available in the text search stage.

    Returns:
        Dictionary defining the $search

    See Also:
        - MongoDB Full-Text Search <https://www.mongodb.com/docs/atlas/atlas-search/aggregation-stages/search/#mongodb-pipeline-pipe.-search>
        - MongoDB Operators <https://www.mongodb.com/docs/atlas/atlas-search/operators-and-collectors/#std-label-operators-ref>
    """
    return {
        "$search": {
            "index": index_name,
            operator: {"query": query, "path": search_field},
        }
    }


def vector_search_stage(
    query_vector: List[float],
    search_field: str,
    index_name: str,
    limit: int = 4,
    filter: MongoDBDocumentType = None,
    oversampling_factor=10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Vector Search Stage without Scores.

    Scoring is applied later depending on strategy.
    vector search includes a vectorSearchScore that is typically used.
    hybrid uses Reciprocal Rank Fusion.

    Args:
        query_vector: List of embedding vector
        search_field: Field in Collection containing embedding vectors
        index_name: Name of Atlas Vector Search Index tied to Collection
        limit: Number of documents to return
        oversampling_factor: this times limit is the number of candidates
        filters: Any MQL match expression comparing an indexed field

    Returns:
        Dictionary defining the $vectorSearch
    """
    return {
        "$vectorSearch": {
            "index": index_name,
            "path": search_field,
            "queryVector": query_vector,
            "numCandidates": limit * oversampling_factor,
            "limit": limit,
            "filter": filter,
        }
    }


def combine_pipelines(
    pipeline: List[Any], stage: List[Dict[str, Any]], collection_name: str
):
    """Combines two aggregations into a single result set."""
    if pipeline:
        pipeline.append({"$unionWith": {"coll": collection_name, "pipeline": stage}})
    else:
        pipeline.extend(stage)
    return pipeline


def reciprocal_rank_stage(
    text_field, score_field: str, penalty: float = 0, **kwargs: Any
) -> List[Dict[str, Any]]:
    """Stage adds Reciprocal Rank Fusion weighting.

        First, it pushes documents retrieved from previous stage
        into a temporary sub-document. It then unwinds to establish
        the rank to each and applies the penalty.

    Args:
        text_field: Collection field containing relevant to text per VectorStore API
        score_field: A unique string to indentify the search being ranked
        penalty: A non-negative float.
        extra_fields: Any fields other than text_field that one wishes to keep.

    Returns:
        RRF score := \frac{1}{rank + penalty} with rank in [1,2,..,n]
    """

    rrf_pipeline = [
        {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
        {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
        {
            "$addFields": {
                f"docs.{score_field}": {
                    "$divide": [1.0, {"$add": ["$rank", penalty, 1]}]
                },
                "docs.rank": "$rank",
                "_id": "$docs._id",
            }
        },
        {"$replaceRoot": {"newRoot": "$docs"}},
    ]

    return rrf_pipeline


def final_hybrid_stage(
    scores_fields: List[str], limit: int, **kwargs: Any
) -> List[Dict[str, Any]]:
    """Sum weighted scores, sort, and apply limit.

    Args:
        scores_fields: List of fields given to scores of vector and text searches
        limit: Number of documents to return

    Returns:
        Final aggregation stages
    """

    return [
        {"$group": {"_id": "$_id", "docs": {"$mergeObjects": "$$ROOT"}}},
        {"$replaceRoot": {"newRoot": "$docs"}},
        {"$set": {score: {"$ifNull": [f"${score}", 0]} for score in scores_fields}},
        {"$addFields": {"score": {"$add": [f"${score}" for score in scores_fields]}}},
        {"$sort": {"score": -1}},
        {"$limit": limit},
    ]

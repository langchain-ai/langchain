"""Aggregation pipeline components used in Atlas Full-Text, Vector, and Hybrid Search

See the following for more:
    - `Full-Text Search <https://www.mongodb.com/docs/atlas/atlas-search/aggregation-stages/search/#mongodb-pipeline-pipe.-search>`_
    - `MongoDB Operators <https://www.mongodb.com/docs/atlas/atlas-search/operators-and-collectors/#std-label-operators-ref>`_
    - `Vector Search <https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/>`_
    - `Filter Example <https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/#atlas-vector-search-pre-filter>`_
"""

from typing import Any, Dict, List, Optional


def text_search_stage(
    query: str,
    search_field: str,
    index_name: str,
    limit: Optional[int] = None,
    filter: Optional[Dict[str, Any]] = None,
    include_scores: Optional[bool] = True,
    **kwargs: Any,
) -> List[Dict[str, Any]]:  # noqa: E501
    """Full-Text search using Lucene's standard (BM25) analyzer

    Args:
        query: Input text to search for
        search_field: Field in Collection that will be searched
        index_name: Atlas Search Index name
        limit: Maximum number of documents to return. Default of no limit
        filter: Any MQL match expression comparing an indexed field
        include_scores: Scores provide measure of relative relevance

    Returns:
        Dictionary defining the $search stage
    """
    pipeline = [
        {
            "$search": {
                "index": index_name,
                "text": {"query": query, "path": search_field},
            }
        }
    ]
    if filter:
        pipeline.append({"$match": filter})  # type: ignore
    if include_scores:
        pipeline.append({"$set": {"score": {"$meta": "searchScore"}}})
    if limit:
        pipeline.append({"$limit": limit})  # type: ignore

    return pipeline  # type: ignore


def vector_search_stage(
    query_vector: List[float],
    search_field: str,
    index_name: str,
    top_k: int = 4,
    filter: Optional[Dict[str, Any]] = None,
    oversampling_factor: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:  # noqa: E501
    """Vector Search Stage without Scores.

    Scoring is applied later depending on strategy.
    vector search includes a vectorSearchScore that is typically used.
    hybrid uses Reciprocal Rank Fusion.

    Args:
        query_vector: List of embedding vector
        search_field: Field in Collection containing embedding vectors
        index_name: Name of Atlas Vector Search Index tied to Collection
        top_k: Number of documents to return
        oversampling_factor: this times limit is the number of candidates
        filter: MQL match expression comparing an indexed field.
            Some operators are not supported.
            See `vectorSearch filter docs <https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/#atlas-vector-search-pre-filter>`_


    Returns:
        Dictionary defining the $vectorSearch
    """
    stage = {
        "index": index_name,
        "path": search_field,
        "queryVector": query_vector,
        "numCandidates": top_k * oversampling_factor,
        "limit": top_k,
    }
    if filter:
        stage["filter"] = filter
    return {"$vectorSearch": stage}


def combine_pipelines(
    pipeline: List[Any], stage: List[Dict[str, Any]], collection_name: str
) -> None:
    """Combines two aggregations into a single result set in-place."""
    if pipeline:
        pipeline.append({"$unionWith": {"coll": collection_name, "pipeline": stage}})
    else:
        pipeline.extend(stage)


def reciprocal_rank_stage(
    score_field: str, penalty: float = 0, **kwargs: Any
) -> List[Dict[str, Any]]:
    """Stage adds Reciprocal Rank Fusion weighting.

        First, it pushes documents retrieved from previous stage
        into a temporary sub-document. It then unwinds to establish
        the rank to each and applies the penalty.

    Args:
        score_field: A unique string to identify the search being ranked
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

    return rrf_pipeline  # type: ignore


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

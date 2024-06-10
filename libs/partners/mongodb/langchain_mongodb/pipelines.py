"""Aggregation pipeline components involved in Atlas Full-Text, Vector, and Hybrid Search.

"""
from typing import Any, Dict, List


def text_search_stage(
    query: str, search_field, index_name: str, operator: str = "phrase"
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
    filter: Dict[str, Any] = None,
    oversampling_factor=10,
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


def combine_pipelines(pipeline: List[Any], stage: Dict[str, Any], collection_name):
    """Combines two aggregations into a single result set."""
    if pipeline:
        pipeline.append({"$unionWith": {"coll": collection_name, "pipeline": stage}})
    else:
        pipeline.extend(stage)
    return pipeline


def reciprocal_rank_stage(
    text_field, score_field: str, penalty: float = 0, extra_fields: List[str] = None
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
                score_field: {"$divide": [1.0, {"$add": ["$rank", penalty, 1]}]}
            }
        },
    ]
    projection_fields = {text_field: f"$docs.{text_field}"}
    projection_fields["_id"] = "$docs._id"
    projection_fields[score_field] = 1
    if extra_fields:
        projection_fields.update({f"$docs.{key}" for key in extra_fields})

    rrf_pipeline.append({"$project": projection_fields})
    return rrf_pipeline


def final_hybrid_stage(
    scores_fields: List[str],
    limit: int,
    text_field: str,
    extra_fields: List[str] = None,
) -> List[Dict[str, Any]]:
    """Sum weighted scores, sort, and apply limit.

    Args:
        scores_fields: List of fields given to scores of vector and text searches
        limit: Number of documents to return
        text_field: Collection field containing relevant to text per VectorStore API
        extra_fields: Any fields other than text_field that one wishes to keep.

    Returns:
        Final aggregation stages
    """

    doc_fields = [text_field]
    if extra_fields:
        doc_fields.extend(extra_fields)

    grouped_fields = {key: {"$first": f"${key}"} for key in doc_fields}
    best_score = {score: {"$max": f"${score}"} for score in scores_fields}
    final_pipeline = [
        {"$group": {"_id": "$_id", **grouped_fields, **best_score}},
        {
            "$project": {
                **{doc_field: 1 for doc_field in doc_fields},
                **{score: {"$ifNull": [f"${score}", 0]} for score in scores_fields},
            }
        },
        {
            "$addFields": {
                "score": {"$add": [f"${score}" for score in scores_fields]},
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": limit},
    ]
    return final_pipeline

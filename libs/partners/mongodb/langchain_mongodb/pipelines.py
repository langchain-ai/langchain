def _reciprocal_rank_stage(text_field, score_field: str, penalty: float = 0, extra_fields: List[str] = None):
    """Pipeline stage that ranks and weights scores.

        Pushes documents retrieved from previous stage into a temporary sub-document.
        It then unwinds to establish the rank to each and applies the penalty.
    """
    rrf_pipeline = [
        {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
        {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
        {
            "$addFields": {
                score_field: {"$divide": [1.0, {"$add": ["$rank", penalty, 1]}]}
            }
        }
    ]
    projection_fields = {text_field: f"$docs.{text_field}"}
    projection_fields["_id"] = "$docs._id"
    projection_fields[score_field] = 1
    if extra_fields:
        projection_fields.update({f"$docs.{key}" for key in extra_fields})

    rrf_pipeline.append({'$project': projection_fields})
    return rrf_pipeline

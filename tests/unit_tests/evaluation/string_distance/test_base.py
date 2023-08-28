import pytest

from langchain.evaluation.string_distance import (
    PairwiseStringDistanceEvalChain,
    StringDistance,
    StringDistanceEvalChain,
)


@pytest.mark.requires("rapidfuzz")
@pytest.mark.parametrize("distance", list(StringDistance))
def test_zero_distance(distance: StringDistance) -> None:
    eval_chain = StringDistanceEvalChain(distance=distance)
    string = "三人行则必有我师"
    result = eval_chain.evaluate_strings(prediction=string, reference=string)
    assert "score" in result
    assert result["score"] == 0


@pytest.mark.asyncio
@pytest.mark.requires("rapidfuzz")
@pytest.mark.parametrize("distance", list(StringDistance))
async def test_zero_distance_async(distance: StringDistance) -> None:
    eval_chain = StringDistanceEvalChain(distance=distance)
    string = "三人行则必有我师"
    result = await eval_chain.aevaluate_strings(prediction=string, reference=string)
    assert "score" in result
    assert result["score"] == 0


@pytest.mark.requires("rapidfuzz")
@pytest.mark.parametrize("distance", list(StringDistance))
@pytest.mark.parametrize("normalize_score", [True, False])
def test_zero_distance_pairwise(
    distance: StringDistance, normalize_score: bool
) -> None:
    eval_chain = PairwiseStringDistanceEvalChain(
        distance=distance, normalize_score=normalize_score
    )
    string = "三人行则必有我师"
    result = eval_chain.evaluate_string_pairs(prediction=string, prediction_b=string)
    assert "score" in result
    assert result["score"] == 0


@pytest.mark.asyncio
@pytest.mark.requires("rapidfuzz")
@pytest.mark.parametrize("distance", list(StringDistance))
async def test_zero_distance_pairwise_async(distance: StringDistance) -> None:
    eval_chain = PairwiseStringDistanceEvalChain(distance=distance)
    string = "三人行则必有我师"
    result = await eval_chain.aevaluate_string_pairs(
        prediction=string, prediction_b=string
    )
    assert "score" in result
    assert result["score"] == 0


@pytest.mark.requires("rapidfuzz")
@pytest.mark.parametrize("distance", list(StringDistance))
@pytest.mark.parametrize("normalize_score", [True, False])
def test_non_zero_distance(distance: StringDistance, normalize_score: bool) -> None:
    eval_chain = StringDistanceEvalChain(
        distance=distance, normalize_score=normalize_score
    )
    prediction = "I like to eat apples."
    reference = "I like apples."
    result = eval_chain.evaluate_strings(prediction=prediction, reference=reference)
    assert "score" in result
    assert 0 < result["score"]
    if normalize_score:
        assert result["score"] < 1.0


@pytest.mark.asyncio
@pytest.mark.requires("rapidfuzz")
@pytest.mark.parametrize("distance", list(StringDistance))
async def test_non_zero_distance_async(distance: StringDistance) -> None:
    eval_chain = StringDistanceEvalChain(distance=distance)
    prediction = "I like to eat apples."
    reference = "I like apples."
    result = await eval_chain.aevaluate_strings(
        prediction=prediction, reference=reference
    )
    assert "score" in result
    assert 0 < result["score"] < 1.0


@pytest.mark.requires("rapidfuzz")
@pytest.mark.parametrize("distance", list(StringDistance))
def test_non_zero_distance_pairwise(distance: StringDistance) -> None:
    eval_chain = PairwiseStringDistanceEvalChain(distance=distance)
    prediction = "I like to eat apples."
    reference = "I like apples."
    result = eval_chain.evaluate_string_pairs(
        prediction=prediction, prediction_b=reference
    )
    assert "score" in result
    assert 0 < result["score"] < 1.0


@pytest.mark.asyncio
@pytest.mark.requires("rapidfuzz")
@pytest.mark.parametrize("distance", list(StringDistance))
async def test_non_zero_distance_pairwise_async(distance: StringDistance) -> None:
    eval_chain = PairwiseStringDistanceEvalChain(distance=distance)
    prediction = "I like to eat apples."
    reference = "I like apples."
    result = await eval_chain.aevaluate_string_pairs(
        prediction=prediction, prediction_b=reference
    )
    assert "score" in result
    assert 0 < result["score"] < 1.0

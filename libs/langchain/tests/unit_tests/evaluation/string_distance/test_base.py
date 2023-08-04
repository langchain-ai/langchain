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
def test_zero_distance_pairwise(distance: StringDistance) -> None:
    eval_chain = PairwiseStringDistanceEvalChain(distance=distance)
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

# """Test embedding model integration."""

# import os

# import pytest  # type: ignore[import-not-found]

# from langchain_pipeshift import PipeshiftEmbeddings

# os.environ["pipeshift_API_KEY"] = "foo"


# def test_initialization() -> None:
#     """Test embedding model initialization."""
#     PipeshiftEmbeddings()


# def test_pipeshift_invalid_model_kwargs() -> None:
#     with pytest.raises(ValueError):
#         PipeshiftEmbeddings(model_kwargs={"model": "foo"})


# def test_pipeshift_incorrect_field() -> None:
#     with pytest.warns(match="not default parameter"):
#         llm = PipeshiftEmbeddings(foo="bar")  # type: ignore[call-arg]
#     assert llm.model_kwargs == {"foo": "bar"}

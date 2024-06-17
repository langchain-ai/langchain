from langchain_mixedbreadai.rerank import MixedbreadAIRerank


def test_initialization() -> None:
    """Test embedding model initialization."""
    MixedbreadAIRerank(mxbai_api_key="test")

from langchain_mixedbreadai.rerank import MixedbreadAIRerank


def test_initialization() -> None:
    """Test embedding model initialization."""
    MixedbreadAIRerank(api_key="test")

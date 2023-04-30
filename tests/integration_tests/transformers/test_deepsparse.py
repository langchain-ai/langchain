from langchain.transformers import DeepSparse


def test_deepsparse_inference() -> None:
    
    """Test valid gpt4all inference."""
    model = "<zoo-stub>"
    meta_agent = DeepSparse(model=model)
    predict = meta_agent('i'm a prompt')
    assert isinstance(predict, str)
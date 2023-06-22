from langchain.llms.openlm import OpenLM


def test_openlm_call() -> None:
    """Test valid call to openlm."""
    llm = OpenLM(model_name="dolly-v2-7b", max_tokens=10)
    output = llm(prompt="Say foo:")
    assert isinstance(output, str)

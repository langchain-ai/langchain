from langchain_community.llms.openlm import OpenLM


def test_openlm_call() -> None:
    """Test valid call to openlm."""
    llm = OpenLM(model_name="dolly-v2-7b", max_tokens=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

"""Test EdenAi API wrapper."""

from langchain.llms import EdenAI

FEATURE="text"
SUB_FEATURE="generation"
PARAMS={"providers": "openai","temperature" : 0.2,"max_tokens" : 250}


def test_edenai_call() -> None:
    """Test simple call to edenai."""
    llm = EdenAI(feature=FEATURE,sub_feature=SUB_FEATURE,params=PARAMS)
    output = llm("Say foo:")
    assert isinstance(output,str)


"""Test Yuan2.0 API wrapper."""
from langchain_core.outputs import LLMResult

from langchain_community.llms import Yuan2


def test_yuan2_call_method() -> None:
    """Test valid call to Yuan2.0."""
    llm = Yuan2(
        infer_api="http://127.0.0.1:8000/yuan",
        max_tokens=1024,
        temp=1.0,
        top_p=0.9,
        top_k=40,
        use_history=False,
    )
    output = llm("写一段快速排序算法。")
    assert isinstance(output, str)


def test_yuan2_generate_method() -> None:
    """Test valid call to Yuan2.0 inference api."""
    llm = Yuan2(
        infer_api="http://127.0.0.1:8000/yuan",
        max_tokens=1024,
        temp=1.0,
        top_p=0.9,
        top_k=40,
        use_history=False,
    )
    output = llm.generate(["who are you?"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)

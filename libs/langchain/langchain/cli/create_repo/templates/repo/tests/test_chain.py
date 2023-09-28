from ____project_name_identifier import get_chain


def test_my_chain() -> None:
    """Edit this test to test your chain."""
    from langchain.llms import HumanInputLLM

    llm = HumanInputLLM(input_func=lambda: "baz")
    chain = get_chain(llm)
    chain.invoke({"text": "foo"})

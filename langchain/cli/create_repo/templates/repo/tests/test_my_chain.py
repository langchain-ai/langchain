from ____project_name_identifier import MyChain

from langchain.chat_models.openai import ChatOpenAI


def test_my_chain() -> None:
    """Edit this test to test your chain."""
    llm = ChatOpenAI()
    MyChain.from_llm(llm)

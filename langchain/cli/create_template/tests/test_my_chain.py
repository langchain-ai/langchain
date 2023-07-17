from ____project_name_identifier import MyChain

from langchain.chat_models.openai import ChatOpenAI


def test_my_chain():
    llm = ChatOpenAI()
    chain = MyChain.from_llm(llm)
    """
    Edit this test to test your chain.
    """

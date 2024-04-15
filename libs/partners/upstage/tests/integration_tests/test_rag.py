from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_upstage import ChatUpstage, UpstageEmbeddings


def test_upstage_rag() -> None:
    """Test simple RAG."""

    model = ChatUpstage()

    # TODO: Do Layout Analysis

    # TODO: Embed each html tag.
    vectorstore = DocArrayInMemorySearch.from_texts(
        ["harrison worked at kensho", "bears like to eat honey"],
        embedding=UpstageEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | prompt | model | output_parser

    result = chain.invoke("What did Harrison do?")
    print(result)
    assert isinstance(result, str)
    assert len(result) > 0

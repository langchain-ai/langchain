"""Test conversation chain and memory."""
import pytest
from langchain_core.documents import Document

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.fake import FakeListLLM
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from tests.unit_tests.retrievers.sequential_retriever import SequentialRetriever


@pytest.mark.asyncio
async def atest_simple() -> None:
    fixed_resp = "I don't know"
    answer = "I know the answer!"
    llm = FakeListLLM(responses=[answer])
    retriever = SequentialRetriever(sequential_responses=[[]])
    memory = ConversationBufferMemory(
        k=1, output_key="answer", memory_key="chat_history", return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        rephrase_question=False,
        response_if_no_docs_found=fixed_resp,
        verbose=True,
    )
    got = await qa_chain.acall("What is the answer?")
    assert got["chat_history"][1].content == fixed_resp
    assert got["answer"] == fixed_resp


@pytest.mark.asyncio
async def atest_fixed_message_response_when_docs_found() -> None:
    fixed_resp = "I don't know"
    answer = "I know the answer!"
    llm = FakeListLLM(responses=[answer])
    retriever = SequentialRetriever(
        sequential_responses=[[Document(page_content=answer)]]
    )
    memory = ConversationBufferMemory(
        k=1, output_key="answer", memory_key="chat_history", return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        rephrase_question=False,
        response_if_no_docs_found=fixed_resp,
        verbose=True,
    )
    got = await qa_chain.acall("What is the answer?")
    assert got["chat_history"][1].content == answer
    assert got["answer"] == answer


def test_fixed_message_response_when_no_docs_found() -> None:
    fixed_resp = "I don't know"
    answer = "I know the answer!"
    llm = FakeListLLM(responses=[answer])
    retriever = SequentialRetriever(sequential_responses=[[]])
    memory = ConversationBufferMemory(
        k=1, output_key="answer", memory_key="chat_history", return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        rephrase_question=False,
        response_if_no_docs_found=fixed_resp,
        verbose=True,
    )
    got = qa_chain("What is the answer?")
    assert got["chat_history"][1].content == fixed_resp
    assert got["answer"] == fixed_resp


def test_no_single_execution_conversationl_retrieval() -> None:
    answer = "I know the answer!"
    llm = FakeListLLM(responses=[answer])
    doc_chain = load_qa_chain(
            llm, chain_type="stuff", prompt=QA_PROMPT
    )
    memory = ConversationBufferMemory(
        k=1, output_key="answer", memory_key="chat_history", return_messages=True
    )
    retriever = SequentialRetriever(sequential_responses=[[]])
    with pytest.raises(ValueError, match="If `question_generator` is None, "
                       "the `chat_history` variable is required"):
        ConversationalRetrievalChain(
            retriever=retriever,
            memory=memory,
            question_generator=None,
            combine_docs_chain=doc_chain
        )
    
    custom_qa_prompt_w_chathist = """\
    Given the following chat history and pieces of contexts, please answer \
    the follow up question. If you don't know the answer, just say that you \
    don't know, don't try to make up an answer.
    ---
    Chat History: {chat_history}
    ---
    Context: {context}
    ---
    Question: {question}
    Helpful Answer:"""
    CUSTOML_QA_PROMPT = PromptTemplate.from_template(
        template=custom_qa_prompt_w_chathist
    )
    custom_doc_chain = load_qa_chain(
            llm, chain_type="stuff", prompt=CUSTOML_QA_PROMPT
    )
    custom_qa_chain = ConversationalRetrievalChain(
        retriever=retriever,
        memory=memory,
        question_generator=None,
        combine_docs_chain=custom_doc_chain
    )
    got = custom_qa_chain({"question": "What is the answer?"})
    assert got["answer"] == answer


def test_fixed_message_response_when_docs_found() -> None:
    fixed_resp = "I don't know"
    answer = "I know the answer!"
    llm = FakeListLLM(responses=[answer])
    retriever = SequentialRetriever(
        sequential_responses=[[Document(page_content=answer)]]
    )
    memory = ConversationBufferMemory(
        k=1, output_key="answer", memory_key="chat_history", return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        rephrase_question=False,
        response_if_no_docs_found=fixed_resp,
        verbose=True,
    )
    got = qa_chain("What is the answer?")
    assert got["chat_history"][1].content == answer
    assert got["answer"] == answer

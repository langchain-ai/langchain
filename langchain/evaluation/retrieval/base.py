from typing import List, Optional

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.evaluation.retrieval.prompts import (
    GRADE_DOCS_PROMPT,
    GRADE_SINGLE_DOC_PROMPT,
)
from langchain.schema import Document


def grade_documents(
    documents: List[Document],
    question: str,
    llm: Optional[BaseLanguageModel] = None,
) -> List[int]:
    _llm = llm or OpenAI(temperature=0)
    if len(documents) == 1:
        return [grade_single_document(documents[0], question, llm=_llm)]
    llm_chain = LLMChain(llm=_llm, prompt=GRADE_DOCS_PROMPT)
    _documents = [
        Document(page_content=d.page_content, metadata={"i": i})
        for i, d in enumerate(documents)
    ]
    document_prompt = PromptTemplate(
        template="DOCUMENT {i}:\n{page_content}", input_variables=["i", "page_content"]
    )
    eval_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name="documents",
    )
    score, _ = eval_chain.combine_docs_and_parse(_documents, question=question)
    return score


def grade_single_document(
    document: Document,
    question: str,
    llm: Optional[BaseLanguageModel] = None,
) -> int:
    _llm = llm or OpenAI(temperature=0)
    llm_chain = LLMChain(llm=_llm, prompt=GRADE_SINGLE_DOC_PROMPT)
    eval_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="document",
    )
    score, _ = eval_chain.combine_docs_and_parse([document], question=question)
    return score[0]

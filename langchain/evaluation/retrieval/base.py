from typing import List, Optional

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.evaluation.retrieval.prompts import (
    GRADE_DOCS_PROMPT,
    GRADE_DOCS_WITH_ANSWER_PROMPT,
)
from langchain.schema import Document


def grade_documents(
    documents: List[Document],
    question: str,
    answer: Optional[str] = None,
    llm: Optional[BaseLanguageModel] = None,
) -> List[int]:
    _llm = llm or OpenAI(temperature=0)
    prompt = GRADE_DOCS_WITH_ANSWER_PROMPT if answer else GRADE_DOCS_PROMPT
    llm_chain = LLMChain(llm=_llm, prompt=prompt)
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

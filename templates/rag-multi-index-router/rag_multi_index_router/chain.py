from operator import itemgetter
from typing import Literal

from langchain.retrievers import (
    ArxivRetriever,
    KayAiRetriever,
    PubMedRetriever,
    WikipediaRetriever,
)
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_functions import (
    PydanticAttrOutputFunctionsParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RouterRunnable,
    RunnableParallel,
    RunnablePassthrough,
)

pubmed = PubMedRetriever(top_k_results=5).with_config(run_name="pubmed")
arxiv = ArxivRetriever(top_k_results=5).with_config(run_name="arxiv")
sec = KayAiRetriever.create(
    dataset_id="company", data_types=["10-K"], num_contexts=5
).with_config(run_name="sec_filings")
wiki = WikipediaRetriever(top_k_results=5, doc_content_chars_max=2000).with_config(
    run_name="wiki"
)

llm = ChatOpenAI(model="gpt-3.5-turbo")


class Search(BaseModel):
    """Search for relevant documents by question topic."""

    question_resource: Literal[
        "medical paper", "scientific paper", "public company finances report", "general"
    ] = Field(
        ...,
        description=(
            "The type of resource that would best help answer the user's question. "
            "If none of the types are relevant return 'general'."
        ),
    )


retriever_name = {
    "medical paper": "PubMed",
    "scientific paper": "ArXiv",
    "public company finances report": "SEC filings (Kay AI)",
    "general": "Wikipedia",
}

classifier = (
    llm.bind(
        functions=[convert_pydantic_to_openai_function(Search)],
        function_call={"name": "Search"},
    )
    | PydanticAttrOutputFunctionsParser(
        pydantic_schema=Search, attr_name="question_resource"
    )
    | retriever_name.get
)

retriever_map = {
    "PubMed": pubmed,
    "ArXiv": arxiv,
    "SEC filings (Kay AI)": sec,
    "Wikipedia": wiki,
}
router_retriever = RouterRunnable(runnables=retriever_map)


def format_docs(docs):
    return "\n\n".join(f"Source {i}:\n{doc.page_content}" for i, doc in enumerate(docs))


system = """Answer the user question. Use the following sources to help \
answer the question. If you don't know the answer say "I'm not sure, I couldn't \
find information on {{topic}}."

Sources:

{sources}"""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])


class Question(BaseModel):
    __root__: str


retriever_chain = (
    {"input": itemgetter("question"), "key": itemgetter("retriever_choice")}
    | router_retriever
    | format_docs
).with_config(run_name="retrieve")
answer_chain = (
    {"sources": retriever_chain, "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)
chain = (
    (
        RunnableParallel(
            question=RunnablePassthrough(), retriever_choice=classifier
        ).with_config(run_name="classify")
        | RunnablePassthrough.assign(answer=answer_chain).with_config(run_name="answer")
    )
    .with_config(run_name="QA with router")
    .with_types(input_type=Question)
)

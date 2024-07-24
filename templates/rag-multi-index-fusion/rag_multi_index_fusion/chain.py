from operator import itemgetter

import numpy as np
from langchain.retrievers import (
    ArxivRetriever,
    KayAiRetriever,
    PubMedRetriever,
    WikipediaRetriever,
)
from langchain.utils.math import cosine_similarity
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import (
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

embeddings = OpenAIEmbeddings()


def fuse_retrieved_docs(input):
    results_map = input["sources"]
    query = input["question"]
    embedded_query = embeddings.embed_query(query)
    names, docs = zip(
        *((name, doc) for name, docs in results_map.items() for doc in docs)
    )
    embedded_docs = embeddings.embed_documents([doc.page_content for doc in docs])
    similarity = cosine_similarity(
        [embedded_query],
        embedded_docs,
    )
    most_similar = np.flip(np.argsort(similarity[0]))[:5]
    return [
        (
            names[i],
            docs[i],
        )
        for i in most_similar
    ]


def format_named_docs(named_docs):
    return "\n\n".join(
        f"Source: {source}\n\n{doc.page_content}" for source, doc in named_docs
    )


system = """Answer the user question. Use the following sources to help \
answer the question. If you don't know the answer say "I'm not sure, I couldn't \
find information on {{topic}}."

Sources:

{sources}"""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])

retrieve_all = RunnableParallel(
    {"ArXiv": arxiv, "Wikipedia": wiki, "PubMed": pubmed, "SEC 10-K Forms": sec}
).with_config(run_name="retrieve_all")


class Question(BaseModel):
    __root__: str


answer_chain = (
    {
        "question": itemgetter("question"),
        "sources": lambda x: format_named_docs(x["sources"]),
    }
    | prompt
    | ChatOpenAI(model="gpt-3.5-turbo-1106")
    | StrOutputParser()
).with_config(run_name="answer")
chain = (
    (
        RunnableParallel(
            {"question": RunnablePassthrough(), "sources": retrieve_all}
        ).with_config(run_name="add_sources")
        | RunnablePassthrough.assign(sources=fuse_retrieved_docs).with_config(
            run_name="fuse"
        )
        | RunnablePassthrough.assign(answer=answer_chain).with_config(
            run_name="add_answer"
        )
    )
    .with_config(run_name="QA with fused results")
    .with_types(input_type=Question)
)

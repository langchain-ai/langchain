import pinecone
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.load import dumps, loads
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Pinecone


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


pinecone.init(api_key="...", environment="...")

prompt = hub.pull("langchain-ai/rag-fusion-query-generation")

generate_queries = (
    prompt | ChatOpenAI(temperature=0) | StrOutputParser() | (lambda x: x.split("\n"))
)

vectorstore = Pinecone.from_existing_index("rag-fusion", OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

chain = (
    {"original_query": lambda x: x}
    | generate_queries
    | retriever.map()
    | reciprocal_rank_fusion
)

import os

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Redis
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_redis.config import EMBED_MODEL, INDEX_NAME, INDEX_SCHEMA, REDIS_URL


def ingest_documents():
    """
    Ingest PDF to Redis from the data/ directory that
    contains Edgar 10k filings data for Nike.
    """
    # Load list of pdfs
    company_name = "Nike"
    data_path = "data/"
    doc = [os.path.join(data_path, file) for file in os.listdir(data_path)][0]

    print("Parsing 10k filing doc for NIKE", doc)  # noqa: T201

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100, add_start_index=True
    )
    loader = UnstructuredFileLoader(doc, mode="single", strategy="fast")
    chunks = loader.load_and_split(text_splitter)

    print("Done preprocessing. Created", len(chunks), "chunks of the original pdf")  # noqa: T201
    # Create vectorstore
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    _ = Redis.from_texts(
        # appending this little bit can sometimes help with semantic retrieval
        # especially with multiple companies
        texts=[f"Company: {company_name}. " + chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embedding=embedder,
        index_name=INDEX_NAME,
        index_schema=INDEX_SCHEMA,
        redis_url=REDIS_URL,
    )


if __name__ == "__main__":
    ingest_documents()

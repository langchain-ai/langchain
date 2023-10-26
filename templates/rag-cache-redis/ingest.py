import os

from langchain.vectorstores import Redis
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader


from rag_cache_redis.redis import (
    REDIS_URL,
    INDEX_NAME,
    INDEX_SCHEMA,
)


def ingest_documents():
    """
    Ingest PDF to Redis from the data/ directory that contains Edgar 10k filings data for Nike.
    """
    # Load list of pdfs
    data_path = "data/"
    doc = [os.path.join(data_path, file) for file in os.listdir(data_path)][0]
    company_name = "Nike"
    print("Parsing 10k filing doc for NIKE", doc)

    # For simplicity, we will just work with one of the 10k files. This will take some time still.
    # To Note: the UnstructuredFileLoader is not the only document loader type that LangChain provides
    # To Note: the RecursiveCharacterTextSplitter is what we use to create smaller chunks of text from the doc.
    # Docs: https://python.langchain.com/docs/integrations/document_loaders/unstructured_file
    # Docs: https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100, add_start_index=True
    )
    loader = UnstructuredFileLoader(doc, mode="single", strategy="fast")
    chunks = loader.load_and_split(text_splitter)

    print("Done preprocessing. Created", len(chunks), "chunks of the original pdf")
    # Create vectorstore
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    _ = Redis.from_texts(
        # appending this little bit can sometimes help with semantic retrieval -- especially with multiple companies
        texts=[f"Company: {company_name}. " + chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embedding=embedder,
        index_name=INDEX_NAME,
        schema=INDEX_SCHEMA,
        redis_url=REDIS_URL
    )


if __name__ == "__main__":
    ingest_documents()

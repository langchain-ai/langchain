import os

from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import ElasticsearchStore


# Metadata extraction function
def _metadata_func(record: dict, metadata: dict) -> dict:
    metadata["name"] = record.get("name")
    metadata["summary"] = record.get("summary")
    metadata["url"] = record.get("url")
    metadata["category"] = record.get("category")
    metadata["updated_at"] = record.get("updated_at")

    return metadata


## Load Data
def _load_documents():
    loader = JSONLoader(
        file_path="./data/documents.json",
        jq_schema=".[]",
        content_key="content",
        metadata_func=_metadata_func,
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=250)
    return text_splitter.split_documents(loader.load())


def _index_elastic(documents):
    ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
    ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME", "elastic")
    ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
    ES_URL = os.getenv("ES_URL", "http://localhost:9200")

    if ELASTIC_CLOUD_ID and ELASTIC_USERNAME and ELASTIC_PASSWORD:
        es_connection_details = {
            "es_cloud_id": ELASTIC_CLOUD_ID,
            "es_user": ELASTIC_USERNAME,
            "es_password": ELASTIC_PASSWORD,
        }
    else:
        es_connection_details = {"es_url": ES_URL}

    # Add to vectorDB
    ElasticsearchStore.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        **es_connection_details,
        index_name="workplace-search-example",
    )


def _index_chroma(documents):
    if not os.getenv("CHROMA_COLLECTION_NAME"):
        return
    from langchain.vectorstores import Chroma

    vecstore = Chroma(
        collection_name=os.environ.get("CHROMA_COLLECTION_NAME"),
        embedding_function=OpenAIEmbeddings(),
    )
    vecstore.add_documents(documents)


def _index_redis(documents):
    if not os.getenv("REDIS_INDEX_NAME"):
        return
    from langchain.vectorstores import Redis

    index_name = os.environ["REDIS_INDEX_NAME"]
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    vecstore = Redis(redis_url, index_name, OpenAIEmbeddings())
    vecstore.add_documents(documents)


def _index_pinecone(documents):
    if not os.getenv("PINECONE_INDEX_NAME"):
        return
    from langchain.vectorstores import Pinecone

    index_name = os.environ["PINECONE_INDEX_NAME"]
    vecstore = Pinecone.from_existing_index(index_name, OpenAIEmbeddings())
    vecstore.add_documents(documents)


def _index_supabase(documents):
    if not os.getenv("SUPABASE_URL"):
        return
    from langchain.vectorstores import SupabaseVectorStore
    from supabase.client import create_client

    supabase_client = create_client(
        os.environ["SUPABASE_URL"], os.getenv("SUPABASE_KEY"), OpenAIEmbeddings()
    )
    vecstore = SupabaseVectorStore(
        supabase_client, OpenAIEmbeddings(), os.getenv("SUPABASE_TABLE_NAME")
    )
    vecstore.add_documents(documents)


def _index_timescale(documents):
    if not os.getenv("TIMESCALE_SERVICE_URL"):
        return
    from langchain.vectorstores import TimescaleVector

    vecstore = TimescaleVector(
        os.environ["TIMESCALE_SERVICE_URL"],
        OpenAIEmbeddings(),
        os.getenv("TIMESCALE_COLLECTION_NAME"),
    )
    vecstore.add_documents(documents)


def _index_weaviate(documents):
    if not os.getenv("WEAVIATE_URL"):
        return
    import weaviate
    from langchain.vectorstores import Weaviate

    client = weaviate.Client(
        url=os.environ["WEAVIATE_URL"], api_key=os.getenv("WEAVIATE_API_KEY")
    )
    vecstore = Weaviate(
        client, os.getenv("WEAVIATE_INDEX_NAME"), os.getenv("WEAVIATE_TEXT_KEY", "text")
    )
    vecstore.add_documents(documents)


documents = _load_documents()
_index_elastic(documents)
_index_chroma(documents)
_index_redis(documents)
_index_pinecone(documents)
_index_supabase(documents)
_index_timescale(documents)
_index_weaviate(documents)

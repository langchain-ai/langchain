import os

from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME", "elastic")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ELASTIC_INDEX_NAME = os.getenv("ELASTIC_INDEX_NAME", "workspace-search-example")


def _metadata_func(record: dict, metadata: dict) -> dict:
    metadata["name"] = record.get("name")
    metadata["summary"] = record.get("summary")
    metadata["url"] = record.get("url")
    # give more descriptive name for metadata filtering.
    metadata["location"] = record.get("category")
    metadata["updated_at"] = record.get("updated_at")
    metadata["created_on"] = record.get("created_on")

    return metadata


loader = JSONLoader(
    file_path="./data/documents.json",
    jq_schema=".[]",
    content_key="content",
    metadata_func=_metadata_func,
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=250)
documents = text_splitter.split_documents(loader.load())

if ELASTIC_CLOUD_ID and ELASTIC_USERNAME and ELASTIC_PASSWORD:
    es_connection_details = {
        "es_cloud_id": ELASTIC_CLOUD_ID,
        "es_user": ELASTIC_USERNAME,
        "es_password": ELASTIC_PASSWORD,
    }
else:
    es_connection_details = {"es_url": ES_URL}

vecstore = ElasticsearchStore(
    ELASTIC_INDEX_NAME,
    embedding=OpenAIEmbeddings(),
    **es_connection_details,
)
vecstore.add_documents(documents)

import os
import tempfile
from datetime import datetime, timedelta

import requests
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.timescalevector import TimescaleVector
from langchain_text_splitters.character import CharacterTextSplitter
from timescale_vector import client


def parse_date(date_string: str) -> datetime:
    if date_string is None:
        return None
    time_format = "%a %b %d %H:%M:%S %Y %z"
    return datetime.strptime(date_string, time_format)


def extract_metadata(record: dict, metadata: dict) -> dict:
    dt = parse_date(record["date"])
    metadata["id"] = str(client.uuid_from_time(dt))
    if dt is not None:
        metadata["date"] = dt.isoformat()
    else:
        metadata["date"] = None
    metadata["author"] = record["author"]
    metadata["commit_hash"] = record["commit"]
    return metadata


def load_ts_git_dataset(
    service_url,
    collection_name="timescale_commits",
    num_records: int = 500,
    partition_interval=timedelta(days=7),
):
    json_url = "https://s3.amazonaws.com/assets.timescale.com/ai/ts_git_log.json"
    tmp_file = "ts_git_log.json"

    temp_dir = tempfile.gettempdir()
    json_file_path = os.path.join(temp_dir, tmp_file)

    if not os.path.exists(json_file_path):
        response = requests.get(json_url)
        if response.status_code == 200:
            with open(json_file_path, "w") as json_file:
                json_file.write(response.text)
        else:
            print(f"Failed to download JSON file. Status code: {response.status_code}")  # noqa: T201

    loader = JSONLoader(
        file_path=json_file_path,
        jq_schema=".commit_history[]",
        text_content=False,
        metadata_func=extract_metadata,
    )

    documents = loader.load()

    # Remove documents with None dates
    documents = [doc for doc in documents if doc.metadata["date"] is not None]

    if num_records > 0:
        documents = documents[:num_records]

    # Split the documents into chunks for embedding
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    # Create a Timescale Vector instance from the collection of documents
    TimescaleVector.from_documents(
        embedding=embeddings,
        ids=[doc.metadata["id"] for doc in docs],
        documents=docs,
        collection_name=collection_name,
        service_url=service_url,
        time_partition_interval=partition_interval,
    )

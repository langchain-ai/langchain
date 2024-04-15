import os

from openai import OpenAI
from opensearchpy import OpenSearch

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "admin")
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "langchain-test")

with open("dummy_data.txt") as f:
    docs = [line.strip() for line in f.readlines()]


client_oai = OpenAI(api_key=OPENAI_API_KEY)


client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
    use_ssl=True,
    verify_certs=False,
)

# Define the index settings and mappings
index_settings = {
    "settings": {
        "index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}
    },
    "mappings": {
        "properties": {
            "vector_field": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {"name": "hnsw", "space_type": "l2", "engine": "faiss"},
            }
        }
    },
}

response = client.indices.create(index=OPENSEARCH_INDEX_NAME, body=index_settings)

print(response)  # noqa: T201


# Insert docs


for each in docs:
    res = client_oai.embeddings.create(input=each, model="text-embedding-ada-002")

    document = {
        "vector_field": res.data[0].embedding,
        "text": each,
    }

    response = client.index(index=OPENSEARCH_INDEX_NAME, body=document, refresh=True)

    print(response)  # noqa: T201

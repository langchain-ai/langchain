# langchain-mongodb

# Installation
```
pip install -U langchain-mongodb
```

# Usage
- See [Getting Started with the LangChain Integration](https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/#get-started-with-the-langchain-integration) for a walkthrough on using your first LangChain implementation with MongoDB Atlas.

## Using MongoDBAtlasVectorSearch
```python
from langchain_mongodb import MongoDBAtlasVectorSearch

# Pull MongoDB Atlas URI from environment variables
MONGODB_ATLAS_CLUSTER_URI = os.environ.get("MONGODB_ATLAS_CLUSTER_URI")

DB_NAME = "langchain_db"
COLLECTION_NAME = "test"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "index_name"
MONGODB_COLLECTION = client[DB_NAME][COLLECITON_NAME]

# Create the vector search via `from_connection_string`
vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_ATLAS_CLUSTER_URI,
    DB_NAME + "." + COLLECTION_NAME,
    OpenAIEmbeddings(disallowed_special=()),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

# Initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
# Create the vector search via instantiation
vector_search_2 = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embeddings=OpenAIEmbeddings(disallowed_special=()),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)
```

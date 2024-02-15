from langchain_mongodb_atlas.vectorstores import MongoDBAtlasVectorStore


def test_initialization():
    """Test initialization of vector store class"""
    MongoDBAtlasVectorStore()

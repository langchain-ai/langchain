from langchain_mongodb_atlas.vectorstores import MongoDBAtlasVectorSearch


def test_initialization():
    """Test initialization of vector store class"""
    MongoDBAtlasVectorSearch()

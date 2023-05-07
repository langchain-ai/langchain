import pytest
import os
from langchain.document_loaders.mongodb import MongodbLoader

if "MONGODB_API_KEY" in os.environ:
    mongo_api_key = os.environ["MONGODB_API_KEY"]
else:
    mongo_api_key = False

@pytest.fixture
def mongodb_loader():
    connection_string = "mongodb://localhost:27017"
    db_name = "test_db"
    collection_name = "test_collection"

    loader = MongodbLoader(connection_string, db_name, collection_name)
    return loader

def test_mongodb_loader_init(mongodb_loader):
    assert isinstance(mongodb_loader, MongodbLoader)

def test_db_and_collection_names(mongodb_loader):
    assert mongodb_loader.db_name != ""
    assert mongodb_loader.collection_name != ""

@pytest.mark.skipif(not mongo_api_key, reason="MONGODB_API_KEY not provided.")
def test_load() -> None:
    if "MONGODB_API_KEY" in os.environ:
        mongo_api_key = os.environ["MONGODB_API_KEY"]
        db_name = os.environ["MONGODB_DB_NAME"]
        collection_name = os.environ["MONGODB_COLLECTION_NAME"]
  
        loader = MongodbLoader(mongo_api_key, db_name, collection_name)
    
        result = loader.load()

        print(result[0])
        print(len(result))
        
        assert len(result) > 0


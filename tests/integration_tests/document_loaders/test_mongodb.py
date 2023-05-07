from langchain.document_loaders.mongodb import MongodbLoader


def test_load() -> None:
    connection_string = "mongodb://localhost:27017/"
    db_name = "sample_restaurants"
    collection_name = "restaurants"

    loader = MongodbLoader(connection_string, db_name, collection_name)
   
    result = loader.load()

    print(result[0])
    print(len(result))
    
    assert len(result) > 0


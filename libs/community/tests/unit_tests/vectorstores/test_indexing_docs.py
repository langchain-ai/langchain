from langchain_core.vectorstores import VectorStore

import langchain_community.vectorstores


def test_compatible_vectorstore_documentation() -> None:
    """Test which vectorstores are compatible with the indexing API.

    This serves as a reminder to update the documentation in [1]
    that specifies which vectorstores are compatible with the
    indexing API.

    Ideally if a developer adds a new vectorstore or modifies
    an existing one in such a way that affects its compatibility
    with the Indexing API, he/she will see this failed test
    case and 1) update docs in [1] and 2) update the `documented`
    dict in this test case.

    [1] langchain/docs/docs/modules/data_connection/indexing.ipynb
    """

    # Check if a vectorstore is compatible with the indexing API
    def check_compatibility(vector_store: VectorStore) -> bool:
        """Check if a vectorstore is compatible with the indexing API."""
        methods = ["delete", "add_documents"]
        for method in methods:
            if not hasattr(vector_store, method):
                return False
        # Checking if the vectorstore has overridden the default delete method
        # implementation which just raises a NotImplementedError
        if getattr(vector_store, "delete") == VectorStore.delete:
            return False
        return True

    # Check all vector store classes for compatibility
    compatible = set()
    for class_name in langchain_community.vectorstores.__all__:
        # Get the definition of the class
        cls = getattr(langchain_community.vectorstores, class_name)

        # If the class corresponds to a vectorstore, check its compatibility
        if issubclass(cls, VectorStore):
            is_compatible = check_compatibility(cls)
            if is_compatible:
                compatible.add(class_name)

    # These are mentioned in the indexing.ipynb documentation
    documented = {
        "Aerospike",
        "AnalyticDB",
        "ApertureDB",
        "AstraDB",
        "AzureCosmosDBVectorSearch",
        "AzureCosmosDBNoSqlVectorSearch",
        "AzureSearch",
        "AwaDB",
        "Bagel",
        "BESVectorStore",
        "BigQueryVectorSearch",
        "Cassandra",
        "Chroma",
        "CouchbaseVectorStore",
        "DashVector",
        "DatabricksVectorSearch",
        "TiDBVectorStore",
        "DeepLake",
        "Dingo",
        "DocumentDBVectorSearch",
        "ElasticVectorSearch",
        "ElasticsearchStore",
        "FAISS",
        "HanaDB",
        "InMemoryVectorStore",
        "LanceDB",
        "Milvus",
        "MomentoVectorIndex",
        "MyScale",
        "OpenSearchVectorSearch",
        "OracleVS",
        "PGVector",
        "Pinecone",
        "Qdrant",
        "Redis",
        "Relyt",
        "Rockset",
        "ScaNN",
        "SemaDB",
        "SingleStoreDB",
        "SupabaseVectorStore",
        "SurrealDBStore",
        "TablestoreVectorStore",
        "TileDB",
        "TimescaleVector",
        "TencentVectorDB",
        "UpstashVectorStore",
        "EcloudESVectorStore",
        "Vald",
        "VDMS",
        "Vearch",
        "Vectara",
        "VespaStore",
        "VLite",
        "Weaviate",
        "Yellowbrick",
        "ZepVectorStore",
        "ZepCloudVectorStore",
        "Zilliz",
        "Lantern",
        "OpenSearchVectorSearch",
    }
    assert compatible == documented

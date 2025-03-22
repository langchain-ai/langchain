import React, { useState } from "react";
import CodeBlock from "@theme-original/CodeBlock";
import { CustomDropdown } from './ChatModelTabs';

export default function VectorStoreTabs(props) {
    const [selectedModel, setSelectedModel] = useState("In-memory");
    const { customVarName, useFakeEmbeddings = false } = props;

    const vectorStoreVarName = customVarName ?? "vector_store";

    const fakeEmbeddingsString = `from langchain_core.embeddings import DeterministicFakeEmbedding\n\nembeddings = DeterministicFakeEmbedding(size=100)`;

    const tabItems = [
        {
            value: "In-memory",
            label: "In-memory",
            text: `from langchain_core.vectorstores import InMemoryVectorStore\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\n${vectorStoreVarName} = InMemoryVectorStore(embeddings)`,
            packageName: "langchain-core",
            default: true,
        },
        {
            value: "AstraDB",
            label: "AstraDB",
            text: `import getpass\nfrom langchain_astradb import AstraDBVectorStore\n\nASTRA_DB_API_ENDPOINT = getpass.getpass("ASTRA_DB_API_ENDPOINT = ")\nASTRA_DB_APPLICATION_TOKEN = getpass.getpass("ASTRA_DB_APPLICATION_TOKEN = ")\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\n${vectorStoreVarName} = AstraDBVectorStore(\n    collection_name="astra_vector_langchain",\n    embedding=embeddings,\n    api_endpoint=ASTRA_DB_API_ENDPOINT,\n    token=ASTRA_DB_APPLICATION_TOKEN,\n)`,
            packageName: "langchain-astradb",
            default: false,
        },
        {
            value: "Chroma",
            label: "Chroma",
            text: `from langchain_chroma import Chroma\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\n${vectorStoreVarName} = Chroma(\n    collection_name="example_collection",\n    embedding_function=embeddings,\n    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary\n)`,
            packageName: "langchain-chroma",
            default: false,
        },
        {
            value: "FAISS",
            label: "FAISS",
            text: `import faiss\nfrom langchain_community.docstore.in_memory import InMemoryDocstore\nfrom langchain_community.vectorstores import FAISS\n\nindex = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\n${vectorStoreVarName} = FAISS(\n    embedding_function=embeddings,\n    index=index,\n    docstore=InMemoryDocstore(),\n    index_to_docstore_id={},\n)`,
            packageName: "langchain-community",
            default: false,
        },
        {
            value: "Milvus",
            label: "Milvus",
            text: `from langchain_milvus import Milvus\n\nURI = "./milvus_example.db"\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\n${vectorStoreVarName} = Milvus(\n    embedding_function=embeddings,\n    connection_args={"uri": URI},\n    index_params={"index_type": "FLAT", "metric_type": "L2"},\n)`,
            packageName: "langchain-milvus",
            default: false,
        },
        {
            value: "MongoDB",
            label: "MongoDB",
            text: `from langchain_mongodb import MongoDBAtlasVectorSearch\nfrom pymongo import MongoClient\n\n# initialize MongoDB python client\nclient = MongoClient(MONGODB_ATLAS_CLUSTER_URI)\n\nDB_NAME = "langchain_test_db"\nCOLLECTION_NAME = "langchain_test_vectorstores"\nATLAS_VECTOR_SEARCH_INDEX_NAME = "langchain-test-index-vectorstores"\n\nMONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\n${vectorStoreVarName} = MongoDBAtlasVectorSearch(\n    collection=MONGODB_COLLECTION,\n    embedding=embeddings,\n    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,\n    relevance_score_fn="cosine",\n)\n\n# Create vector search index on the collection\n# Since we are using the default OpenAI embedding model (ada-v2) we need to specify the dimensions as 1536\nvector_store.create_vector_search_index(dimensions=1536)`,
            packageName: "langchain-mongodb",
            default: false,
        },
        {
            value: "PGVector",
            label: "PGVector",
            text: `from langchain_postgres import PGVector\n\n# Use the following docker command to launch a postgres instance with pgvector enabled.\n# docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16\nconnection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!\ncollection_name = "my_docs"\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\n${vectorStoreVarName} = PGVector(\n    embeddings=embeddings,\n    collection_name=collection_name,\n    connection=connection,\n    use_jsonb=True,\n)`,
            packageName: "langchain-postgres",
            default: false,
        },
        {
            value: "Pinecone",
            label: "Pinecone",
            text: `import time\nimport os\nfrom pinecone import Pinecone, ServerlessSpec\nfrom langchain_pinecone import PineconeVectorStore\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\npc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))\n\nindex_name = "langchain-test-index"  # change if desired\n\nexisting_indexes = [index_info["name"] for index_info in pc.list_indexes()]\n\nif index_name not in existing_indexes:\n    pc.create_index(\n        name=index_name,\n        dimension=1536,\n        metric="cosine",\n        spec=ServerlessSpec(cloud="aws", region="us-east-1"),\n        deletion_protection="enabled",  # Defaults to "disabled"\n    )\n    while not pc.describe_index(index_name).status["ready"]:\n        time.sleep(1)\n\nindex = pc.Index(index_name)\n${vectorStoreVarName} = PineconeVectorStore(index=index, embedding=embeddings)`,
            packageName: "langchain-pinecone",
            default: false,
        },
        {
            value: "Qdrant",
            label: "Qdrant",
            text: `from langchain_qdrant import QdrantVectorStore\nfrom qdrant_client import QdrantClient\nfrom qdrant_client.http.models import Distance, VectorParams\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\nclient = QdrantClient(":memory:")\n\nclient.create_collection(\n    collection_name="demo_collection",\n    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),\n)\n\n${vectorStoreVarName} = QdrantVectorStore(\n    client=client,\n    collection_name="demo_collection",\n    embedding=embeddings,\n)`,
            packageName: "langchain-qdrant",
            default: false,
        },
    ];

    const modelOptions = tabItems
    .filter((item) => !item.shouldHide)
    .map((item) => ({
      value: item.value,
      label: item.label,
      text: item.text,
      packageName: item.packageName,
    }));

const selectedOption = modelOptions.find(
  (option) => option.value === selectedModel
);

return (
    <div>
      <CustomDropdown
        selectedOption={selectedOption}
        options={modelOptions}
        onSelect={setSelectedModel}
        modelType="vectorstore"
      />

      <CodeBlock language="bash">
        {`pip install -qU ${selectedOption.packageName}`}
      </CodeBlock>
      <CodeBlock language="python">
        {selectedOption.text}
      </CodeBlock>
    </div>
  );
  }

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
            text: `from langchain_astradb import AstraDBVectorStore\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\n${vectorStoreVarName} = AstraDBVectorStore(\n    embedding=embeddings,\n    api_endpoint=ASTRA_DB_API_ENDPOINT,\n    collection_name="astra_vector_langchain",\n    token=ASTRA_DB_APPLICATION_TOKEN,\n    namespace=ASTRA_DB_NAMESPACE,\n)`,
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
            text: `import faiss\nfrom langchain_community.docstore.in_memory import InMemoryDocstore\nfrom langchain_community.vectorstores import FAISS\n\nembedding_dim = len(embeddings.embed_query("hello world"))\nindex = faiss.IndexFlatL2(embedding_dim)\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\n${vectorStoreVarName} = FAISS(\n    embedding_function=embeddings,\n    index=index,\n    docstore=InMemoryDocstore(),\n    index_to_docstore_id={},\n)`,
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
            text: `from langchain_mongodb import MongoDBAtlasVectorSearch\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\n${vectorStoreVarName} = MongoDBAtlasVectorSearch(\n    embedding=embeddings,\n    collection=MONGODB_COLLECTION,\n    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,\n    relevance_score_fn="cosine",\n)`,
            packageName: "langchain-mongodb",
            default: false,
        },
        {
            value: "PGVector",
            label: "PGVector",
            text: `from langchain_postgres import PGVector\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\n${vectorStoreVarName} = PGVector(\n    embeddings=embeddings,\n    collection_name="my_docs",\n    connection="postgresql+psycopg://...",\n)`,
            packageName: "langchain-postgres",
            default: false,
        },
        {
          value: "PGVectorStore",
          label: "PGVectorStore",
          text: `from langchain_postgres import PGEngine, PGVectorStore\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\n$engine = PGEngine.from_connection_string(\n    url="postgresql+psycopg://..."\n)\n\n${vectorStoreVarName} = PGVectorStore.create_sync(\n    engine=pg_engine,\n    table_name='test_table',\n    embedding_service=embedding\n)`,
          packageName: "langchain-postgres",
          default: false,
      },
        {
            value: "Pinecone",
            label: "Pinecone",
            text: `from langchain_pinecone import PineconeVectorStore\nfrom pinecone import Pinecone\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\npc = Pinecone(api_key=...)\nindex = pc.Index(index_name)\n\n${vectorStoreVarName} = PineconeVectorStore(embedding=embeddings, index=index)`,
            packageName: "langchain-pinecone",
            default: false,
        },
        {
            value: "Qdrant",
            label: "Qdrant",
            text: `from qdrant_client.models import Distance, VectorParams\nfrom langchain_qdrant import QdrantVectorStore\nfrom qdrant_client import QdrantClient\n${useFakeEmbeddings ? fakeEmbeddingsString : ""}\nclient = QdrantClient(":memory:")\n\nvector_size = len(embeddings.embed_query("sample text"))\n\nif not client.collection_exists("test"):\n    client.create_collection(\n        collection_name="test",\n        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)\n    )\n${vectorStoreVarName} = QdrantVectorStore(\n    client=client,\n    collection_name="test",\n    embedding=embeddings,\n)`,
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

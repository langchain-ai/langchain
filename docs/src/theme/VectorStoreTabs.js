import React from "react";
import Tabs from "@theme/Tabs";
import TabItem from "@theme/TabItem";
import CodeBlock from "@theme-original/CodeBlock";

export default function VectorStoreTabs(props) {
    const { customVarName } = props;

    const vectorStoreVarName = customVarName ?? "vector_store";

    const tabItems = [
        {
            value: "In-memory",
            label: "In-memory",
            text: `from langchain_core.vectorstores import InMemoryVectorStore\n\n${vectorStoreVarName} = InMemoryVectorStore(embeddings)`,
            packageName: "langchain-core",
            default: true,
        },
        {
            value: "AstraDB",
            label: "AstraDB",
            text: `from langchain_astradb import AstraDBVectorStore\n\n${vectorStoreVarName} = AstraDBVectorStore(\n    embedding=embeddings,\n    api_endpoint=ASTRA_DB_API_ENDPOINT,\n    collection_name="astra_vector_langchain",\n    token=ASTRA_DB_APPLICATION_TOKEN,\n    namespace=ASTRA_DB_NAMESPACE,\n)`,
            packageName: "langchain-astradb",
            default: false,
        },
        {
            value: "Chroma",
            label: "Chroma",
            text: `from langchain_chroma import Chroma\n\n${vectorStoreVarName} = Chroma(embedding_function=embeddings)`,
            packageName: "langchain-chroma",
            default: false,
        },
        {
            value: "FAISS",
            label: "FAISS",
            text: `from langchain_community.vectorstores import FAISS\n\n${vectorStoreVarName} = FAISS(embedding_function=embeddings)`,
            packageName: "langchain-community",
            default: false,
        },
        {
            value: "Milvus",
            label: "Milvus",
            text: `from langchain_milvus import Milvus\n\n${vectorStoreVarName} = Milvus(embedding_function=embeddings)`,
            packageName: "langchain-milvus",
            default: false,
        },
        {
            value: "MongoDB",
            label: "MongoDB",
            text: `from langchain_mongodb import MongoDBAtlasVectorSearch\n\n${vectorStoreVarName} = MongoDBAtlasVectorSearch(\n    embedding=embeddings,\n    collection=MONGODB_COLLECTION,\n    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,\n    relevance_score_fn="cosine",\n)`,
            packageName: "langchain-mongodb",
            default: false,
        },
        {
            value: "PGVector",
            label: "PGVector",
            text: `from langchain_postgres import PGVector\n\n${vectorStoreVarName} = PGVector(\n    embedding=embeddings,\n    collection_name="my_docs",\n    connection="postgresql+psycopg://...",\n)`,
            packageName: "langchain-postgres",
            default: false,
        },
        {
            value: "Pinecone",
            label: "Pinecone",
            text: `from langchain_pinecone import PineconeVectorStore\nfrom pinecone import Pinecone\n\npc = Pinecone(api_key=...)\nindex = pc.Index(index_name)\n\n${vectorStoreVarName} = PineconeVectorStore(embedding=embeddings, index=index)`,
            packageName: "langchain-pinecone",
            default: false,
        },
        {
            value: "Qdrant",
            label: "Qdrant",
            text: `from langchain_qdrant import QdrantVectorStore\nfrom qdrant_client import QdrantClient\n\nclient = QdrantClient(":memory:")\n${vectorStoreVarName} = QdrantVectorStore(\n    client=client,\n    collection_name="test",\n    embedding=embeddings,\n)`,
            packageName: "langchain-qdrant",
            default: false,
        },
    ];

    return (
        <Tabs groupId="vectorStoreTabs">
            {tabItems.map((tabItem) => (
                <TabItem
                    key={tabItem.value}
                    value={tabItem.value}
                    label={tabItem.label}
                    default={tabItem.default}
                >
                    <CodeBlock language="bash">{`pip install -qU ${tabItem.packageName}`}</CodeBlock>
                    <CodeBlock language="python">{tabItem.text}</CodeBlock>
                </TabItem>
            ))}
        </Tabs>
    );
}

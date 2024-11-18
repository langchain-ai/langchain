import React, { useState } from "react";
import EmbeddingTabs from "@theme/EmbeddingTabs";
import VectorStoreTabs from "@theme/VectorStoreTabs";
import CodeBlock from "@theme-original/CodeBlock";

export default function EmbeddingVectorStoreTabs() {
    const [selectedEmbedding, setSelectedEmbedding] = useState("OpenAI");
    const [selectedVectorStore, setSelectedVectorStore] = useState("Chroma");

    const handleEmbeddingChange = (value) => {
        setSelectedEmbedding(value);
    };

    const handleVectorStoreChange = (value) => {
        setSelectedVectorStore(value);
    };

    return (
        <div>
            <h3>Select embeddings:</h3>
            <EmbeddingTabs onChange={handleEmbeddingChange} />

            <h3>Select vector store:</h3>
            <VectorStoreTabs onChange={handleVectorStoreChange} />

            <h3>Initialize vector store with embeddings:</h3>
            <CodeBlock language="python">
                {`# Example initialization with selected embedding and vector store
# Embedding: ${selectedEmbedding}
# Vector Store: ${selectedVectorStore}
# Initialize your vector store here with the selected embedding
`}
            </CodeBlock>
        </div>
    );
}

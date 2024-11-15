import React, { useState } from "react";
import ChatModelTabs from "@theme/ChatModelTabs";
import EmbeddingTabs from "@theme/EmbeddingTabs";
import CodeBlock from "@theme-original/CodeBlock";

export default function CombinedTabs() {
  const [selectedChatModel, setSelectedChatModel] = useState("OpenAI");
  const [selectedEmbedding, setSelectedEmbedding] = useState("OpenAI");

  const handleChatModelChange = (value) => {
    setSelectedChatModel(value);
  };

  const handleEmbeddingChange = (value) => {
    setSelectedEmbedding(value);
  };

  const getChatModelCode = () => {
    switch (selectedChatModel) {
      case "OpenAI":
        return `from langchain_openai import ChatOpenAI\n\nmodel = ChatOpenAI(model="gpt-4o-mini")`;
      case "Anthropic":
        return `from langchain_anthropic import ChatAnthropic\n\nmodel = ChatAnthropic(model="claude-3-5-sonnet-20240620")`;
      case "Google":
        return `from langchain_google_vertexai import ChatVertexAI\n\nmodel = ChatVertexAI(model="gemini-1.5-flash")`;
      default:
        return "";
    }
  };

  const getEmbeddingCode = () => {
    switch (selectedEmbedding) {
      case "OpenAI":
        return `from langchain_openai import OpenAIEmbeddings\n\nembeddings = OpenAIEmbeddings(model="text-embedding-3-large")`;
      case "HuggingFace":
        return `from langchain_huggingface import HuggingFaceEmbeddings\n\nembeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")`;
      case "Fake Embedding":
        return `from langchain_core.embeddings import FakeEmbeddings\n\nembeddings = FakeEmbeddings(size=4096)`;
      default:
        return "";
    }
  };

  return (
    <div>
      <ChatModelTabs onChange={handleChatModelChange} />
      <EmbeddingTabs onChange={handleEmbeddingChange} />
      <CodeBlock language="python">
        {`${getChatModelCode()}\n\n${getEmbeddingCode()}`}
      </CodeBlock>
    </div>
  );
}
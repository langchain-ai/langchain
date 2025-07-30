import React, { useState } from "react";
import CodeBlock from "@theme-original/CodeBlock";
import { CustomDropdown } from './ChatModelTabs';

export default function EmbeddingTabs(props) {
  const [selectedModel, setSelectedModel] = useState("OpenAI");
  const {
    openaiParams,
    hideOpenai,
    azureOpenaiParams,
    hideAzureOpenai,
    googleGenAIParams,
    hideGoogleGenAI,
    googleVertexAIParams,
    hideGoogleVertexAI,
    awsParams,
    hideAws,
    huggingFaceParams,
    hideHuggingFace,
    ollamaParams,
    hideOllama,
    cohereParams,
    hideCohere,
    mistralParams,
    hideMistral,
    nomicParams,
    hideNomic,
    nvidiaParams,
    hideNvidia,
    voyageaiParams,
    hideVoyageai,
    ibmParams,
    hideIBM,
    fakeEmbeddingParams,
    hideFakeEmbedding,
    customVarName,
  } = props;

  const openAIParamsOrDefault = openaiParams ?? `model="text-embedding-3-large"`;
  const azureParamsOrDefault =
    azureOpenaiParams ??
    `\n    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],\n    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],\n    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],\n`;
  const googleGenAIParamsOrDefault = googleGenAIParams ?? `model="models/gemini-embedding-001"`;
  const googleVertexAIParamsOrDefault = googleVertexAIParams ?? `model="text-embedding-005"`;
  const awsParamsOrDefault = awsParams ?? `model_id="amazon.titan-embed-text-v2:0"`;
  const huggingFaceParamsOrDefault = huggingFaceParams ?? `model_name="sentence-transformers/all-mpnet-base-v2"`;
  const ollamaParamsOrDefault = ollamaParams ?? `model="llama3"`;
  const cohereParamsOrDefault = cohereParams ?? `model="embed-english-v3.0"`;
  const mistralParamsOrDefault = mistralParams ?? `model="mistral-embed"`;
  const nomicsParamsOrDefault = nomicParams ?? `model="nomic-embed-text-v1.5"`;
  const nvidiaParamsOrDefault = nvidiaParams ?? `model="NV-Embed-QA"`;
  const voyageaiParamsOrDefault = voyageaiParams ?? `model="voyage-3"`;
  const ibmParamsOrDefault = ibmParams ??
    `\n    model_id="ibm/slate-125m-english-rtrvr",\n    url="https://us-south.ml.cloud.ibm.com",\n    project_id="<WATSONX PROJECT_ID>",\n`;
  const fakeEmbeddingParamsOrDefault = fakeEmbeddingParams ?? `size=4096`;

  const embeddingVarName = customVarName ?? "embeddings";

  const tabItems = [
    {
      value: "OpenAI",
      label: "OpenAI",
      text: `from langchain_openai import OpenAIEmbeddings\n\n${embeddingVarName} = OpenAIEmbeddings(${openAIParamsOrDefault})`,
      apiKeyName: "OPENAI_API_KEY",
      packageName: "langchain-openai",
      default: true,
      shouldHide: hideOpenai,
    },
    {
      value: "Azure",
      label: "Azure",
      text: `from langchain_openai import AzureOpenAIEmbeddings\n\n${embeddingVarName} = AzureOpenAIEmbeddings(${azureParamsOrDefault})`,
      apiKeyName: "AZURE_OPENAI_API_KEY",
      packageName: "langchain-openai",
      default: false,
      shouldHide: hideAzureOpenai,
    },
    {
      value: "GoogleGenAI",
      label: "Google Gemini",
      text: `from langchain_google_genai import GoogleGenerativeAIEmbeddings\n\n${embeddingVarName} = GoogleGenerativeAIEmbeddings(${googleGenAIParamsOrDefault})`,
      apiKeyName: "GOOGLE_API_KEY",
      packageName: "langchain-google-genai",
      default: false,
      shouldHide: hideGoogleGenAI,
    },
    {
      value: "GoogleVertexAI",
      label: "Google Vertex",
      text: `from langchain_google_vertexai import VertexAIEmbeddings\n\n${embeddingVarName} = VertexAIEmbeddings(${googleVertexAIParamsOrDefault})`,
      apiKeyName: undefined,
      packageName: "langchain-google-vertexai",
      default: false,
      shouldHide: hideGoogleVertexAI,
    },
    {
      value: "AWS",
      label: "AWS",
      text: `from langchain_aws import BedrockEmbeddings\n\n${embeddingVarName} = BedrockEmbeddings(${awsParamsOrDefault})`,
      apiKeyName: undefined,
      packageName: "langchain-aws",
      default: false,
      shouldHide: hideAws,
    },
    {
      value: "HuggingFace",
      label: "HuggingFace",
      text: `from langchain_huggingface import HuggingFaceEmbeddings\n\n${embeddingVarName} = HuggingFaceEmbeddings(${huggingFaceParamsOrDefault})`,
      apiKeyName: undefined,
      packageName: "langchain-huggingface",
      default: false,
      shouldHide: hideHuggingFace,
    },
    {
      value: "Ollama",
      label: "Ollama",
      text: `from langchain_ollama import OllamaEmbeddings\n\n${embeddingVarName} = OllamaEmbeddings(${ollamaParamsOrDefault})`,
      apiKeyName: undefined,
      packageName: "langchain-ollama",
      default: false,
      shouldHide: hideOllama,
    },
    {
      value: "Cohere",
      label: "Cohere",
      text: `from langchain_cohere import CohereEmbeddings\n\n${embeddingVarName} = CohereEmbeddings(${cohereParamsOrDefault})`,
      apiKeyName: "COHERE_API_KEY",
      packageName: "langchain-cohere",
      default: false,
      shouldHide: hideCohere,
    },
    {
      value: "MistralAI",
      label: "MistralAI",
      text: `from langchain_mistralai import MistralAIEmbeddings\n\n${embeddingVarName} = MistralAIEmbeddings(${mistralParamsOrDefault})`,
      apiKeyName: "MISTRALAI_API_KEY",
      packageName: "langchain-mistralai",
      default: false,
      shouldHide: hideMistral,
    },
    {
      value: "Nomic",
      label: "Nomic",
      text: `from langchain_nomic import NomicEmbeddings\n\n${embeddingVarName} = NomicEmbeddings(${nomicsParamsOrDefault})`,
      apiKeyName: "NOMIC_API_KEY",
      packageName: "langchain-nomic",
      default: false,
      shouldHide: hideNomic,
    },
    {
      value: "NVIDIA",
      label: "NVIDIA",
      text: `from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings\n\n${embeddingVarName} = NVIDIAEmbeddings(${nvidiaParamsOrDefault})`,
      apiKeyName: "NVIDIA_API_KEY",
      packageName: "langchain-nvidia-ai-endpoints",
      default: false,
      shouldHide: hideNvidia,
    },
    {
      value: "Voyage AI",
      label: "Voyage AI",
      text: `from langchain-voyageai import VoyageAIEmbeddings\n\n${embeddingVarName} = VoyageAIEmbeddings(${voyageaiParamsOrDefault})`,
      apiKeyName: "VOYAGE_API_KEY",
      packageName: "langchain-voyageai",
      default: false,
      shouldHide: hideVoyageai,
    },
    {
      value: "IBM",
      label: "IBM watsonx",
      text: `from langchain_ibm import WatsonxEmbeddings\n\n${embeddingVarName} = WatsonxEmbeddings(${ibmParamsOrDefault})`,
      apiKeyName: "WATSONX_APIKEY",
      packageName: "langchain-ibm",
      default: false,
      shouldHide: hideIBM,
    },
    {
      value: "Fake",
      label: "Fake",
      text: `from langchain_core.embeddings import DeterministicFakeEmbedding\n\n${embeddingVarName} = DeterministicFakeEmbedding(${fakeEmbeddingParamsOrDefault})`,
      apiKeyName: undefined,
      packageName: "langchain-core",
      default: false,
      shouldHide: hideFakeEmbedding,
    },
  ];

  const modelOptions = tabItems
    .filter((item) => !item.shouldHide)
    .map((item) => ({
      value: item.value,
      label: item.label,
      text: item.text,
      apiKeyName: item.apiKeyName,
      apiKeyText: item.apiKeyText,
      packageName: item.packageName,
    }));

  const selectedOption = modelOptions.find(
    (option) => option.value === selectedModel
  );

  let apiKeyText = "";
  if (selectedOption.apiKeyName) {
    apiKeyText = `import getpass
import os

if not os.environ.get("${selectedOption.apiKeyName}"):
  os.environ["${selectedOption.apiKeyName}"] = getpass.getpass("Enter API key for ${selectedOption.label}: ")`;
  } else if (selectedOption.apiKeyText) {
    apiKeyText = selectedOption.apiKeyText;
  }

  return (
    <div>
      <CustomDropdown
        selectedOption={selectedOption}
        options={modelOptions}
        onSelect={setSelectedModel}
        modelType="embeddings"
      />

      <CodeBlock language="bash">
        {`pip install -qU ${selectedOption.packageName}`}
      </CodeBlock>
      <CodeBlock language="python">
        {apiKeyText ? apiKeyText + "\n\n" + selectedOption.text : selectedOption.text}
      </CodeBlock>
    </div>
  );
}

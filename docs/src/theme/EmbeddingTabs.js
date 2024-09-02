import React from "react";
import Tabs from "@theme/Tabs";
import TabItem from "@theme/TabItem";
import CodeBlock from "@theme-original/CodeBlock";

export default function EmbeddingTabs(props) {
    const {
      openaiParams,
      hideOpenai,
      huggingFaceParams,
      hideHuggingFace,
      fakeEmbeddingParams,
      hideFakeEmbedding,
      customVarName,
    } = props;
  
    const openAIParamsOrDefault = openaiParams ?? `model="text-embedding-3-large"`;
    const huggingFaceParamsOrDefault = huggingFaceParams ?? `model="sentence-transformers/all-mpnet-base-v2"`;
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
        value: "HuggingFace",
        label: "HuggingFace",
        text: `from langchain_huggingface import HuggingFaceEmbeddings\n\n${embeddingVarName} = HuggingFaceEmbeddings(${huggingFaceParamsOrDefault})`,
        apiKeyName: undefined,
        packageName: "langchain-huggingface",
        default: false,
        shouldHide: hideHuggingFace,
      },
      {
        value: "Fake Embedding",
        label: "Fake Embedding",
        text: `from langchain_core.embeddings import FakeEmbeddings\n\n${embeddingVarName} = FakeEmbeddings(${fakeEmbeddingParamsOrDefault})`,
        apiKeyName: undefined,
        packageName: "langchain-core",
        default: false,
        shouldHide: hideFakeEmbedding,
      },
    ];
  
    return (
        <Tabs groupId="modelTabs">
            {tabItems
                .filter((tabItem) => !tabItem.shouldHide)
                .map((tabItem) => {
                    const apiKeyText = tabItem.apiKeyName ? `import getpass

    os.environ["${tabItem.apiKeyName}"] = getpass.getpass()` : '';
                    return (
                        <TabItem
                            value={tabItem.value}
                            label={tabItem.label}
                            default={tabItem.default}
                        >
                            <CodeBlock language="bash">{`pip install -qU ${tabItem.packageName}`}</CodeBlock>              
                            <CodeBlock language="python">{apiKeyText + (apiKeyText ? "\n\n" : '') + tabItem.text}</CodeBlock>
                        </TabItem>
                    );
                })
            }
        </Tabs>
    );
  }
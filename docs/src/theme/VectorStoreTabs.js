import React from "react";
import Tabs from "@theme/Tabs";
import TabItem from "@theme/TabItem";
import CodeBlock from "@theme-original/CodeBlock";

export default function VectorStoreTabs(props) {
    const { customVarName } = props;

    const vectorStoreVarName = customVarName ?? "vector_store";

    const tabItems = [
        {
            value: "Chroma",
            label: "Chroma",
            text: `${vectorStoreVarName} = Chroma(embedding_function=embeddings)`,
            packageName: "langchain-chroma",
            default: true,
        },
        {
            value: "FAISS",
            label: "FAISS",
            text: `${vectorStoreVarName} = FAISS(embedding_function=embeddings)`,
            packageName: "langchain-faiss",
            default: false,
        },
        // Add more vector stores as needed
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

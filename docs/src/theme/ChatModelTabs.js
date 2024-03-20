/* eslint-disable react/jsx-props-no-spreading */
import React from "react";
import Tabs from "@theme/Tabs";
import TabItem from "@theme/TabItem";
import CodeBlock from "@theme-original/CodeBlock";

function Setup({ apiKeyName, packageName }) {
  const apiKeyText = `import getpass
import os

os.environ["${apiKeyName}"] = getpass.getpass()`;
  return (
    <>
      <h5>Install dependencies</h5>
      <CodeBlock language="bash">{`pip install -qU ${packageName}`}</CodeBlock>
      <h5>Set environment variables</h5>
      <CodeBlock language="python">{apiKeyText}</CodeBlock>
    </>
  );
}

/**
 * @param {{ openaiParams?: string, anthropicParams?: string, fireworksParams?: string, mistralParams?: string, googleParams?: string, hideOpenai?: boolean, hideAnthropic?: boolean, hideFireworks?: boolean, hideMistral?: boolean, hideGoogle?: boolean }} props
 */
export default function ChatModelTabs(props) {
  const {
    openaiParams,
    anthropicParams,
    fireworksParams,
    mistralParams,
    googleParams,
    hideOpenai,
    hideAnthropic,
    hideFireworks,
    hideMistral,
    hideGoogle,
  } = props;

  // OpenAI
  const openAIText = `from langchain_openai import ChatOpenAI
  
model = ChatOpenAI(${openaiParams ?? ""})`;
  const openAIProps = { value: "OpenAI", label: "OpenAI", default: true };

  // Anthropic
  const anthropicText = `from langchain_anthropic import ChatAnthropic
  
model = ChatAnthropic(${anthropicParams ?? ""})`;
  const anthropicProps = { value: "Anthropic", label: "Anthropic" };

  // FireworksAI
  const fireworksText = `from langchain_fireworks import ChatFireworks
  
model = ChatFireworks(${fireworksParams ?? ""})`;
  const fireworksProps = { value: "FireworksAI", label: "FireworksAI" };

  // MistralAI
  const mistralText = `from langchain_mistralai import ChatMistralAI
  
model = ChatMistralAI(${mistralParams ?? ""})`;
  const mistralProps = { value: "MistralAI", label: "MistralAI" };

  // Google
  const googleText = `from langchain_google_genai import ChatGoogleGenerativeAI
  
model = ChatGoogleGenerativeAI(${googleParams ?? ""})`;
  const googleProps = { value: "Google", label: "Google" };

  return (
    <Tabs groupId="modelTabs">
      {hideOpenai ? null : (
        <TabItem {...openAIProps}>
          <Setup apiKeyName="OPENAI_API_KEY" packageName="langchain-openai" />
          <CodeBlock language="python">{openAIText}</CodeBlock>
        </TabItem>
      )}
      {hideAnthropic ? null : (
        <TabItem {...anthropicProps}>
          <Setup
            apiKeyName="ANTHROPIC_API_KEY"
            packageName="langchain-anthropic"
          />
          <CodeBlock language="python">{anthropicText}</CodeBlock>
        </TabItem>
      )}
      {hideFireworks ? null : (
        <TabItem {...fireworksProps}>
          <Setup
            apiKeyName="FIREWORKS_API_KEY"
            packageName="langchain-fireworks"
          />
          <CodeBlock language="python">{fireworksText}</CodeBlock>
        </TabItem>
      )}
      {hideMistral ? null : (
        <TabItem {...mistralProps}>
          <Setup
            apiKeyName="MISTRAL_API_KEY"
            packageName="langchain-mistralai"
          />
          <CodeBlock language="python">{mistralText}</CodeBlock>
        </TabItem>
      )}
      {hideGoogle ? null : (
        <TabItem {...googleProps}>
          <Setup
            apiKeyName="GOOGLE_API_KEY"
            packageName="langchain-google-genai"
          />
          <CodeBlock language="python">{googleText}</CodeBlock>
        </TabItem>
      )}
    </Tabs>
  );
}

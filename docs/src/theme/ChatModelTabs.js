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

  const openAIParamsOrDefault = openaiParams ?? `model="gpt-3.5-turbo-0125"`
  const anthropicParamsOrDefault = anthropicParams ?? `model="claude-3-sonnet-20240229"`
  const fireworksParamsOrDefault = fireworksParams ?? `model="accounts/fireworks/models/mixtral-8x7b-instruct"`
  const mistralParamsOrDefault = mistralParams ?? `model="mistral-large-latest"`
  const googleParamsOrDefault = googleParams ?? `model="gemini-pro"`

  const tabItems = [
    {
      value: "OpenAI",
      label: "OpenAI",
      text: `from langchain_openai import ChatOpenAI\n\nmodel = ChatOpenAI(${openAIParamsOrDefault})`,
      apiKeyName: "OPENAI_API_KEY",
      packageName: "langchain-openai",
      default: true,
      shouldHide: hideOpenai,
    },
    {
      value: "Anthropic",
      label: "Anthropic",
      text: `from langchain_anthropic import ChatAnthropic\n\nmodel = ChatAnthropic(${anthropicParamsOrDefault})`,
      apiKeyName: "ANTHROPIC_API_KEY",
      packageName: "langchain-anthropic",
      default: false,
      shouldHide: hideAnthropic,
    },
    {
      value: "FireworksAI",
      label: "FireworksAI",
      text: `from langchain_fireworks import ChatFireworks\n\nmodel = ChatFireworks(${fireworksParamsOrDefault})`,
      apiKeyName: "FIREWORKS_API_KEY",
      packageName: "langchain-fireworks",
      default: false,
      shouldHide: hideFireworks,
    },
    {
      value: "MistralAI",
      label: "MistralAI",
      text: `from langchain_mistralai import ChatMistralAI\n\nmodel = ChatMistralAI(${mistralParamsOrDefault})`,
      apiKeyName: "MISTRAL_API_KEY",
      packageName: "langchain-mistralai",
      default: false,
      shouldHide: hideMistral,
    },
    {
      value: "Google",
      label: "Google",
      text: `from langchain_google_genai import ChatGoogleGenerativeAI\n\nmodel = ChatGoogleGenerativeAI(${googleParamsOrDefault})`,
      apiKeyName: "GOOGLE_API_KEY",
      packageName: "langchain-google-genai",
      default: false,
      shouldHide: hideGoogle,
    }
  ]

  return (
    <Tabs groupId="modelTabs">
      {tabItems.filter((tabItem) => !tabItem.shouldHide).map((tabItem) => (
        <TabItem value={tabItem.value} label={tabItem.label} default={tabItem.default}>
          <Setup apiKeyName={tabItem.apiKeyName} packageName={tabItem.packageName} />
          <CodeBlock language="python">{tabItem.text}</CodeBlock>
        </TabItem>
      ))}
    </Tabs>
  );
}

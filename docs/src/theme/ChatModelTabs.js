/* eslint-disable react/jsx-props-no-spreading, react/destructuring-assignment */
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
 * @typedef {Object} ChatModelTabsProps - Component props.
 * @property {string} [openaiParams] - Parameters for OpenAI chat model. Defaults to `model="gpt-3.5-turbo-0125"`
 * @property {string} [anthropicParams] - Parameters for Anthropic chat model. Defaults to `model="claude-3-sonnet-20240229"`
 * @property {string} [fireworksParams] - Parameters for Fireworks chat model. Defaults to `model="accounts/fireworks/models/mixtral-8x7b-instruct"`
 * @property {string} [mistralParams] - Parameters for Mistral chat model. Defaults to `model="mistral-large-latest"`
 * @property {string} [googleParams] - Parameters for Google chat model. Defaults to `model="gemini-pro"`
 * @property {string} [togetherParams] - Parameters for Together chat model. Defaults to `model="mistralai/Mixtral-8x7B-Instruct-v0.1"`
 * @property {boolean} [hideOpenai] - Whether or not to hide OpenAI chat model.
 * @property {boolean} [hideAnthropic] - Whether or not to hide Anthropic chat model.
 * @property {boolean} [hideFireworks] - Whether or not to hide Fireworks chat model.
 * @property {boolean} [hideMistral] - Whether or not to hide Mistral chat model.
 * @property {boolean} [hideGoogle] - Whether or not to hide Google chat model.
 * @property {boolean} [hideTogether] - Whether or not to hide Together chat model.
 * @property {string} [customVarName] - Custom variable name for the model. Defaults to `model`.
 */

/**
 * @param {ChatModelTabsProps} props - Component props.
 */
export default function ChatModelTabs(props) {
  const {
    openaiParams,
    anthropicParams,
    fireworksParams,
    mistralParams,
    googleParams,
    togetherParams,
    hideOpenai,
    hideAnthropic,
    hideFireworks,
    hideMistral,
    hideGoogle,
    hideTogether,
    customVarName,
  } = props;

  const openAIParamsOrDefault = openaiParams ?? `model="gpt-3.5-turbo-0125"`;
  const anthropicParamsOrDefault =
    anthropicParams ?? `model="claude-3-sonnet-20240229"`;
  const fireworksParamsOrDefault =
    fireworksParams ??
    `model="accounts/fireworks/models/mixtral-8x7b-instruct"`;
  const mistralParamsOrDefault =
    mistralParams ?? `model="mistral-large-latest"`;
  const googleParamsOrDefault = googleParams ?? `model="gemini-pro"`;
  const togetherParamsOrDefault =
    togetherParams ??
    `\n    base_url="https://api.together.xyz/v1",\n    api_key=os.environ["TOGETHER_API_KEY"],\n    model="mistralai/Mixtral-8x7B-Instruct-v0.1",`;

  const llmVarName = customVarName ?? "model";

  const tabItems = [
    {
      value: "OpenAI",
      label: "OpenAI",
      text: `from langchain_openai import ChatOpenAI\n\n${llmVarName} = ChatOpenAI(${openAIParamsOrDefault})`,
      apiKeyName: "OPENAI_API_KEY",
      packageName: "langchain-openai",
      default: true,
      shouldHide: hideOpenai,
    },
    {
      value: "Anthropic",
      label: "Anthropic",
      text: `from langchain_anthropic import ChatAnthropic\n\n${llmVarName} = ChatAnthropic(${anthropicParamsOrDefault})`,
      apiKeyName: "ANTHROPIC_API_KEY",
      packageName: "langchain-anthropic",
      default: false,
      shouldHide: hideAnthropic,
    },
    {
      value: "FireworksAI",
      label: "FireworksAI",
      text: `from langchain_fireworks import ChatFireworks\n\n${llmVarName} = ChatFireworks(${fireworksParamsOrDefault})`,
      apiKeyName: "FIREWORKS_API_KEY",
      packageName: "langchain-fireworks",
      default: false,
      shouldHide: hideFireworks,
    },
    {
      value: "MistralAI",
      label: "MistralAI",
      text: `from langchain_mistralai import ChatMistralAI\n\n${llmVarName} = ChatMistralAI(${mistralParamsOrDefault})`,
      apiKeyName: "MISTRAL_API_KEY",
      packageName: "langchain-mistralai",
      default: false,
      shouldHide: hideMistral,
    },
    {
      value: "Google",
      label: "Google",
      text: `from langchain_google_genai import ChatGoogleGenerativeAI\n\n${llmVarName} = ChatGoogleGenerativeAI(${googleParamsOrDefault})`,
      apiKeyName: "GOOGLE_API_KEY",
      packageName: "langchain-google-genai",
      default: false,
      shouldHide: hideGoogle,
    },
    {
      value: "TogetherAI",
      label: "TogetherAI",
      text: `from langchain_openai import ChatOpenAI\n\n${llmVarName} = Together(${togetherParamsOrDefault})`,
      apiKeyName: "TOGETHER_API_KEY",
      packageName: "langchain-openai",
      default: false,
      shouldHide: hideTogether,
    },
  ];

  return (
    <Tabs groupId="modelTabs">
      {tabItems
        .filter((tabItem) => !tabItem.shouldHide)
        .map((tabItem) => (
          <TabItem
            value={tabItem.value}
            label={tabItem.label}
            default={tabItem.default}
          >
            <Setup
              apiKeyName={tabItem.apiKeyName}
              packageName={tabItem.packageName}
            />
            <CodeBlock language="python">{tabItem.text}</CodeBlock>
          </TabItem>
        ))}
    </Tabs>
  );
}

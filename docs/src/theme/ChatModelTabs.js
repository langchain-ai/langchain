/* eslint-disable react/jsx-props-no-spreading, react/destructuring-assignment */
import React, { useState } from "react";
import CodeBlock from "@theme-original/CodeBlock";

// Create a custom dropdown since Docusaurus's dropdown component isn't easily accessible
export const CustomDropdown = ({ selectedOption, options, onSelect, modelType }) => {
  const [isOpen, setIsOpen] = React.useState(false);

  // Close dropdown when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event) => {
      if (isOpen && !event.target.closest('.dropdown')) {
        setIsOpen(false);
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, [isOpen]);

  // Determine the text and link based on the modelType
  const getModelTextAndLink = () => {
    switch (modelType) {
      case 'chat':
        return { text: 'chat model', link: '/docs/integrations/chat/' };
      case 'embeddings':
        return { text: 'embeddings model', link: '/docs/integrations/text_embedding/' };
      case 'vectorstore':
        return { text: 'vector store', link: '/docs/integrations/vectorstores/' };
      default:
        return { text: 'chat model', link: '/docs/integrations/chat/' };
    }
  };

  const { text, link } = getModelTextAndLink();

  return (
    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem', gap: '0.75rem' }}>
      <span style={{ 
        fontSize: '1rem',
        fontWeight: '500',
      }}>
        Select <a href={link}>{text}</a>:
      </span>
      <div className={`dropdown ${isOpen ? 'dropdown--show' : ''}`}>
        <button 
          className="button button--secondary" 
          onClick={() => setIsOpen(!isOpen)}
          style={{ 
            backgroundColor: 'var(--ifm-background-color)',
            border: '1px solid var(--ifm-color-emphasis-300)',
            fontWeight: 'normal',
            fontSize: '1rem',
            padding: '0.5rem 1rem',
            color: 'var(--ifm-font-color-base)',
          }}
        >
          {selectedOption.label}
          <span style={{ 
            marginLeft: '0.4rem',
            fontSize: '0.875rem'
          }}>▾</span>
        </button>
        <div className="dropdown__menu" style={{
          maxHeight: '210px',
          overflowY: 'auto',
          overflowX: 'hidden',
          marginBottom: 0,
        }}>
          {options.map((option) => (
            <li key={option.value}>
              <a 
                className={`dropdown__link ${option.value === selectedOption.value ? 'dropdown__link--active' : ''}`}
                href="#" 
                onClick={(e) => {
                  e.preventDefault();
                  onSelect(option.value);
                  setIsOpen(false);
                }}
              >
                {option.label}
              </a>
            </li>
          ))}
        </div>
      </div>
    </div>
  );
};


/**
 * @typedef {Object} ChatModelTabsProps - Component props.
 * @property {string} [openaiParams] - Parameters for OpenAI chat model. Defaults to `model="gpt-3.5-turbo-0125"`
 * @property {string} [anthropicParams] - Parameters for Anthropic chat model. Defaults to `model="claude-3-sonnet-20240229"`
 * @property {string} [cohereParams] - Parameters for Cohere chat model. Defaults to `model="command-r-plus"`
 * @property {string} [fireworksParams] - Parameters for Fireworks chat model. Defaults to `model="accounts/fireworks/models/mixtral-8x7b-instruct"`
 * @property {string} [groqParams] - Parameters for Groq chat model. Defaults to `model="llama3-8b-8192"`
 * @property {string} [mistralParams] - Parameters for Mistral chat model. Defaults to `model="mistral-large-latest"`
 * @property {string} [googleParams] - Parameters for Google chat model. Defaults to `model="gemini-pro"`
 * @property {string} [togetherParams] - Parameters for Together chat model. Defaults to `model="mistralai/Mixtral-8x7B-Instruct-v0.1"`
 * @property {string} [nvidiaParams] - Parameters for Nvidia NIM model. Defaults to `model="meta/llama3-70b-instruct"`
  * @property {string} [databricksParams] - Parameters for Databricks model. Defaults to `endpoint="databricks-meta-llama-3-1-70b-instruct"`
 * @property {string} [awsBedrockParams] - Parameters for AWS Bedrock chat model.
 * @property {boolean} [hideOpenai] - Whether or not to hide OpenAI chat model.
 * @property {boolean} [hideAnthropic] - Whether or not to hide Anthropic chat model.
 * @property {boolean} [hideCohere] - Whether or not to hide Cohere chat model.
 * @property {boolean} [hideFireworks] - Whether or not to hide Fireworks chat model.
 * @property {boolean} [hideGroq] - Whether or not to hide Groq chat model.
 * @property {boolean} [hideMistral] - Whether or not to hide Mistral chat model.
 * @property {boolean} [hideGoogle] - Whether or not to hide Google VertexAI chat model.
 * @property {boolean} [hideTogether] - Whether or not to hide Together chat model.
 * @property {boolean} [hideAzure] - Whether or not to hide Microsoft Azure OpenAI chat model.
 * @property {boolean} [hideNvidia] - Whether or not to hide NVIDIA NIM model.
 * @property {boolean} [hideAWS] - Whether or not to hide AWS models.
 * @property {boolean} [hideDatabricks] - Whether or not to hide Databricks models.
 * @property {string} [customVarName] - Custom variable name for the model. Defaults to `model`.
 */

/**
 * @param {ChatModelTabsProps} props - Component props.
 */
export default function ChatModelTabs(props) {
  const [selectedModel, setSelectedModel] = useState("OpenAI");
  const {
    openaiParams,
    anthropicParams,
    cohereParams,
    fireworksParams,
    groqParams,
    mistralParams,
    googleParams,
    togetherParams,
    azureParams,
    nvidiaParams,
    awsBedrockParams,
    databricksParams,
    hideOpenai,
    hideAnthropic,
    hideCohere,
    hideFireworks,
    hideGroq,
    hideMistral,
    hideGoogle,
    hideTogether,
    hideAzure,
    hideNvidia,
    hideAWS,
    hideDatabricks,
    customVarName,
  } = props;

  const openAIParamsOrDefault = openaiParams ?? `model="gpt-4o-mini"`;
  const anthropicParamsOrDefault =
    anthropicParams ?? `model="claude-3-5-sonnet-20240620"`;
  const cohereParamsOrDefault = cohereParams ?? `model="command-r-plus"`;
  const fireworksParamsOrDefault =
    fireworksParams ??
    `model="accounts/fireworks/models/llama-v3p1-70b-instruct"`;
  const groqParamsOrDefault = groqParams ?? `model="llama3-8b-8192"`;
  const mistralParamsOrDefault =
    mistralParams ?? `model="mistral-large-latest"`;
  const googleParamsOrDefault = googleParams ?? `model="gemini-1.5-flash"`;
  const togetherParamsOrDefault =
    togetherParams ??
    `\n    base_url="https://api.together.xyz/v1",\n    api_key=os.environ["TOGETHER_API_KEY"],\n    model="mistralai/Mixtral-8x7B-Instruct-v0.1",\n`;
  const azureParamsOrDefault =
    azureParams ??
    `\n    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],\n    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],\n    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],\n`;
  const nvidiaParamsOrDefault = nvidiaParams ?? `model="meta/llama3-70b-instruct"`
  const awsBedrockParamsOrDefault = awsBedrockParams ?? `model="anthropic.claude-3-5-sonnet-20240620-v1:0",\n    beta_use_converse_api=True`;
  const databricksParamsOrDefault = databricksParams ?? `endpoint="databricks-meta-llama-3-1-70b-instruct"`

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
      value: "Azure",
      label: "Azure",
      text: `from langchain_openai import AzureChatOpenAI\n\n${llmVarName} = AzureChatOpenAI(${azureParamsOrDefault})`,
      apiKeyName: "AZURE_OPENAI_API_KEY",
      packageName: "langchain-openai",
      default: false,
      shouldHide: hideAzure,
    },
    {
      value: "Google",
      label: "Google",
      text: `from langchain_google_vertexai import ChatVertexAI\n\n${llmVarName} = ChatVertexAI(${googleParamsOrDefault})`,
      apiKeyText: "# Ensure your VertexAI credentials are configured",
      packageName: "langchain-google-vertexai",
      default: false,
      shouldHide: hideGoogle,
    },
    {
      value: "AWS",
      label: "AWS",
      text: `from langchain_aws import ChatBedrock\n\n${llmVarName} = ChatBedrock(${awsBedrockParamsOrDefault})`,
      apiKeyText: "# Ensure your AWS credentials are configured",
      packageName: "langchain-aws",
      default: false,
      shouldHide: hideAWS,
    },
    {
      value: "Cohere",
      label: "Cohere",
      text: `from langchain_cohere import ChatCohere\n\n${llmVarName} = ChatCohere(${cohereParamsOrDefault})`,
      apiKeyName: "COHERE_API_KEY",
      packageName: "langchain-cohere",
      default: false,
      shouldHide: hideCohere,
    },
    {
      value: "NVIDIA",
      label: "NVIDIA",
      text: `from langchain_nvidia_ai_endpoints import ChatNVIDIA\n\n${llmVarName} = ChatNVIDIA(${nvidiaParamsOrDefault})`,
      apiKeyName: "NVIDIA_API_KEY",
      packageName: "langchain-nvidia-ai-endpoints",
      default: false,
      shouldHide: hideNvidia,
    },
    {
      value: "FireworksAI",
      label: "Fireworks AI",
      text: `from langchain_fireworks import ChatFireworks\n\n${llmVarName} = ChatFireworks(${fireworksParamsOrDefault})`,
      apiKeyName: "FIREWORKS_API_KEY",
      packageName: "langchain-fireworks",
      default: false,
      shouldHide: hideFireworks,
    },
    {
      value: "Groq",
      label: "Groq",
      text: `from langchain_groq import ChatGroq\n\n${llmVarName} = ChatGroq(${groqParamsOrDefault})`,
      apiKeyName: "GROQ_API_KEY",
      packageName: "langchain-groq",
      default: false,
      shouldHide: hideGroq,
    },
    {
      value: "MistralAI",
      label: "Mistral AI",
      text: `from langchain_mistralai import ChatMistralAI\n\n${llmVarName} = ChatMistralAI(${mistralParamsOrDefault})`,
      apiKeyName: "MISTRAL_API_KEY",
      packageName: "langchain-mistralai",
      default: false,
      shouldHide: hideMistral,
    },
    {
      value: "TogetherAI",
      label: "Together AI",
      text: `from langchain_openai import ChatOpenAI\n\n${llmVarName} = ChatOpenAI(${togetherParamsOrDefault})`,
      apiKeyName: "TOGETHER_API_KEY",
      packageName: "langchain-openai",
      default: false,
      shouldHide: hideTogether,
    },
    {
      value: "Databricks",
      label: "Databricks",
      text: `from databricks_langchain import ChatDatabricks\n\nos.environ["DATABRICKS_HOST"] = "https://example.staging.cloud.databricks.com/serving-endpoints"\n\n${llmVarName} = ChatDatabricks(${databricksParamsOrDefault})`,
      apiKeyName: "DATABRICKS_TOKEN",
      packageName: "databricks-langchain",
      default: false,
      shouldHide: hideDatabricks,
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
      modelType="chat"
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
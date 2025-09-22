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
 * @property {Object} [overrideParams] - An object for overriding the default parameters for each chat model, e.g. `{ openai: { model: "gpt-4o-mini" } }`
 * @property {string} [customVarName] - Custom variable name for the model. Defaults to `model`.
 */

/**
 * @param {ChatModelTabsProps} props - Component props.
 */
export default function ChatModelTabs(props) {
  const [selectedModel, setSelectedModel] = useState("google_genai");
  const {
    overrideParams,
    customVarName,
  } = props;

  const llmVarName = customVarName ?? "model";

  const tabItems = [
    {
      value: "openai",
      label: "OpenAI",
      model: "gpt-4o-mini",
      apiKeyName: "OPENAI_API_KEY",
      packageName: "langchain[openai]",
    },
    {
      value: "anthropic",
      label: "Anthropic",
      model: "claude-3-7-sonnet-20250219",
      comment: "# Note: Model versions may become outdated. Check https://docs.anthropic.com/en/docs/about-claude/models/overview for latest versions",
      apiKeyName: "ANTHROPIC_API_KEY",
      packageName: "langchain[anthropic]",
    },
    {
      value: "azure",
      label: "Azure",
      text: `from langchain_openai import AzureChatOpenAI

${llmVarName} = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)`,
      apiKeyName: "AZURE_OPENAI_API_KEY",
      packageName: "langchain[openai]",
    },
    {
      value: "google_genai",
      label: "Google Gemini",
      model: "gemini-2.5-flash",
      apiKeyName: "GOOGLE_API_KEY",
      packageName: "langchain[google-genai]",
    },
    {
      value: "google_vertexai",
      label: "Google Vertex",
      model: "gemini-2.5-flash",
      apiKeyText: "# Ensure your VertexAI credentials are configured",
      packageName: "langchain[google-vertexai]",
    },
    {
      value: "bedrock_converse",
      label: "AWS",
      model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
      apiKeyText: "# Ensure your AWS credentials are configured",
      packageName: "langchain[aws]",
    },
    {
      value: "groq",
      label: "Groq",
      model: "llama3-8b-8192",
      apiKeyName: "GROQ_API_KEY",
      packageName: "langchain[groq]",
    },
    {
      value: "cohere",
      label: "Cohere",
      model: "command-r-plus",
      apiKeyName: "COHERE_API_KEY",
      packageName: "langchain[cohere]",
    },
    {
      value: "nvidia",
      label: "NVIDIA",
      model: "meta/llama3-70b-instruct",
      apiKeyName: "NVIDIA_API_KEY",
      packageName: "langchain-nvidia-ai-endpoints",
    },
    {
      value: "fireworks",
      label: "Fireworks AI",
      model: "accounts/fireworks/models/llama-v3p1-70b-instruct",
      apiKeyName: "FIREWORKS_API_KEY",
      packageName: "langchain[fireworks]",
    },
    {
      value: "mistralai",
      label: "Mistral AI",
      model: "mistral-large-latest",
      apiKeyName: "MISTRAL_API_KEY",
      packageName: "langchain[mistralai]",
    },
    {
      value: "together",
      label: "Together AI",
      model: "mistralai/Mixtral-8x7B-Instruct-v0.1",
      apiKeyName: "TOGETHER_API_KEY",
      packageName: "langchain[together]",
    },
    {
      value: "ibm",
      label: "IBM watsonx",
      text: `from langchain_ibm import ChatWatsonx

${llmVarName} = ChatWatsonx(
    model_id="ibm/granite-34b-code-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="<WATSONX PROJECT_ID>"
)`,
      apiKeyName: "WATSONX_APIKEY",
      packageName: "langchain-ibm",
    },
    {
      value: "databricks",
      label: "Databricks",
      text: `from databricks_langchain import ChatDatabricks\n\nos.environ["DATABRICKS_HOST"] = "https://example.staging.cloud.databricks.com/serving-endpoints"\n\n${llmVarName} = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")`,
      apiKeyName: "DATABRICKS_TOKEN",
      packageName: "databricks-langchain",
    },
    {
      value: "xai",
      label: "xAI",
      model: "grok-2",
      apiKeyName: "XAI_API_KEY",
      packageName: "langchain-xai",
    },
    {
      value: "perplexity",
      label: "Perplexity",
      model: "llama-3.1-sonar-small-128k-online",
      apiKeyName: "PPLX_API_KEY",
      packageName: "langchain-perplexity",
    },
    {
      value: "deepseek",
      label: "DeepSeek",
      model: "deepseek-chat",
      apiKeyName: "DEEPSEEK_API_KEY",
      packageName: "langchain-deepseek",
    },
    {
      value: "chatocigenai",
      label: "ChatOCIGenAI",
      model: "cohere.command-r-plus-08-2024",
      apiKeyName: "OCI_API_KEY",
      packageName: "langchain-oci",
    }
  ].map((item) => ({
    ...item,
    ...overrideParams?.[item.value],
  }));

  const modelOptions = tabItems
    .map((item) => ({
      value: item.value,
      label: item.label,
    }));

  const selectedTabItem = tabItems.find(
    (option) => option.value === selectedModel
  );

  let apiKeyText = "";
  if (selectedTabItem.apiKeyName) {
    apiKeyText = `import getpass
import os

if not os.environ.get("${selectedTabItem.apiKeyName}"):
  os.environ["${selectedTabItem.apiKeyName}"] = getpass.getpass("Enter API key for ${selectedTabItem.label}: ")`;
  } else if (selectedTabItem.apiKeyText) {
    apiKeyText = selectedTabItem.apiKeyText;
  }

  const initModelText = selectedTabItem?.text || `from langchain.chat_models import init_chat_model

${llmVarName} = init_chat_model("${selectedTabItem.model}", model_provider="${selectedTabItem.value}"${selectedTabItem?.kwargs ? `, ${selectedTabItem.kwargs}` : ""})`;

  // Add comment if available
  const commentText = selectedTabItem?.comment ? selectedTabItem.comment + "\n\n" : "";

  return (
    <div>
      <CustomDropdown
        selectedOption={selectedTabItem}
        options={modelOptions}
        onSelect={setSelectedModel}
        modelType="chat"
      />

      <CodeBlock language="bash">
        {`pip install -qU "${selectedTabItem.packageName}"`}
      </CodeBlock>
      <CodeBlock language="python">
        {apiKeyText ? apiKeyText + "\n\n" + commentText + initModelText : commentText + initModelText}
      </CodeBlock>
    </div>
  );
}

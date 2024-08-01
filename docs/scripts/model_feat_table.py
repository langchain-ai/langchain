import sys
from pathlib import Path

from langchain_community import llms
from langchain_core.language_models.llms import LLM, BaseLLM

CHAT_MODEL_IGNORE = ("FakeListChatModel", "HumanInputChatModel")

CHAT_MODEL_FEAT_TABLE = {
    "ChatAnthropic": {
        "tool_calling": True,
        "multimodal": True,
        "package": "langchain-anthropic",
        "link": "/docs/integrations/chat/anthropic/",
    },
    "ChatMistralAI": {
        "tool_calling": True,
        "json_model": True,
        "package": "langchain-mistralai",
        "link": "/docs/integrations/chat/mistralai/",
    },
    "ChatFireworks": {
        "tool_calling": True,
        "json_mode": True,
        "package": "langchain-fireworks",
        "link": "/docs/integrations/chat/fireworks/",
    },
    "AzureChatOpenAI": {
        "tool_calling": True,
        "json_mode": True,
        "multimodal": True,
        "package": "langchain-openai",
        "link": "/docs/integrations/chat/azure_chat_openai/",
    },
    "ChatOpenAI": {
        "tool_calling": True,
        "json_mode": True,
        "multimodal": True,
        "package": "langchain-openai",
        "link": "/docs/integrations/chat/openai/",
    },
    "ChatTogether": {
        "tool_calling": True,
        "json_mode": True,
        "package": "langchain-together",
        "link": "/docs/integrations/chat/together/",
    },
    "ChatVertexAI": {
        "tool_calling": True,
        "multimodal": True,
        "package": "langchain-google-vertexai",
        "link": "/docs/integrations/chat/google_vertex_ai_palm/",
    },
    "ChatGoogleGenerativeAI": {
        "tool_calling": True,
        "multimodal": True,
        "package": "langchain-google-genai",
        "link": "/docs/integrations/chat/google_generative_ai/",
    },
    "ChatGroq": {
        "tool_calling": True,
        "json_mode": True,
        "package": "langchain-groq",
        "link": "/docs/integrations/chat/groq/",
    },
    "ChatCohere": {
        "tool_calling": True,
        "package": "langchain-cohere",
        "link": "/docs/integrations/chat/cohere/",
    },
    "ChatBedrock": {
        "tool_calling": True,
        "package": "langchain-aws",
        "link": "/docs/integrations/chat/bedrock/",
    },
    "ChatHuggingFace": {
        "tool_calling": True,
        "local": True,
        "package": "langchain-huggingface",
        "link": "/docs/integrations/chat/huggingface/",
    },
    "ChatNVIDIA": {
        "tool_calling": True,
        "json_mode": False,
        "local": True,
        "multimodal": False,
        "package": "langchain-nvidia-ai-endpoints",
        "link": "/docs/integrations/chat/nvidia_ai_endpoints/",
    },
    "ChatOllama": {
        "tool_calling": True,
        "local": True,
        "json_mode": True,
        "package": "langchain-ollama",
        "link": "/docs/integrations/chat/ollama/",
    },
    "vLLM Chat (via ChatOpenAI)": {
        "local": True,
        "package": "langchain-openai",
        "link": "/docs/integrations/chat/vllm/",
    },
    "ChatLlamaCpp": {
        "tool_calling": True,
        "local": True,
        "package": "langchain-community",
        "link": "/docs/integrations/chat/llamacpp",
    },
    "ChatAI21": {
        "tool_calling": True,
        "package": "langchain-ai21",
        "link": "/docs/integrations/chat/ai21",
    },
    "ChatWatsonx": {
        "tool_calling": True,
        "package": "langchain-ibm",
        "link": "/docs/integrations/chat/ibm_watsonx",
    },
    "ChatUpstage": {
        "tool_calling": True,
        "package": "langchain-upstage",
        "link": "/docs/integrations/chat/upstage",
    },
}

for feats in CHAT_MODEL_FEAT_TABLE.values():
    feats["structured_output"] = feats.get("tool_calling", False)

CHAT_MODEL_TEMPLATE = """\
---
sidebar_position: 0
sidebar_class_name: hidden
keywords: [compatibility]
custom_edit_url:
hide_table_of_contents: true
---

# Chat models

:::info

If you'd like to write your own chat model, see [this how-to](/docs/how_to/custom_chat_model/).
If you'd like to contribute an integration, see [Contributing integrations](/docs/contributing/integrations/).

:::

## Advanced features

The following table shows all the chat model classes that support one or more advanced features.

:::info
While all these LangChain classes support the indicated advanced feature, you may have
to open the provider-specific documentation to learn which hosted models or backends support
the feature.
:::

{table}

"""


def get_chat_model_table() -> str:
    """Get the table of chat models."""
    header = [
        "model",
        "tool_calling",
        "structured_output",
        "json_mode",
        "local",
        "multimodal",
        "package",
    ]
    title = [
        "Model",
        "[Tool calling](/docs/how_to/tool_calling)",
        "[Structured output](/docs/how_to/structured_output/)",
        "JSON mode",
        "Local",
        "[Multimodal](/docs/how_to/multimodal_inputs/)",
        "Package",
    ]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for llm, feats in sorted(CHAT_MODEL_FEAT_TABLE.items()):
        # Fields are in the order of the header
        row = [
            f"[{llm}]({feats['link']})",
        ]
        for h in header[1:]:
            value = feats.get(h)
            if h == "package":
                value = value or "langchain-community"
                name = value[len("langchain-") :]
                link = f"https://api.python.langchain.com/en/latest/{name}_api_reference.html"
                value = f"[{value}]({link})"
                row.append(value)
            else:
                if value == "partial":
                    row.append("🟡")
                elif value is True:
                    row.append("✅")
                else:
                    row.append("❌")
        rows.append(row)
    return "\n".join(["|".join(row) for row in rows])


if __name__ == "__main__":
    output_dir = Path(sys.argv[1])
    output_integrations_dir = output_dir / "integrations"
    output_integrations_dir_chat = output_integrations_dir / "chat"
    output_integrations_dir_chat.mkdir(parents=True, exist_ok=True)

    chat_model_page = CHAT_MODEL_TEMPLATE.format(table=get_chat_model_table())
    with open(output_integrations_dir / "chat" / "index.mdx", "w") as f:
        f.write(chat_model_page)

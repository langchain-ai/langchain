import sys
from pathlib import Path

from langchain_community import llms
from langchain_core.language_models.llms import LLM, BaseLLM

LLM_IGNORE = ("FakeListLLM", "OpenAIChat", "PromptLayerOpenAIChat")
LLM_FEAT_TABLE_CORRECTION = {
    "TextGen": {"_astream": False, "_agenerate": False},
    "Ollama": {
        "_stream": False,
    },
    "PromptLayerOpenAI": {"batch_generate": False, "batch_agenerate": False},
}
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


LLM_TEMPLATE = """\
---
sidebar_position: 1
sidebar_class_name: hidden
keywords: [compatibility]
custom_edit_url:
---

# LLMs

## Features (natively supported)
All LLMs implement the Runnable interface, which comes with default implementations of all methods, ie. `ainvoke`, `batch`, `abatch`, `stream`, `astream`. This gives all LLMs basic support for async, streaming and batch, which by default is implemented as below:
- *Async* support defaults to calling the respective sync method in asyncio's default thread pool executor. This lets other async functions in your application make progress while the LLM is being executed, by moving this call to a background thread.
- *Streaming* support defaults to returning an `Iterator` (or `AsyncIterator` in the case of async streaming) of a single value, the final result returned by the underlying LLM provider. This obviously doesn't give you token-by-token streaming, which requires native support from the LLM provider, but ensures your code that expects an iterator of tokens can work for any of our LLM integrations.
- *Batch* support defaults to calling the underlying LLM in parallel for each input by making use of a thread pool executor (in the sync batch case) or `asyncio.gather` (in the async batch case). The concurrency can be controlled with the `max_concurrency` key in `RunnableConfig`.

Each LLM integration can optionally provide native implementations for async, streaming or batch, which, for providers that support it, can be more efficient. The table shows, for each integration, which features have been implemented with native support.

{table}

"""

CHAT_MODEL_TEMPLATE = """\
---
sidebar_position: 0
sidebar_class_name: hidden
keywords: [compatibility]
custom_edit_url:
hide_table_of_contents: true
---

# Chat models

## Advanced features

The following table shows all the chat model classes that support one or more advanced features.

:::info
While all these LangChain classes support the indicated advanced feature, you may have
to open the provider-specific documentation to learn which hosted models or backends support
the feature.
:::

{table}

"""


def get_llm_table():
    llm_feat_table = {}
    for cm in llms.__all__:
        llm_feat_table[cm] = {}
        cls = getattr(llms, cm)
        if issubclass(cls, LLM):
            for feat in ("_stream", "_astream", ("_acall", "_agenerate")):
                if isinstance(feat, tuple):
                    feat, name = feat
                else:
                    feat, name = feat, feat
                llm_feat_table[cm][name] = getattr(cls, feat) != getattr(LLM, feat)
        else:
            for feat in [
                "_stream",
                "_astream",
                ("_generate", "batch_generate"),
                "_agenerate",
                ("_agenerate", "batch_agenerate"),
            ]:
                if isinstance(feat, tuple):
                    feat, name = feat
                else:
                    feat, name = feat, feat
                llm_feat_table[cm][name] = getattr(cls, feat) != getattr(BaseLLM, feat)
    final_feats = {
        k: v
        for k, v in {**llm_feat_table, **LLM_FEAT_TABLE_CORRECTION}.items()
        if k not in LLM_IGNORE
    }

    header = [
        "model",
        "_agenerate",
        "_stream",
        "_astream",
        "batch_generate",
        "batch_agenerate",
    ]
    title = [
        "Model",
        "Invoke",
        "Async invoke",
        "Stream",
        "Async stream",
        "Batch",
        "Async batch",
    ]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for llm, feats in sorted(final_feats.items()):
        rows += [[llm, "✅"] + ["✅" if feats.get(h) else "❌" for h in header[1:]]]
    return "\n".join(["|".join(row) for row in rows])


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
                row.append(value or "langchain-community")
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
    output_integrations_dir_llms = output_integrations_dir / "llms"
    output_integrations_dir_chat = output_integrations_dir / "chat"
    output_integrations_dir_llms.mkdir(parents=True, exist_ok=True)
    output_integrations_dir_chat.mkdir(parents=True, exist_ok=True)

    llm_page = LLM_TEMPLATE.format(table=get_llm_table())

    with open(output_integrations_dir / "llms" / "index.mdx", "w") as f:
        f.write(llm_page)
    chat_model_page = CHAT_MODEL_TEMPLATE.format(table=get_chat_model_table())
    with open(output_integrations_dir / "chat" / "index.mdx", "w") as f:
        f.write(chat_model_page)

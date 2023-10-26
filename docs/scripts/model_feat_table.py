import os
from pathlib import Path

from langchain import chat_models, llms
from langchain.chat_models.base import BaseChatModel, SimpleChatModel
from langchain.llms.base import BaseLLM, LLM

INTEGRATIONS_DIR = (
    Path(os.path.abspath(__file__)).parents[1]
    / "docs"
    / "integrations"
)
LLM_IGNORE = ("FakeListLLM", "OpenAIChat", "PromptLayerOpenAIChat")
LLM_FEAT_TABLE_CORRECTION = {
    "TextGen": {"_astream": False, "_agenerate": False},
    "Ollama": {
        "_stream": False,
    },
    "PromptLayerOpenAI": {"batch_generate": False, "batch_agenerate": False},
}
CHAT_MODEL_IGNORE = ("FakeListChatModel", "HumanInputChatModel")
CHAT_MODEL_FEAT_TABLE_CORRECTION = {
    "ChatMLflowAIGateway": {"_agenerate": False},
    "PromptLayerChatOpenAI": {"_stream": False, "_astream": False},
    "ChatKonko": {"_astream": False, "_agenerate": False},
}

LLM_TEMPLATE = """\
---
sidebar_position: 0
sidebar_class_name: hidden
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
sidebar_position: 1
sidebar_class_name: hidden
---

# Chat models

## Features (natively supported)
All ChatModels implement the Runnable interface, which comes with default implementations of all methods, ie. `ainvoke`, `batch`, `abatch`, `stream`, `astream`. This gives all ChatModels basic support for async, streaming and batch, which by default is implemented as below:
- *Async* support defaults to calling the respective sync method in asyncio's default thread pool executor. This lets other async functions in your application make progress while the ChatModel is being executed, by moving this call to a background thread.
- *Streaming* support defaults to returning an `Iterator` (or `AsyncIterator` in the case of async streaming) of a single value, the final result returned by the underlying ChatModel provider. This obviously doesn't give you token-by-token streaming, which requires native support from the ChatModel provider, but ensures your code that expects an iterator of tokens can work for any of our ChatModel integrations.
- *Batch* support defaults to calling the underlying ChatModel in parallel for each input by making use of a thread pool executor (in the sync batch case) or `asyncio.gather` (in the async batch case). The concurrency can be controlled with the `max_concurrency` key in `RunnableConfig`.

Each ChatModel integration can optionally provide native implementations to truly enable async or streaming.
The table shows, for each integration, which features have been implemented with native support.

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


def get_chat_model_table():
    feat_table = {}
    for cm in chat_models.__all__:
        feat_table[cm] = {}
        cls = getattr(chat_models, cm)
        if issubclass(cls, SimpleChatModel):
            comparison_cls = SimpleChatModel
        else:
            comparison_cls = BaseChatModel
        for feat in ("_stream", "_astream", "_agenerate"):
            feat_table[cm][feat] = getattr(cls, feat) != getattr(comparison_cls, feat)
    final_feats = {
        k: v
        for k, v in {**feat_table, **CHAT_MODEL_FEAT_TABLE_CORRECTION}.items()
        if k not in CHAT_MODEL_IGNORE
    }
    header = ["model", "_agenerate", "_stream", "_astream"]
    title = ["Model", "Invoke", "Async invoke", "Stream", "Async stream"]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for llm, feats in sorted(final_feats.items()):
        rows += [[llm, "✅"] + ["✅" if feats.get(h) else "❌" for h in header[1:]]]
    return "\n".join(["|".join(row) for row in rows])


if __name__ == "__main__":
    llm_page = LLM_TEMPLATE.format(table=get_llm_table())
    with open(INTEGRATIONS_DIR / "llms" / "index.mdx", "w") as f:
        f.write(llm_page)
    chat_model_page = CHAT_MODEL_TEMPLATE.format(table=get_chat_model_table())
    with open(INTEGRATIONS_DIR / "chat" / "index.mdx", "w") as f:
        f.write(chat_model_page)

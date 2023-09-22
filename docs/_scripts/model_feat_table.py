import os
from pathlib import Path

from langchain import chat_models, llms
from langchain.chat_models.base import BaseChatModel, SimpleChatModel
from langchain.llms.base import BaseLLM, LLM

INTEGRATIONS_DIR = (
    Path(os.path.abspath(__file__)).parents[1] / "extras" / "integrations"
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

import DocCardList from "@theme/DocCardList";

## Features (natively supported)
All `LLM`s implement the LCEL `Runnable` interface, meaning they all expose functioning `invoke`, `ainvoke`, `batch`, `abatch`, `stream`, and `astream` methods.
*That is, they all have functioning sync, async, streaming, and batch generation methods.*

This table highlights specifically those integrations that **natively support** batching, streaming, and asynchronous generation (meaning these features are built into the 3rd-party integration).

{table}

<DocCardList />
"""

CHAT_MODEL_TEMPLATE = """\
---
sidebar_position: 1
sidebar_class_name: hidden
---

# Chat models

import DocCardList from "@theme/DocCardList";

## Features (natively supported)
All `ChatModel`s implement the LCEL `Runnable` interface, meaning they all expose functioning `invoke`, `ainvoke`, `stream`, and `astream` (and `batch`, `abatch`) methods.
*That is, they all have functioning sync, async and streaming generation methods.*

This table highlights specifically those integrations that **natively support** streaming and asynchronous generation (meaning these features are built into the 3rd-party integration).

{table}

<DocCardList />
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
    title = ["Model", "Generate", "Async generate", "Stream", "Async stream", "Batch", "Async batch"]
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
    title = ["Model", "Generate", "Async generate", "Stream", "Async stream"]
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

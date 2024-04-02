from typing import Any


def __getattr__(name) -> Any:
    if name == "ChatMistralAI":
        from langchain_mistralai.chat_models import ChatMistralAI

        return ChatMistralAI
    elif name == "MistralAIEmbeddings":
        from langchain_mistralai.embeddings import MistralAIEmbeddings

        return MistralAIEmbeddings
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["ChatMistralAI", "MistralAIEmbeddings"]

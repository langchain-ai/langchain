from langchain_community.embeddings.self_hosted_hugging_face import (
    DEFAULT_EMBED_INSTRUCTION,
    DEFAULT_INSTRUCT_MODEL,
    DEFAULT_MODEL_NAME,
    DEFAULT_QUERY_INSTRUCTION,
    SelfHostedHuggingFaceEmbeddings,
    SelfHostedHuggingFaceInstructEmbeddings,
    _embed_documents,
    load_embedding_model,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_INSTRUCT_MODEL",
    "DEFAULT_EMBED_INSTRUCTION",
    "DEFAULT_QUERY_INSTRUCTION",
    "_embed_documents",
    "load_embedding_model",
    "SelfHostedHuggingFaceEmbeddings",
    "SelfHostedHuggingFaceInstructEmbeddings",
]

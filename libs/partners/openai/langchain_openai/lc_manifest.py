from typing import Literal

# we want to register integration classes, as well as provide a hook

manifest: dict[
    str,
    dict[
        Literal[
            "chat_model",
            "retriever",
            "embeddings",
            "vectorstore",
            "tool",
            "toolkit",
            "document_loader",
        ],
        list[str],
    ],
] = {
    "openai": {
        "chat_model": [
            "langchain_openai.chat_models.base.ChatOpenAI",
        ],
        "embeddings": [
            "langchain_openai.embeddings.base.OpenAIEmbeddings",
        ],
    },
    "azure": {
        "chat_model": [
            "langchain_openai.chat_models.azure.AzureChatOpenAI",
        ],
        "embeddings": [
            "langchain_openai.embeddings.azure.AzureOpenAIEmbeddings",
        ],
    },
}

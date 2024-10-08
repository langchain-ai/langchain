from enum import Enum


class ComponentKind(str, Enum):
    """Type of component.

    - Type literals use the component folder names in the docs/integrations/.
      If the component folder name is missed, the component `ingex` page is not created.
    - Only components with the interface (base) class are defined as the Component.
    - Components with the small number of classes are not defined as the Component.
    - Agent and Chain are marked as components even they are part of the
      `langchain` package only.
    """

    AGENT = "agents"
    CACHE = "caches"
    CALLBACK = "callbacks"
    CHAIN = "chains"
    CHAT_LOADER = "chat_loaders"
    CHAT_MESSAGE_HISTORY = "chat_message_history"
    CHAT_MODEL = "chat"
    DOCUMENT_LOADER = "document_loaders"
    DOCUMENT_TRANSFORMER = "document_transformers"
    EMBEDDING = "text_embedding"
    EXAMPLE_SELECTOR = "example_selectors"
    LLM = "llms"
    MEMORY = "memory"
    OUTPUT_PARSER = "output_parsers"
    RETRIEVER = "retrievers"
    TOOL = "tools"
    TOOLKIT = "toolkits"
    VECTORSTORE = "vectorstores"

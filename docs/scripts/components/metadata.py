from pydantic.v1 import BaseModel

from docs.scripts.components.kind import ComponentKind


class ComponentMetadata(BaseModel):
    """Metadata for a component."""

    kind: ComponentKind
    """Type of the component."""
    base_class: str
    """The base class of the component. Only classes that inherit from this class are defined as Component."""


component_metadatas = [
    ComponentMetadata(
        kind=ComponentKind.AGENT, base_class="langchain.agents.agent.Agent"
    ),
    ComponentMetadata(
        kind=ComponentKind.CACHE, base_class="langchain_core.caches.BaseCache"
    ),
    ComponentMetadata(
        kind=ComponentKind.CALLBACK,
        base_class="langchain_core.callbacks.BaseCallbackHandler",
    ),
    ComponentMetadata(
        kind=ComponentKind.CHAIN, base_class="langchain.chains.base.Chain"
    ),
    ComponentMetadata(
        kind=ComponentKind.CHAT_LOADER,
        base_class="langchain_core.chat_loaders.BaseChatLoader",
    ),
    # ComponentMetadata(
    # TODO: Now BaseChatMessageHistory is messing up with BaseMemory.
    #     kind=ComponentKind.CHAT_MESSAGE_HISTORY,
    #     base_class="langchain_core.chat_history.BaseChatMessageHistory",
    # ),
    ComponentMetadata(
        kind=ComponentKind.CHAT_MODEL,
        base_class="langchain_core.language_models.chat_models.BaseChatModel",
    ),
    ComponentMetadata(
        kind=ComponentKind.DOCUMENT_LOADER,
        base_class="langchain_core.document_loaders.base.BaseLoader",
    ),
    ComponentMetadata(
        kind=ComponentKind.DOCUMENT_TRANSFORMER,
        base_class="langchain_core.documents.BaseDocumentTransformer",
    ),
    ComponentMetadata(
        kind=ComponentKind.EMBEDDING, base_class="langchain_core.embeddings.Embeddings"
    ),
    ComponentMetadata(
        kind=ComponentKind.EXAMPLE_SELECTOR,
        base_class="langchain_core.example_selectors.BaseExampleSelector",
    ),
    ComponentMetadata(
        kind=ComponentKind.LLM, base_class="langchain_core.language_models.BaseLLM"
    ),  # TODO or LLM class?
    ComponentMetadata(
        kind=ComponentKind.MEMORY,
        # TODO: Now BaseChatMessageHistory is messing up with BaseMemory.
        # base_class="langchain_core.memory.BaseMemory",
        base_class="langchain_core.chat_history.BaseChatMessageHistory",
    ),
    # ComponentMetadata(
    # TODO: Now output_parser is not presented as a component in doc but only in the code.
    #     kind=ComponentKind.OUTPUT_PARSER,
    #     base_class="langchain_core.output_parsers.base.BaseOutputParser",
    # ),
    ComponentMetadata(
        kind=ComponentKind.RETRIEVER,
        base_class="langchain_core.retrievers.BaseRetriever",
    ),
    ComponentMetadata(
        kind=ComponentKind.TOOL, base_class="langchain_core.tools.BaseTool"
    ),
    ComponentMetadata(
        kind=ComponentKind.TOOLKIT, base_class="langchain_core.tools.BaseToolkit"
    ),
    ComponentMetadata(
        kind=ComponentKind.VECTORSTORE,
        base_class="langchain_core.vectorstores.VectorStore",
    ),
]

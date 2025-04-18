"""Tools for interacting with Vectara."""

import json
import os
from typing import Dict, List, Optional, Type, Union

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, Field

from langchain_community.tools.vectorstore.tool import BaseVectorStoreTool

try:
    from langchain_vectara import (
        CorpusConfig,
        GenerationConfig,
        SearchConfig,
        Vectara,
        VectaraQueryConfig,
    )
except ImportError:
    raise ImportError(
        "The langchain-vectara package is not installed. "
        "Please install it with `pip install langchain-vectara`"
    )


class VectaraIngestInput(BaseModel):
    """Input for the Vectara ingest tool."""

    documents: List[str] = Field(description="List of texts to ingest into Vectara")
    metadatas: Optional[List[Dict]] = Field(
        default=None, description="Optional metadata for each document"
    )
    ids: Optional[List[str]] = Field(
        default=None, description="Optional list of IDs associated with each document"
    )
    corpus_key: Optional[str] = Field(
        default=None, description="Corpus key where documents will be ingested"
    )
    doc_metadata: Optional[Dict] = Field(
        default=None, description="Optional metadata at the document level"
    )
    doc_type: Optional[str] = Field(
        default=None,
        description="Optional document type ('core' or 'structured'). Defaults to 'structured'",
    )


class VectaraSearch(BaseVectorStoreTool, BaseTool):
    """Tool for searching the Vectara platform.

    Example:
        .. code-block:: python

            from langchain_community.tools import VectaraSearch
            from langchain_vectara import Vectara  # Import from langchain-vectara
            from langchain_vectara import VectaraQueryConfig, SearchConfig, CorpusConfig

            # Initialize the Vectara vectorstore
            vectara = Vectara(
                vectara_api_key="your-api-key"
            )

            # Create the tool
            tool = VectaraSearch(
                name="vectara_search",
                description="Search for information in the Vectara corpus",
                vectorstore=vectara,
                corpus_key="your-corpus-id"
            )

            # Create a VectaraQueryConfig
            corpus_config = CorpusConfig(
                corpus_key="your-corpus-id",
                metadata_filter="doc.type = 'article'",
                lexical_interpolation=0.2
            )

            search_config = SearchConfig(
                corpora=[corpus_config],
                limit=10
            )

            query_config = VectaraQueryConfig(
                search=search_config,
                generation=None
            )

            # Use the tool with the config
            results = tool.run({
                "query": "What is RAG?",
                "config": query_config
            })
    """

    name: str = "vectara_search"
    description: str = (
        "Search for information in your Vectara corpus using semantic search. "
        "This tool understands the meaning of your query beyond simple keyword matching. "
        "Useful for retrieving specific information from your documents based on meaning and context. "
    )

    # Default corpus_key if not provided in the config
    corpus_key: Optional[str] = None

    @staticmethod
    def get_description(name: str, description: str) -> str:
        """Get the description for the tool.

        Args:
            name: The name of the Vectara corpus or knowledge base
            description: Additional description of the knowledge base

        Returns:
            A formatted description for the tool
        """
        template: str = (
            "Useful for when you need to answer questions about {name} using semantic search. "
            "This tool understands the meaning and context of your query, not just keywords. "
            "Whenever you need information about {description} you should use this. "
        )
        return template.format(name=name, description=description)

    def _run(
        self,
        query: str,
        config: Optional[VectaraQueryConfig] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the Vectara search."""
        try:
            if not isinstance(self.vectorstore, Vectara):
                return (
                    "Error: The vectorstore must be a Vectara instance from the "
                    "langchain-vectara package. Please install it with "
                    "`pip install langchain-vectara`"
                )

            if not config and not self.corpus_key:
                return (
                    "Error: A corpus_key is required for search. "
                    "You can provide it either directly to the tool or in the config object."
                )

            if not config:
                search_config = SearchConfig()

                if self.corpus_key:
                    corpus_config = CorpusConfig(corpus_key=self.corpus_key)
                    search_config.corpora = [corpus_config]

                config = VectaraQueryConfig(search=search_config)

            results = self.vectorstore.similarity_search_with_score(
                query, config=config
            )

            if not results:
                return "No results found"

            # Directly serialize structured results with scores
            return json.dumps(
                [
                    {
                        "index": i,
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "Vectara"),
                        "metadata": doc.metadata,
                        "score": score,
                    }
                    for i, (doc, score) in enumerate(results)
                ],
                indent=2,
            )
        except Exception as e:
            return f"Error searching Vectara: {str(e)}"


class VectaraGeneration(BaseVectorStoreTool, BaseTool):
    """Tool for generating summaries from the Vectara platform.

    Example:
        .. code-block:: python

            from langchain_community.tools import VectaraGeneration
            from langchain_vectara import Vectara  # Import from langchain-vectara
            from langchain_vectara import VectaraQueryConfig, SearchConfig, CorpusConfig, GenerationConfig

            # Initialize the Vectara vectorstore
            vectara = Vectara(
                vectara_api_key="your-api-key"
            )

            # Create the tool
            tool = VectaraGeneration(
                name="vectara_generation",
                description="Generate summaries from your Vectara corpus",
                vectorstore=vectara,
                corpus_key="your-corpus-id"  # Optional, can be provided in config
            )

            # Create a VectaraQueryConfig with search and generation settings
            corpus_config = CorpusConfig(
                corpus_key="your-corpus-id",
                metadata_filter="doc.type = 'article'",
                lexical_interpolation=0.2
            )

            search_config = SearchConfig(
                corpora=[corpus_config],
                limit=10
            )

            generation_config = GenerationConfig(
                max_used_search_results=10,
                response_language="eng",
                generation_preset_name="vectara-summary-ext-24-05-med-omni"
            )

            query_config = VectaraQueryConfig(
                search=search_config,
                generation=generation_config
            )

            # Use the tool with the config
            results = tool.run({
                "query": "What is RAG?",
                "config": query_config
            })
    """

    name: str = "vectara_generation"
    description: str = (
        "Generate AI responses from your Vectara corpus using semantic search. "
        "This tool understands the meaning of your query and generates a concise summary from the most relevant results. "
    )

    # Default corpus_key if not provided in the config
    corpus_key: Optional[str] = None

    @staticmethod
    def get_description(name: str, description: str) -> str:
        """Get the description for the tool.

        Args:
            name: The name of the Vectara corpus or knowledge base
            description: Additional description of the knowledge base

        Returns:
            A formatted description for the tool
        """
        template: str = (
            "Useful for when you need AI-generated answers about {name}. "
            "This tool understands the meaning of your query and generates a concise response from relevant documents. "
            "Whenever you need a comprehensive overview about {description} you should use this. "
        )
        return template.format(name=name, description=description)

    def _run(
        self,
        query: str,
        config: Optional[VectaraQueryConfig] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the Vectara generation."""
        try:
            if not isinstance(self.vectorstore, Vectara):
                return (
                    "Error: The vectorstore must be a Vectara instance from the "
                    "langchain-vectara package. Please install it with "
                    "`pip install langchain-vectara`"
                )

            if not config and not self.corpus_key:
                return (
                    "Error: A corpus_key is required for generation. "
                    "You can provide it either directly to the tool or in the config object."
                )

            if not config:
                search_config = SearchConfig()

                if self.corpus_key:
                    corpus_config = CorpusConfig(corpus_key=self.corpus_key)
                    search_config.corpora = [corpus_config]

                generation_config = GenerationConfig(
                    max_used_search_results=7,
                    response_language="eng",
                    generation_preset_name="vectara-summary-ext-24-05-med-omni",
                    enable_factual_consistency_score=True,
                )

                config = VectaraQueryConfig(
                    search=search_config, generation=generation_config
                )

            rag = self.vectorstore.as_rag(config)
            result = rag.invoke(query)

            if not result:
                return "No results found"

            return json.dumps(
                {
                    "summary": result.get("answer"),
                    "factual_consistency_score": result.get("fcs", "N/A"),
                },
                indent=2,
            )

        except Exception as e:
            return f"Error generating response from Vectara: {str(e)}"


class VectaraIngest(BaseVectorStoreTool, BaseTool):
    """Tool for ingesting documents into the Vectara platform.

    Example:
        .. code-block:: python

            from langchain_community.tools import VectaraIngest
            from langchain_vectara import Vectara  # Import from langchain-vectara

            # Initialize the Vectara vectorstore
            vectara = Vectara(
                vectara_api_key="your-api-key"
            )

            # Create the tool
            tool = VectaraIngest(
                name="vectara_ingest",
                description="Ingest documents into the Vectara corpus",
                vectorstore=vectara,
                corpus_key="your-corpus-id"  # Required for ingestion
            )

            # Use the tool with additional parameters
            result = tool.run({
                "documents": ["Document 1", "Document 2"],
                "metadatas": [{"source": "file1"}, {"source": "file2"}],
                "ids": ["doc1", "doc2"],
                "doc_metadata": {"batch": "batch1"},
                "doc_type": "structured"
            })
    """

    name: str = "vectara_ingest"
    description: str = (
        "Ingest documents into your Vectara corpus for semantic search. "
        "Useful for adding new information to your knowledge base that can be queried using natural language. "
        "Documents will be processed to understand their meaning and context. "
        "Input should be a list of texts to ingest."
    )
    args_schema: Type[BaseModel] = VectaraIngestInput

    # Required corpus_key for ingestion
    corpus_key: str = Field(
        ...,  # This makes it required
        description="Corpus key where documents will be ingested",
    )

    @staticmethod
    def get_description(name: str, description: str) -> str:
        """Get the description for the tool.

        Args:
            name: The name of the Vectara corpus or knowledge base
            description: Additional description of the knowledge base

        Returns:
            A formatted description for the tool
        """
        template: str = (
            "Useful for when you need to add new information to {name} for semantic search. "
            "Documents added will be processed for meaning and context to enable natural language querying. "
            "Whenever you need to ingest new documents about {description} you should use this. "
            "Input should be a list of text documents to ingest."
        )
        return template.format(name=name, description=description)

    def _run(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        corpus_key: Optional[str] = None,
        doc_metadata: Optional[Dict] = None,
        doc_type: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the Vectara ingest.

        Args:
            documents: List of texts to ingest into Vectara
            metadatas: Optional metadata for each document
            ids: Optional list of IDs for each document
            corpus_key: Optional corpus key override
            doc_metadata: Optional metadata at the document level
            doc_type: Optional document type ('core' or 'structured')
            run_manager: Optional callback manager

        Returns:
            String describing the ingestion result
        """
        try:
            if not isinstance(self.vectorstore, Vectara):
                return (
                    "Error: The vectorstore must be a Vectara instance from the "
                    "langchain-vectara package. Please install it with "
                    "`pip install langchain-vectara`"
                )

            active_corpus_key = corpus_key or self.corpus_key

            if not active_corpus_key:
                return "Error: corpus_key is required for ingestion"

            kwargs = {"corpus_key": active_corpus_key}

            if doc_metadata is not None:
                kwargs["doc_metadata"] = doc_metadata

            if doc_type is not None:
                kwargs["doc_type"] = doc_type

            doc_ids = self.vectorstore.add_texts(
                texts=documents, metadatas=metadatas, ids=ids, **kwargs
            )

            return f"Successfully ingested {len(doc_ids)} documents into Vectara corpus {active_corpus_key} with IDs: {', '.join(doc_ids)}"
        except Exception as e:
            return f"Error ingesting documents to Vectara: {str(e)}"

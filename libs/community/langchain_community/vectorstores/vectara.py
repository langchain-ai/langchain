from __future__ import annotations

import json
import logging
import os
from hashlib import md5
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import requests
from langchain_core.callbacks.manager import (
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    """
    Configuration for generating citations in summary output.

    Attributes:
        style: The citation style to be used. Valid options include:
            - "numeric": Formats citations as simple numerals (e.g. [1], [2], ...).
            - "none": Omits citations from the output.
            - "html": Formats citations as HTML links using the specified URL and text
              patterns, e.g. `<a href="url_pattern">text_pattern</a>`.
            - "markdown": Formats citations as Markdown links using the specified URL
              and text patterns, e.g. `[text_pattern](url_pattern)`.
        url_pattern: (Optional) The URL pattern to be used when the citation style is
            "html" or "markdown". This pattern can incorporate metadata placeholders
            (e.g. "https://my.doc/foo/{doc.id}/{part.id}").
        text_pattern: (Optional) The text pattern to be used when the citation style is
            "html" or "markdown". This pattern sets the display text (or link text) in
            the final citation. If not provided, it defaults to using the result index
            (e.g. [N](<url_pattern>)). Metadata placeholders (e.g. "{doc.title}") can be
             used to insert document attributes.
    """

    style: Optional[Union[Literal["none", "numeric", "html", "markdown"], Any]] = Field(
        default=None
    )
    url_pattern: Optional[str] = Field(default=None)
    text_pattern: Optional[str] = Field(default=None)


class GenerationConfig(BaseModel):
    """Configuration for summary generation.

    Attributes:
        max_used_search_results: The maximum number of search results to be available to
            the prompt.
        response_language: requested language for the summary
            generation_preset_name: name of the prompt to use for summarization (see
            https://docs.vectara.com/docs/learn/grounded-generation/select-a-summarizer)
        enable_factual_consistency_score: Score based on the HHEM that indicates the
        factual accuracy of the summary
    """

    max_used_search_results: int = 7
    response_language: str = "eng"
    generation_preset_name: str = "vectara-summary-ext-24-05-med-omni"
    enable_factual_consistency_score: bool = True
    citations: Optional[Citation] = None
    model_parameters: Optional[Dict[str, Any]] = Field(default=None)


class ContextConfig(BaseModel):
    """
    Configuration on the presentation of each document part in the result set.

    Attributes:
        characters_before: Number of characters shown before the matching part,
            ignored if `sentences_before` is set.
        characters_after: Number of characters shown after the matching part,
            ignored if `sentences_after` is set.
        sentences_before: Number of sentences shown before the matching part to
            provide context.
        sentences_after: Number of sentences shown after the matching part to
            provide context.
        start_tag: Tag wrapping the document part at the start, useful for UI
            highlighting.
        end_tag: Tag wrapping the document part at the end, useful for UI highlighting.
    """

    characters_before: Optional[int] = None
    characters_after: Optional[int] = None
    sentences_before: Optional[int] = 2
    sentences_after: Optional[int] = 2
    start_tag: Optional[str] = None
    end_tag: Optional[str] = None


class CorpusConfig(BaseModel):
    """
    Configuration for a corpus used in search and retrieval.

    Attributes:
        corpus_key: Unique identifier for the corpus.
        custom_dimensions: User-defined numerical dimensions that influence scoring
            (https://docs.vectara.com/docs/learn/semantic-search/add-custom-dimensions).
        metadata_filter: SQL-like filter string to restrict search based on metadata
            attributes.
        lexical_interpolation: Weighting between lexical (keyword) and embedding search,
            from 0 (embedding only) to 1 (lexical only).
    """

    corpus_key: Optional[str] = None
    custom_dimensions: Optional[dict[str, float]] = None
    metadata_filter: Optional[str] = None
    lexical_interpolation: Optional[float] = 0.0


class CustomerSpecificReranker(BaseModel):
    """
    A reranker specific to the customer, used to refine search results.

    Attributes:
        type: Identifier for this reranker type.
        reranker_id: Unique ID of the reranker (deprecated, use `reranker_name` instead)
        reranker_name: Name of the reranker, excluding MMR rerankers.
        limit: Maximum number of results returned after reranking.
        cutoff: Minimum score threshold for results after reranking.
    """

    type: Literal["customer_reranker"] = "customer_reranker"
    reranker_id: Optional[str] = None
    reranker_name: Optional[str] = None
    limit: Optional[int] = None
    cutoff: Optional[float] = None


class UserFunctionReranker(BaseModel):
    """
    A reranker that applies a user-defined function to reorder search results.

    Attributes:
        type: Identifier for this reranker type.
        user_function: The user-defined function used for reranking.
        limit: Maximum number of results returned after reranking.
        cutoff: Minimum score threshold for results after reranking.
    """

    type: Literal["userfn"] = "userfn"
    user_function: Optional[str] = None
    limit: Optional[int] = None
    cutoff: Optional[float] = None


class MmrReranker(BaseModel):
    """
    A reranker that applies Maximal Marginal Relevance (MMR) to diversify search results

    Attributes:
        type: Identifier for this reranker type.
        diversity_bias: Controls result diversity, higher values increase diversity.
        limit: Maximum number of results returned after reranking.
        cutoff: Minimum score threshold for results after reranking.
    """

    type: Literal["mmr"] = "mmr"
    diversity_bias: Optional[float] = 0.2
    limit: Optional[int] = None
    cutoff: Optional[float] = None


class NoneReranker(BaseModel):
    """
    A reranker that applies no additional reranking to search results.

    Attributes:
        type: Identifier for this reranker type.
        limit : Maximum number of results returned after reranking.
    """

    type: Literal["none"] = "none"
    limit: Optional[int] = None


class ChainReranker(BaseModel):
    """
    A reranker that applies multiple rerankers sequentially.

    Attributes:
        type: Identifier for this reranker type.
        rerankers: List of rerankers applied in sequence.
    """

    type: Literal["chain"] = "chain"
    rerankers: List[
        Union[CustomerSpecificReranker, UserFunctionReranker, MmrReranker, NoneReranker]
    ] = Field()


class SearchConfig(BaseModel):
    """
    Configuration for searching one or more corpora.

    Attributes:
        corpora: List of corpora to search within.
        offset: Number of results to skip, useful for pagination.
        limit: Maximum number of search results to return.
        context_configuration: Context settings for search results.
        reranker: Reranker to refine search results.
    """

    corpora: List[CorpusConfig] = Field(default_factory=list)
    offset: Optional[int] = None
    limit: Optional[int] = None
    context_configuration: Optional[ContextConfig] = Field(
        default_factory=ContextConfig
    )
    reranker: Optional[
        Union[
            CustomerSpecificReranker,
            UserFunctionReranker,
            MmrReranker,
            NoneReranker,
            ChainReranker,
        ]
    ] = None


class Cell(BaseModel):
    """
    Represents a cell in a table.

    Attributes:
        text_value: A text value.
        int_value: A signed 64-bit integer value.
        float_value: A floating-point value with double precision.
        bool_value: A boolean value.
        colspan: Number of columns the cell spans (default: 1).
        rowspan: Number of rows the cell spans (default: 1).
    """

    text_value: Optional[str] = None
    int_value: Optional[int] = None
    float_value: Optional[float] = None
    bool_value: Optional[bool] = None
    colspan: Optional[int] = None
    rowspan: Optional[int] = None


class Data(BaseModel):
    """
    The data of a table.

    Attributes:
        headers: The headers of the table.
        rows: The rows in the data.
    """

    headers: Optional[List[List["Cell"]]] = None
    rows: Optional[List[List["Cell"]]] = None


class Table(BaseModel):
    """
    A table in a document.

    Attributes:
        id (Optional[str]): The unique ID of the table within the document.
        title (Optional[str]): The title of the table.
        data (Optional[Data]): The data of the table.
        description (Optional[str]): The description of the table.
    """

    id: Optional[str] = None
    title: Optional[str] = None
    data: Optional["Data"] = None
    description: Optional[str] = None


class CoreDocumentPart(BaseModel):
    """
    A part of a document that maps directly to a search result, typically a sentence.

    Attributes:
        text: The text content of the document part.
        metadata: Metadata associated with the document part.
        table_id: ID of the table this document part belongs to.
        context: Additional context for the document part.
        custom_dimensions: Custom dimensions for search relevance.
    """

    text: str
    metadata: Optional[Dict[str, Any]] = None
    table_id: Optional[str] = None
    context: Optional[str] = None
    custom_dimensions: Optional[dict[str, float]] = None


class CoreDocument(BaseModel):
    """
    Represents a document in Vectara's core document model.

    Attributes:
        id: Unique identifier for the document within the corpus.
        type (Literal["core"]): Specifies the document type as "core".
        metadata: Document-level metadata that can be used in query-time filtering.
        tables: List of tables contained within the document.
        document_parts: Individual parts of the document, typically sentences, that are
        indexed separately.
    """

    id: str
    type: Literal["core"] = "core"
    metadata: Optional[Dict[str, Any]] = None
    tables: Optional[List["Table"]] = None
    document_parts: List["CoreDocumentPart"] = Field(default_factory=list)


class StructuredDocumentSection(BaseModel):
    """
    A logical section within a structured document.

    Attributes:
        text: The text content of the section.
        id: The section ID, automatically converted to a metadata field.
        title: The section title.
        metadata: Section-level metadata for filtering.
        tables: Tables contained within the section.
        sections: Nested subsections within this section.
    """

    text: str
    id: Optional[int] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tables: Optional[List["Table"]] = None
    sections: Optional[List["StructuredDocumentSection"]] = None


class StructuredDocument(BaseModel):
    """
    A document with layout features.

    Attributes:
        id (str): Unique document ID within the corpus.
        type (Literal["structured"]): Document type, always "structured".
        title (Optional[str]): The title of the document.
        description (Optional[str]): The description of the document.
        metadata (Optional[Dict[str, Any]]): Arbitrary JSON metadata for filtering and
            categorization.
        custom_dimensions (Optional[CustomDimensions]): User-defined dimensions for
            ranking.
        sections (List[StructuredDocumentSection]): Subsections of the document.
        chunking_strategy (Optional[ComponentsSchemasMaxCharsChunkingStrategy]):
            Strategy for chunking document content.
    """

    id: str
    sections: List["StructuredDocumentSection"] = Field()
    type: Literal["structured"] = "structured"
    title: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    custom_dimensions: Optional[dict[str, float]] = None
    chunking_strategy: Optional["ChunkingStrategy"] = None


class ChunkingStrategy(BaseModel):
    """
    Defines a chunking strategy that limits the number of characters per chunk.

     Attributes:
        type (Literal["max_chars_chunking_strategy"]): Identifies the chunking strategy
            type.
        max_chars_per_chunk (int): Maximum number of characters allowed per chunk.
    """

    max_chars_per_chunk: int = Field()
    type: Literal["max_chars_chunking_strategy"] = "max_chars_chunking_strategy"


class TableExtractionConfig(BaseModel):
    """
    Configuration for extracting tables from a document.
    """

    extract_tables: bool = Field()


class File(BaseModel):
    file_path: str
    metadata: Optional[Dict[str, Any]] = None
    chunking_strategy: Optional[ChunkingStrategy] = None
    table_extraction_config: Optional[TableExtractionConfig] = None
    filename: Optional[str] = None


class VectaraQueryConfig(BaseModel):
    """
    Configuration for a Vectara query.

    Attributes:
        search (SearchConfig): Configuration for search parameters.
        generation (GenerationConfig): Configuration for generating summaries.
        stream_response (bool): Whether to stream responses in real-time.
        save_history (bool): Whether to save query history. Defaults to False.
        chat (bool): If True, the last document is treated as the answer with
            `chat_conv_id` in metadata.
        chat_conv_id (Optional[str]): Conversation ID for chat mode.
    """

    search: SearchConfig = Field(default_factory=SearchConfig)
    generation: Union[GenerationConfig, None] = Field(default_factory=GenerationConfig)
    stream_response: bool = False
    save_history: bool = False
    chat: bool = Field(default=False)
    chat_conv_id: Optional[str] = None


class Vectara(VectorStore):
    """`Vectara API` vector store.

     See (https://vectara.com).

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import Vectara

            vectorstore = Vectara(
                vectara_api_key=vectara_api_key
            )
    """

    def __init__(
        self,
        vectara_api_key: Optional[str] = None,
        vectara_api_timeout: int = 120,
        source: str = "langchain",
        vectara_base_url: str = "https://api.vectara.io",
        vectara_verify_ssl: bool = True,
    ):
        """Initialize with Vectara API."""
        self._vectara_api_key = vectara_api_key or os.environ.get("VECTARA_API_KEY")
        self._base_url = vectara_base_url.rstrip("/")
        self._verify_ssl = vectara_verify_ssl
        if self._vectara_api_key is None:
            logger.warning("Can't find Vectara API key in environment.")
            raise ValueError("unable to find Vectara API key.")

        self._source = source

        self._session = requests.Session()  # to reuse connections
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        self._session.mount("http://", adapter)
        self.vectara_api_timeout = vectara_api_timeout

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return None

    def _get_post_headers(self) -> dict[str, str | None]:
        """Returns headers that should be attached to each post request."""
        return {
            "x-api-key": self._vectara_api_key,
            "Content-Type": "application/json",
            "X-Source": self._source,
        }

    def _get_error_message(self, response: dict) -> str:
        # Extract error messages
        field_errors = response.get("field_errors", {})
        field_error_msg = (
            "; ".join([f"{key}: {value}" for key, value in field_errors.items()])
            if field_errors
            else ""
        )

        messages = response.get("messages", [])
        message_error_msg = "; ".join(messages) if messages else ""

        # Combine error messages
        error_message = (
            "; ".join(filter(None, [field_error_msg, message_error_msg]))
            or "Unknown error"
        )
        return error_message

    def _generate_doc_id(self, text: str) -> str:
        """
        Generate a unique document ID by hashing the given text with MD5.

        Args:
            text (str): The text to be hashed.

        Returns:
            str: The hexadecimal digest of the MD5 hash.
        """
        return md5(text.encode("utf-8")).hexdigest()

    def _delete_doc(self, doc_id: str, corpus_key: str) -> bool:
        """
        Deletes a document from the specified corpus.

        Args:
            doc_id (str): The ID of the document to delete.
            corpus_key (str): The key of the corpus from which the document should be
                deleted.

        Returns:
            bool: True if the document was successfully deleted, False otherwise.
        """

        if not doc_id:
            raise ValueError("Document ID cannot be empty.")
        if not corpus_key:
            raise ValueError("Corpus key cannot be empty.")

        response = self._session.delete(
            f"{self._base_url}/v2/corpora/{corpus_key}/documents/{doc_id}",
            verify=True,
            headers=self._get_post_headers(),
            timeout=self.vectara_api_timeout,
        )
        if response.status_code != 204:
            logger.error(
                f"Delete request failed for doc_id = {doc_id} with status code "
                f"{response.status_code}, reason {response.json()}"
            )
            return False
        return True

    def _index_doc(
        self, doc: Union["CoreDocument", "StructuredDocument"], corpus_key: str
    ) -> str:
        """
        Indexes a CoreDocument or StructuredDocument into the specified corpus.

        Args:
            doc (Union[CoreDocument, StructuredDocument]): The document to be indexed.
            corpus_key (str): The key of the corpus where the document should be stored.

        Returns:
            str: "SUCCEEDED" if the document is successfully indexed, otherwise an
                error message.
        """

        if not corpus_key:
            raise ValueError("Corpus key cannot be empty.")

        doc_payload = json.dumps(doc.model_dump(mode="json", exclude_none=True))
        response = self._session.post(
            headers=self._get_post_headers(),
            url=f"{self._base_url}/v2/corpora/{corpus_key}/documents",
            data=doc_payload,
            timeout=self.vectara_api_timeout,
            verify=self._verify_ssl,
        )

        result = response.json()
        status_code = response.status_code

        if status_code == 201:
            return "SUCCEEDED"

        error_message = self._get_error_message(result)

        if status_code == 409 or status_code == 412:
            return "ALREADY_EXISTS"

        return error_message

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        Delete document using document ids.

        Args:
            ids (Optional[List[str]]): List of document IDs to delete.
            **kwargs: Additional arguments:
                - corpus_key (str): Required. The key of the corpus where documents will
                    be stored.


        Returns:
            Optional[bool]: True if deletion is successful, False otherwise. None if not
                implemented.
        """
        corpus_key = kwargs.get("corpus_key")
        if not corpus_key:
            raise ValueError("Corpus key cannot be empty.")
        if ids:
            success = [self._delete_doc(doc_id, corpus_key) for doc_id in ids]
            return all(success)
        else:
            return True

    def add_files(
        self,
        files_list: List[File],
        corpus_key: str,
        **kwargs: Any,
    ) -> List[str]:
        """
        Upload files such as PDFs and Word Documents for automatic text extraction and
            metadata parsing.

        Args:
            files_list (List[File]): A list of File objects containing file paths and
                optional metadata.
            corpus_key (str): The key of the corpus where the document should be stored.

        Returns:
            List[str]: A list of document IDs for successfully indexed files.
        """
        doc_ids = []
        for file_obj in files_list:
            file_path = file_obj.file_path
            if not os.path.exists(file_path):
                logger.error(f"File {file_path} does not exist, skipping")
                continue

            if file_obj.filename is None:
                file_obj.filename = file_path.split("/")[-1]

            files: Dict[str, tuple | str] = {
                "file": (file_obj.filename, open(file_path, "rb"))
            }

            if file_obj.metadata:
                file_obj.metadata["framework"] = self._source
                files["metadata"] = (
                    None,
                    json.dumps(file_obj.metadata),
                    "application/json",
                )

            if file_obj.chunking_strategy:
                files["chunking_strategy"] = (
                    None,
                    json.dumps(file_obj.chunking_strategy.model_dump()),
                    "application/json",
                )
            if file_obj.table_extraction_config:
                files["table_extraction_config"] = (
                    None,
                    json.dumps(file_obj.table_extraction_config.model_dump()),
                    "application/json",
                )
            if file_obj.filename:
                files["filename"] = file_obj.filename

            headers = self._get_post_headers()
            headers.pop("Content-Type", None)
            response = self._session.post(
                f"{self._base_url}/v2/corpora/{corpus_key}/upload_file",
                files=files,
                verify=self._verify_ssl,
                headers=headers,
                timeout=self.vectara_api_timeout,
            )
            result = response.json()
            if response.status_code == 201:
                doc_ids.append(result.get("id"))
            else:
                error_message = self._get_error_message(result)
                logger.error(
                    f"File upload failed ({response.status_code}), reason: "
                    f"{error_message}"
                )

        return doc_ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Each text is treated as a separate document.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with each text.
            ids: Optional list of IDs associated with each text.
            **kwargs: Additional arguments:
                - corpus_key (str): Required. The key of the corpus where documents will
                    be stored.
                - doc_metadata (dict): Optional. Metadata at the document level.
                - doc_type (Literal["core", "structured"]): Optional. The document type.
                    Defaults to "structured".

        Returns:
            List of document IDs for the documents that were successfully indexed.

        Raises:
            ValueError: If corpus_key is not provided in kwargs.
        """
        corpus_key: Optional[str] = kwargs.get("corpus_key")
        if not corpus_key:
            raise ValueError("Missing required parameter: 'corpus_key'.")

        doc_metadata: dict = kwargs.get("doc_metadata", {})
        doc_metadata["source"] = "langchain"

        doc_type: Literal["core", "structured"] = kwargs.get("doc_type", "structured")

        texts = list(texts)
        n_texts = len(texts)

        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != n_texts:
            if n_texts > len(metadatas):
                logger.warning(
                    f"Mismatch in metadatas and texts length. Expected {n_texts} "
                    f"but got {len(metadatas)}. Auto-generating metadata for the "
                    f"remaining documents."
                )
                metadatas += [{}] * (n_texts - len(metadatas))
            else:
                metadatas = metadatas[:n_texts]

        final_ids: List[str] = []
        if ids is None:
            # Generate auto IDs per text using MD5 of text and doc_metadata as a seed
            for text in texts:
                final_ids.append(self._generate_doc_id(text))
        else:
            ids = list(ids)
            if len(ids) < n_texts:
                logger.warning(
                    f"Length of ids ({len(ids)}) is less than number of texts "
                    f"({n_texts}). Auto-generating IDs for the remaining documents."
                )
                final_ids = ids[:]
                for text in texts[len(ids) :]:
                    final_ids.append(self._generate_doc_id(text))
            elif len(ids) > n_texts:
                # If more IDs than texts, we only use the first n_texts
                final_ids = ids[:n_texts]
            else:
                final_ids = ids

        results = []
        for i, text in enumerate(texts):
            doc_id = final_ids[i]

            doc: Union[CoreDocument, StructuredDocument]
            if doc_type == "core":
                doc = CoreDocument(
                    id=doc_id,
                    metadata=doc_metadata,
                    document_parts=[CoreDocumentPart(text=text, metadata=metadatas[i])],
                )
            elif doc_type == "structured":
                doc = StructuredDocument(
                    id=doc_id,
                    metadata=doc_metadata,
                    sections=[
                        StructuredDocumentSection(text=text, metadata=metadatas[i])
                    ],
                )
            else:
                raise ValueError("Invalid doc_type. Must be 'core' or 'structured'.")

            success_str = self._index_doc(doc, corpus_key=corpus_key)
            if success_str == "SUCCEEDED":
                results.append(doc_id)
            elif success_str == "ALREADY_EXISTS":
                # If already exists, try deleting then indexing again
                logger.error(
                    f"Unable to index document {doc_id}. Reason: {success_str}"
                )
                self._delete_doc(doc_id, corpus_key)
                success_str = self._index_doc(doc, corpus_key=corpus_key)
                if success_str == "SUCCEEDED":
                    results.append(doc_id)
                else:
                    logger.error(
                        f"Unable to index document {doc_id}. Reason: {success_str}"
                    )
            else:
                logger.error(
                    f"Unable to index document {doc_id}. Reason: {success_str}"
                )

        return results

    def _get_query_body(
        self,
        query: str,
        config: VectaraQueryConfig,
        **kwargs: Any,
    ) -> dict:
        """Build the body for the API

        Args:
            query: Text to look up documents similar to.
            config: VectaraQueryConfig object
        Returns:
            A dictionary with the body of the query
        """
        search_config = (
            config.search.model_dump(mode="json", exclude_none=True)
            if config.search
            else None
        )
        generation_config = (
            config.generation.model_dump(mode="json", exclude_none=True)
            if config.generation
            else None
        )
        body = {
            "query": query,
            "search": search_config,
            "generation": generation_config,
            "stream_response": config.stream_response,
            "save_history": config.save_history,
        }
        if config.chat:
            body["chat"] = {"store": True}

        return body

    def vectara_query(
        self,
        query: str,
        config: VectaraQueryConfig,
        **kwargs: Any,
    ) -> Union[List[Tuple[Document, float]]]:
        """Run a Vectara query

        Args:
            query: Text to look up documents similar to.
            config: VectaraQueryConfig object
        Returns:
             If streaming is enabled, an iterator that yields streamed chunks.
            Otherwise,
                A list of k Documents matching the given query
                If summary is enabled, last document is the summary text with
                'summary'=True
                if chat is set to Ture, last document is the answer with chat_convo_id
                in the metadata
        """

        body = self._get_query_body(query, config, **kwargs)
        if config.chat:
            if config.chat_conv_id:
                response = self._session.post(
                    headers=self._get_post_headers(),
                    url=f"{self._base_url}/v2/chats/{config.chat_conv_id}/turns",
                    data=json.dumps(body),
                    timeout=self.vectara_api_timeout,
                    verify=self._verify_ssl,
                )
            else:
                response = self._session.post(
                    headers=self._get_post_headers(),
                    url=f"{self._base_url}/v2/chats",
                    data=json.dumps(body),
                    timeout=self.vectara_api_timeout,
                    verify=self._verify_ssl,
                )
        else:
            response = self._session.post(
                headers=self._get_post_headers(),
                url=f"{self._base_url}/v2/query",
                data=json.dumps(body),
                timeout=self.vectara_api_timeout,
                verify=self._verify_ssl,
            )
        result = response.json()
        error_message = self._get_error_message(response.json())

        if response.status_code != 200:
            logger.error(
                f"Query failed (code {response.status_code}), reason {error_message}"
            )
            return []

        if "warnings" in result:
            logger.warning(f"Query warning(s) {', '.join(result['warnings'])}")

        search_results = result["search_results"]
        res = [
            (
                Document(
                    page_content=search_result["text"],
                    metadata=search_result.get("document_metadata", {}),
                ),
                search_result["score"],
            )
            for search_result in search_results
        ]

        if config.generation or config.chat:
            summary_text = (
                result.get("answer") if config.chat else result.get("summary")
            )

            if isinstance(summary_text, tuple) and len(summary_text) > 0:
                summary_text = summary_text[0]
            fcs = (result.get("factual_consistency_score"),)
            if config.chat:
                metadata = {"chat_convo_id": result["chat_id"], "fcs": fcs}
            else:
                metadata = {"summary": True, "fcs": fcs}
            res.append(
                (
                    Document(page_content=summary_text, metadata=metadata),
                    0.0,
                )
            )
        return res

    def similarity_search_with_score(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return Vectara documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            **kwargs (Any): Additional parameters including:
                - search (SearchConfig): Configuration for search parameters.
                - generation (GenerationConfig): Configuration for generating summaries.
                - stream_response (bool): Whether to stream responses in real-time.
                - save_history (bool): Whether to save query history.

        Returns:
            List of Documents most similar to the query and score for each.
        """
        config = VectaraQueryConfig(**kwargs)
        if kwargs.get("filter"):
            config.search.corpora[0].metadata_filter = kwargs.get("filter", "")

        docs = self.vectara_query(query, config)
        return docs

    def similarity_search(  # type: ignore[override]
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Document]:
        """Return Vectara documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            **kwargs (Any): Additional parameters including:
                - search (SearchConfig): Configuration for search parameters.
                - generation (GenerationConfig): Configuration for generating summaries.
                - stream_response (bool): Whether to stream responses in real-time.
                - save_history (bool): Whether to save query history.

        Returns:
            List of Documents most similar to the query
        """
        docs_and_scores = self.similarity_search_with_score(
            query,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(  # type: ignore[override]
        self,
        query: str,
        fetch_k: int = 50,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List["Document"]:
        """
        Return documents selected using maximal marginal relevance (MMR), optimizing
        for similarity to the query while ensuring diversity in the results.

        Args:
            query (str): Text to look up documents similar to.
            fetch_k (int): Number of documents to retrieve before applying MMR.
            lambda_mult (float): Diversity parameter (0 for max diversity, 1 for
                min diversity).
            **kwargs (Any): Additional VectaraQueryConfig parameters.

        Returns:
            List[Document]: Documents selected by maximal marginal relevance.
        """

        mmr_reranker = MmrReranker(diversity_bias=lambda_mult, limit=fetch_k)

        if "search" not in kwargs or not isinstance(kwargs["search"], SearchConfig):
            kwargs["search"] = SearchConfig(reranker=mmr_reranker)
        else:
            kwargs["search"].reranker = mmr_reranker

        return self.similarity_search(query, **kwargs)

    def get_document(self, doc_id: str, corpus_key: str) -> Optional[Document]:
        """
        Retrieve a single document by its ID from Vectara and merge the text parts.

        Args:
            doc_id: The ID of the document to retrieve.
            corpus_key (str): The key of the corpus to retrieve the document from.

        Returns:
            A Document object if retrieval is successful, or None if retrieval fails or
                the document type is unrecognized.
        """

        url = f"{self._base_url}/v2/corpora/{corpus_key}/documents/{doc_id}"
        response = self._session.get(
            url,
            headers=self._get_post_headers(),
            timeout=self.vectara_api_timeout,
            verify=self._verify_ssl,
        )

        if response.status_code == 200:
            doc_json = response.json()
            text_parts = []
            doc_metadata = doc_json.get("metadata", {}).copy()

            for part in doc_json.get("parts", []):
                text_parts.append(part.get("text", ""))
                part_metadata = part.get("metadata", {})
                for k, v in part_metadata.items():
                    doc_metadata[k] = v

            merged_text = "\n".join(text_parts).strip()
            return Document(page_content=merged_text, metadata=doc_metadata)
        else:
            error_message = self._get_error_message(response.json())
            logger.error(
                f"Failed to retrieve doc_id='{doc_id}',  "
                f"status=({response.status_code}), reason: {error_message}"
            )
            return None

    def get_by_ids(self, ids: Sequence[str], /, **kwargs: Any) -> List[Document]:
        """
        Retrieve documents from Vectara by their IDs and merge their content.

        Args:
            ids: A sequence of document IDs to retrieve.
            **kwargs: Additional parameters, including:
                - corpus_key (str): Required. The corpus where the documents reside.

        Returns:
            A list of `Document` objects whose `page_content` is the merged text
            from each document part. If any document fails or is not found, it is
            skipped in the final result.
        """
        corpus_key = kwargs.get("corpus_key")
        if not corpus_key:
            raise ValueError("Missing required parameter: 'corpus_key'.")

        results = []
        for doc_id in ids:
            doc_obj = self.get_document(doc_id, corpus_key)
            if doc_obj:
                results.append(doc_obj)
        return results

    @classmethod
    def from_texts(
        cls: Type["Vectara"],
        texts: List[str],
        embedding: Optional["Embeddings"] = None,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "Vectara":
        """
        Construct a Vectara wrapper from raw documents.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embedding (Optional[Embeddings]): Ignored, as Vectara generates its own
                embeddings.
            metadatas: Optional list of metadatas associated with each text.
            ids: Optional list of IDs associated with each text.
            **kwargs: Additional arguments:
                - vectara_api_key (str): API key to use for initializing Vectara
                - corpus_key (str): Required. The key of the corpus.
                - doc_metadata (dict): Optional. Metadata at the document level.
                - doc_type (Literal["core", "structured"]): Optional. The document type.

        Returns:
            List of document IDs for the documents that were successfully indexed.

        Raises:
            ValueError: If corpus_key is not provided in kwargs.

         Example:
            .. code-block:: python

                from langchain_community.vectorstores import Vectara
                vectara = Vectara.from_texts(
                    texts,
                    vectara_api_key=api_key,
                    doc_type="core",
                    corpus_key="your_corpus_key"
                )

        """
        corpus_key: Optional[str] = kwargs.pop("corpus_key", None)
        if not corpus_key:
            raise ValueError("Missing required parameter: 'corpus_key'.")

        doc_metadata: dict = kwargs.pop("doc_metadata", {})
        doc_type: Literal["core", "structured"] = kwargs.pop("doc_type", "structured")

        vectara = cls(**kwargs)

        vectara.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            doc_metadata=doc_metadata,
            doc_type=doc_type,
            corpus_key=corpus_key,
            **kwargs,
        )

        return vectara

    @classmethod
    def from_documents(
        cls: Type["Vectara"],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "Vectara":
        """
        Construct a Vectara wrapper from documents and embeddings.

        Args:
            documents: List of Document objects to add.
            embedding: Embedding function to use (ignored by Vectara).
            **kwargs: Additional keyword arguments. Expected keys include:
                - vectara_api_key (str): API key for initializing Vectara.
                - corpus_key (str): Required. The corpus key for indexing.
                - doc_metadata (dict): Optional document-level metadata.
                - doc_type (Literal["core", "structured"]): Optional.

        Returns:
            Vectara: An instance of Vectara with the documents indexed.
        """

        corpus_key: Optional[str] = kwargs.get("corpus_key")
        if not corpus_key:
            raise ValueError("Missing required parameter: 'corpus_key'.")

        return super().from_documents(documents, embedding, **kwargs)

    def as_rag(self, config: VectaraQueryConfig) -> VectaraRAG:
        """Return a Vectara RAG runnable."""
        return VectaraRAG(self, config)

    def as_chat(self, config: VectaraQueryConfig) -> VectaraRAG:
        """Return a Vectara RAG runnable for chat."""
        config.chat = True
        return VectaraRAG(self, config)

    def as_retriever(self, **kwargs: Any) -> VectaraRetriever:
        """return a retriever object."""
        return VectaraRetriever(
            vectorstore=self, config=kwargs.get("config", VectaraQueryConfig())
        )


class VectaraRetriever(VectorStoreRetriever):  # type: ignore[override]
    """Vectara Retriever class."""

    vectorstore: Vectara
    """VectorStore to use for retrieval."""

    config: VectaraQueryConfig
    """Configuration for this retriever."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        docs_and_scores = self.vectorstore.vectara_query(query, self.config, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        return self.vectorstore.add_documents(documents, **kwargs)


class VectaraRAG(Runnable):
    """Vectara RAG runnable.

    Parameters:
        vectara: Vectara object
        config: VectaraQueryConfig object
    """

    def __init__(self, vectara: Vectara, config: VectaraQueryConfig):
        self.vectara = vectara
        self.config = config or VectaraQueryConfig()

    def stream(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[dict]:
        """Get streaming output from Vectara RAG.

        Args:
            input: The input query
            config: RunnableConfig object
            kwargs: Any additional arguments

        Returns:
            The output dictionary with question, answer and context
        """
        body = self.vectara._get_query_body(
            input,
            self.config,
        )
        body["stream_response"] = True
        if self.config.chat:
            if self.config.chat_conv_id:
                response = self.vectara._session.post(
                    headers=self.vectara._get_post_headers(),
                    url=f"{self.vectara._base_url}/v2/chats/{self.config.chat_conv_id}"
                    f"/turns",
                    data=json.dumps(body),
                    timeout=self.vectara.vectara_api_timeout,
                    stream=True,
                    verify=self.vectara._verify_ssl,
                )
            else:
                response = self.vectara._session.post(
                    headers=self.vectara._get_post_headers(),
                    url=f"{self.vectara._base_url}/v2/chats",
                    data=json.dumps(body),
                    timeout=self.vectara.vectara_api_timeout,
                    stream=True,
                    verify=self.vectara._verify_ssl,
                )
        else:
            response = self.vectara._session.post(
                headers=self.vectara._get_post_headers(),
                url=f"{self.vectara._base_url}/v2/query",
                data=json.dumps(body),
                timeout=self.vectara.vectara_api_timeout,
                stream=True,
                verify=self.vectara._verify_ssl,
            )

        if response.status_code != 200:
            result = response.json()
            if response.status_code == 400:
                logger.error(
                    f"Query failed (code {response.status_code}), "
                    f"reason {result['field_errors']}"
                )
            else:
                logger.error(
                    f"Query failed (code {response.status_code}), "
                    f"reason {result['messages'][0]}"
                )

            return

        yield {"question": input}  # First chunk is the question

        for line in response.iter_lines():
            line = line.decode("utf-8")
            if line:
                key, value = line.split(":", 1)
                if key == "data":
                    line = json.loads(value)
                    if line["type"] == "generation_chunk":
                        yield {"answer": line["generation_chunk"]}

                    elif line["type"] == "factual_consistency_score":
                        yield {"fcs": line["factual_consistency_score"]}

                    elif line["type"] == "search_results":
                        documents = [
                            (
                                Document(
                                    page_content=search_result["text"],
                                    metadata=search_result["document_metadata"],
                                ),
                                search_result["score"],
                            )
                            for search_result in line["search_results"]
                        ]

                        yield {"context": documents}

                    elif line["type"] == "chat_info":
                        self.config.chat_conv_id = line["chat_id"]
                        yield {"chat_id": line["chat_id"]}
        return

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> dict:
        res = {"answer": ""}
        for chunk in self.stream(input):
            if "context" in chunk:
                res["context"] = chunk["context"]
            elif "question" in chunk:
                res["question"] = chunk["question"]
            elif "answer" in chunk:
                res["answer"] += chunk["answer"]
            elif "fcs" in chunk:
                res["fcs"] = chunk["fcs"]
            elif "chat_id" in chunk:
                res["chat_id"] = chunk["chat_id"]
            else:
                logger.error(f"Unknown chunk type: {chunk}")
        return res

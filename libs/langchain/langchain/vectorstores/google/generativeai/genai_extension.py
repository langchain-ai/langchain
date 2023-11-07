"""Temporary high-level library of the PaLM API.

The content of this file should eventually go into the Python package
google.generativeai.
"""

import datetime
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, MutableSequence, Optional

import google.ai.generativelanguage as genai
from google.api_core import client_options as client_options_lib
from google.api_core import gapic_v1
from google.protobuf import timestamp_pb2

import langchain

_logger = logging.getLogger(__name__)
_DEFAULT_API_ENDPOINT = "autopush-generativelanguage.sandbox.googleapis.com"
_USER_AGENT = f"langchain/{langchain.__version__}"
_default_page_size = 20
_default_text_service_model = "models/text-bison-001"
_MAX_REQUEST_PER_BATCH = 100
_name_regex = re.compile(r"^corpora/([^/]+?)(/documents/([^/]+?)(/chunks/([^/]+?))?)?$")


@dataclass
class EntityName:
    corpus_id: str
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.chunk_id is not None and self.document_id is None:
            raise ValueError(f"Chunk must have document ID but found {self}")

    @classmethod
    def from_str(cls, encoded: str) -> "EntityName":
        matched = _name_regex.match(encoded)
        if not matched:
            raise ValueError(f"Invalid entity name: {encoded}")

        return cls(
            corpus_id=matched.group(1),
            document_id=matched.group(3),
            chunk_id=matched.group(5),
        )

    def __repr__(self) -> str:
        name = f"corpora/{self.corpus_id}"
        if self.document_id is None:
            return name
        name += f"/documents/{self.document_id}"
        if self.chunk_id is None:
            return name
        name += f"/chunks/{self.chunk_id}"
        return name

    def __str__(self) -> str:
        return repr(self)

    def is_corpus(self) -> bool:
        return self.document_id is None

    def is_document(self) -> bool:
        return self.document_id is not None and self.chunk_id is None

    def is_chunk(self) -> bool:
        return self.chunk_id is not None


@dataclass
class Corpus:
    name: str
    display_name: Optional[str]
    create_time: Optional[timestamp_pb2.Timestamp]
    update_time: Optional[timestamp_pb2.Timestamp]

    @property
    def corpus_id(self) -> str:
        name = EntityName.from_str(self.name)
        return name.corpus_id

    @classmethod
    def from_corpus(cls, c: genai.Corpus) -> "Corpus":
        return cls(
            name=c.name,
            display_name=c.display_name,
            create_time=c.create_time,
            update_time=c.update_time,
        )


@dataclass
class Document:
    name: str
    display_name: Optional[str]
    create_time: Optional[timestamp_pb2.Timestamp]
    update_time: Optional[timestamp_pb2.Timestamp]
    custom_metadata: Optional[MutableSequence[genai.CustomMetadata]]

    @property
    def corpus_id(self) -> str:
        name = EntityName.from_str(self.name)
        return name.corpus_id

    @property
    def document_id(self) -> str:
        name = EntityName.from_str(self.name)
        assert isinstance(name.document_id, str)
        return name.document_id

    @classmethod
    def from_document(cls, d: genai.Document) -> "Document":
        return cls(
            name=d.name,
            display_name=d.display_name,
            create_time=d.create_time,
            update_time=d.update_time,
            custom_metadata=d.custom_metadata,
        )


@dataclass
class Config:
    """Global configuration for Google Generative AI API.

    Normally, the defaults should work fine. Change them only if you understand
    why.

    Attributes:
        api_endpoint: The Google Generative API endpoint address.
        user_agent: The user agent to use for logging.
        page_size: For paging RPCs, how many entities to return per RPC.
    """

    api_endpoint: str = _DEFAULT_API_ENDPOINT
    user_agent: str = _USER_AGENT
    page_size: int = _default_page_size


def set_defaults(config: Config) -> None:
    """Set global defaults for operations with Google Generative AI API."""
    global _config
    _config = config
    _set_default_retriever(build_semantic_retriever())
    _set_default_text_service(build_text_service())


_config = Config()


def build_semantic_retriever() -> genai.RetrieverServiceClient:
    return genai.RetrieverServiceClient(
        client_info=gapic_v1.client_info.ClientInfo(user_agent=_USER_AGENT),
        client_options=client_options_lib.ClientOptions(
            api_endpoint=_config.api_endpoint
        ),
    )


_default_retriever: genai.RetrieverServiceClient = build_semantic_retriever()


def _set_default_retriever(retriever: genai.RetrieverServiceClient) -> None:
    global _default_retriever
    _default_retriever = retriever


def build_text_service() -> genai.TextServiceClient:
    return genai.TextServiceClient(
        client_info=gapic_v1.client_info.ClientInfo(user_agent=_USER_AGENT),
        client_options=client_options_lib.ClientOptions(
            api_endpoint=_config.api_endpoint
        ),
    )


_default_text_service: genai.TextServiceClient = build_text_service()


def _set_default_text_service(text_service: genai.TextServiceClient) -> None:
    global _default_text_service
    _default_text_service = text_service


def list_corpora(
    *,
    client: Optional[genai.RetrieverServiceClient] = None,
) -> Iterator[Corpus]:
    if client is None:
        client = _default_retriever
    for corpus in client.list_corpora(
        genai.ListCorporaRequest(page_size=_config.page_size)
    ):
        yield Corpus.from_corpus(corpus)


def get_corpus(
    *,
    corpus_id: str,
    client: Optional[genai.RetrieverServiceClient] = None,
) -> Optional[Corpus]:
    if client is None:
        client = _default_retriever
    try:
        corpus = client.get_corpus(
            genai.GetCorpusRequest(name=str(EntityName(corpus_id=corpus_id)))
        )
        return Corpus.from_corpus(corpus)
    except Exception:
        return None


def create_corpus(
    *,
    corpus_id: Optional[str] = None,
    display_name: Optional[str] = None,
    client: Optional[genai.RetrieverServiceClient] = None,
) -> Corpus:
    if client is None:
        client = _default_retriever

    name: Optional[str]
    if corpus_id is not None:
        name = str(EntityName(corpus_id=corpus_id))
    else:
        name = None

    new_display_name = display_name or f"Untitled {datetime.datetime.now()}"

    new_corpus = client.create_corpus(
        genai.CreateCorpusRequest(
            corpus=genai.Corpus(name=name, display_name=new_display_name)
        )
    )

    return Corpus.from_corpus(new_corpus)


def delete_corpus(
    *,
    corpus_id: str,
    client: Optional[genai.RetrieverServiceClient] = None,
) -> None:
    if client is None:
        client = _default_retriever
    client.delete_corpus(
        genai.DeleteCorpusRequest(name=str(EntityName(corpus_id=corpus_id)), force=True)
    )


def list_documents(
    *,
    corpus_id: str,
    client: Optional[genai.RetrieverServiceClient] = None,
) -> Iterator[Document]:
    if client is None:
        client = _default_retriever
    for document in client.list_documents(
        genai.ListDocumentsRequest(
            parent=str(EntityName(corpus_id=corpus_id)), page_size=_default_page_size
        )
    ):
        yield Document.from_document(document)


def get_document(
    *,
    corpus_id: str,
    document_id: str,
    client: Optional[genai.RetrieverServiceClient] = None,
) -> Optional[Document]:
    if client is None:
        client = _default_retriever
    try:
        document = client.get_document(
            genai.GetDocumentRequest(
                name=str(EntityName(corpus_id=corpus_id, document_id=document_id))
            )
        )
        return Document.from_document(document)
    except Exception:
        return None


def create_document(
    *,
    corpus_id: str,
    document_id: Optional[str] = None,
    display_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    client: Optional[genai.RetrieverServiceClient] = None,
) -> Document:
    if client is None:
        client = _default_retriever
    name: Optional[str]
    if document_id is not None:
        name = str(EntityName(corpus_id=corpus_id, document_id=document_id))
    else:
        name = None

    new_display_name = display_name or f"Untitled {datetime.datetime.now()}"
    new_metadatas = _convert_to_metadata(metadata) if metadata else None

    new_document = client.create_document(
        genai.CreateDocumentRequest(
            parent=str(EntityName(corpus_id=corpus_id)),
            document=genai.Document(
                name=name, display_name=new_display_name, custom_metadata=new_metadatas
            ),
        )
    )

    return Document.from_document(new_document)


def delete_document(
    *,
    corpus_id: str,
    document_id: str,
    client: Optional[genai.RetrieverServiceClient] = None,
) -> None:
    if client is None:
        client = _default_retriever
    client.delete_document(
        genai.DeleteDocumentRequest(
            name=str(EntityName(corpus_id=corpus_id, document_id=document_id)),
            force=True,
        )
    )


def batch_create_chunk(
    *,
    corpus_id: str,
    document_id: str,
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    client: Optional[genai.RetrieverServiceClient] = None,
) -> List[genai.Chunk]:
    if client is None:
        client = _default_retriever
    if metadatas is None:
        metadatas = [{} for _ in texts]
    if len(texts) != len(metadatas):
        raise ValueError(
            f"metadatas's length {len(metadatas)} and texts's length {len(texts)} are mismatched"
        )

    doc_name = str(EntityName(corpus_id=corpus_id, document_id=document_id))

    created_chunks: List[genai.Chunk] = []

    batch_request = genai.BatchCreateChunksRequest(
        parent=doc_name,
        requests=[],
    )
    for text, metadata in zip(texts, metadatas):
        batch_request.requests.append(
            genai.CreateChunkRequest(
                parent=doc_name,
                chunk=genai.Chunk(
                    data=genai.ChunkData(string_value=text),
                    custom_metadata=_convert_to_metadata(metadata),
                ),
            )
        )

        if len(batch_request.requests) >= _MAX_REQUEST_PER_BATCH:
            response = client.batch_create_chunks(batch_request)
            created_chunks.extend(list(response.chunks))
            # Prepare a new batch for next round.
            batch_request = genai.BatchCreateChunksRequest(
                parent=doc_name,
                requests=[],
            )

    # Process left over.
    if len(batch_request.requests) > 0:
        response = client.batch_create_chunks(batch_request)
        created_chunks.extend(list(response.chunks))

    return created_chunks


def delete_chunk(
    *,
    corpus_id: str,
    document_id: str,
    chunk_id: str,
    client: Optional[genai.RetrieverServiceClient] = None,
) -> None:
    if client is None:
        client = _default_retriever
    client.delete_chunk(
        genai.DeleteChunkRequest(
            name=str(
                EntityName(
                    corpus_id=corpus_id, document_id=document_id, chunk_id=chunk_id
                )
            )
        )
    )


def query_corpus(
    *,
    corpus_id: str,
    query: str,
    k: int = 4,
    filter: Optional[Dict[str, Any]] = None,
    client: Optional[genai.RetrieverServiceClient] = None,
) -> List[genai.RelevantChunk]:
    if client is None:
        client = _default_retriever
    response = client.query_corpus(
        genai.QueryCorpusRequest(
            name=str(EntityName(corpus_id=corpus_id)),
            query=query,
            metadata_filters=_convert_filter(filter),
            results_count=k,
        )
    )
    return list(response.relevant_chunks)


def query_document(
    *,
    corpus_id: str,
    document_id: str,
    query: str,
    k: int = 4,
    filter: Optional[Dict[str, Any]] = None,
    client: Optional[genai.RetrieverServiceClient] = None,
) -> List[genai.RelevantChunk]:
    if client is None:
        client = _default_retriever
    response = client.query_document(
        genai.QueryDocumentRequest(
            name=str(EntityName(corpus_id=corpus_id, document_id=document_id)),
            query=query,
            metadata_filters=_convert_filter(filter),
            results_count=k,
        )
    )
    return list(response.relevant_chunks)


@dataclass
class Passage:
    text: str
    ids: List[str]


@dataclass
class TextAnswer:
    answer: str
    attributed_passages: List[Passage]
    answerable_probability: float


def generate_text_answer(
    *,
    prompt: str,
    passages: List[str],
    answer_style: genai.AnswerStyle,
    client: Optional[genai.TextServiceClient] = None,
) -> TextAnswer:
    if client is None:
        client = _default_text_service
    response = client.generate_text_answer(
        genai.GenerateTextAnswerRequest(
            question=genai.TextPrompt(text=prompt),
            model=_default_text_service_model,
            answer_style=answer_style,
            grounding_source=genai.GroundingSource(
                passages=genai.InlinePassages(
                    passages=[
                        genai.InlinePassage(
                            text=chunk,
                            id=str(uuid.uuid4()),
                        )
                        for chunk in passages
                    ]
                )
            ),
        )
    )
    return TextAnswer(
        answer=response.answer.output,
        attributed_passages=[
            Passage(text=passage.text, ids=list(passage.passage_ids))
            for passage in response.attributed_passages
        ],
        answerable_probability=response.answerable_probability,
    )


def _convert_to_metadata(metadata: Dict[str, Any]) -> List[genai.CustomMetadata]:
    cs: List[genai.CustomMetadata] = []
    for key, value in metadata.items():
        if isinstance(value, str):
            c = genai.CustomMetadata(key=key, string_value=value)
        elif isinstance(value, (float, int)):
            c = genai.CustomMetadata(key=key, numeric_value=value)
        else:
            raise ValueError(f"Metadata value {value} is not supported")

        cs.append(c)
    return cs


def _convert_filter(fs: Optional[Dict[str, Any]]) -> List[genai.MetadataFilter]:
    if fs is None:
        return []
    assert isinstance(fs, dict)

    filters: List[genai.MetadataFilter] = []
    for key, value in fs.items():
        if isinstance(value, str):
            condition = genai.Condition(
                operation=genai.Condition.Operator.EQUAL, string_value=value
            )
        elif isinstance(value, (float, int)):
            condition = genai.Condition(
                operation=genai.Condition.Operator.EQUAL, numeric_value=value
            )
        else:
            raise ValueError(f"Filter value {value} is not supported")

        filters.append(genai.MetadataFilter(key=key, conditions=[condition]))

    return filters

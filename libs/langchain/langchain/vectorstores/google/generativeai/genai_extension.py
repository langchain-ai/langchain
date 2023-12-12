"""Temporary high-level library of the Google GenerativeAI API.

The content of this file should eventually go into the Python package
google.generativeai.
"""

import datetime
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, MutableSequence, Optional

import google.ai.generativelanguage as genai
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as gapi_exception
from google.api_core import gapic_v1
from google.auth import credentials, exceptions
from google.protobuf import timestamp_pb2

import langchain

_logger = logging.getLogger(__name__)
_DEFAULT_API_ENDPOINT = "generativelanguage.googleapis.com"
_USER_AGENT = f"langchain/{langchain.__version__}"
_DEFAULT_PAGE_SIZE = 20
_DEFAULT_GENERATE_SERVICE_MODEL = "models/aqa"
_MAX_REQUEST_PER_CHUNK = 100
_NAME_REGEX = re.compile(r"^corpora/([^/]+?)(/documents/([^/]+?)(/chunks/([^/]+?))?)?$")


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
        matched = _NAME_REGEX.match(encoded)
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
        testing: Are the unit tests running?
    """

    api_endpoint: str = _DEFAULT_API_ENDPOINT
    user_agent: str = _USER_AGENT
    page_size: int = _DEFAULT_PAGE_SIZE
    testing: bool = False


def set_defaults(config: Config) -> None:
    """Set global defaults for operations with Google Generative AI API."""
    global _config
    _config = config


_config = Config()


class TestCredentials(credentials.Credentials):
    """Credentials that do not provide any authentication information.

    Useful for unit tests where the credentials are not used.
    """

    @property
    def expired(self) -> bool:
        """Returns `False`, test credentials never expire."""
        return False

    @property
    def valid(self) -> bool:
        """Returns `True`, test credentials are always valid."""
        return True

    def refresh(self, request: Any) -> None:
        """Raises :class:``InvalidOperation``, test credentials cannot be
        refreshed.
        """
        raise exceptions.InvalidOperation("Test credentials cannot be refreshed.")

    def apply(self, headers: Any, token: Any = None) -> None:
        """Anonymous credentials do nothing to the request.

        The optional ``token`` argument is not supported.

        Raises:
            google.auth.exceptions.InvalidValue: If a token was specified.
        """
        if token is not None:
            raise exceptions.InvalidValue("Test credentials don't support tokens.")

    def before_request(self, request: Any, method: Any, url: Any, headers: Any) -> None:
        """Test credentials do nothing to the request."""


def _get_test_credentials() -> Optional[credentials.Credentials]:
    """Returns a fake credential for testing or None.

    If _config.testing is True, a fake credential is returned.
    Otherwise, we are in a real environment and a None is returned.

    If None is passed to the clients later on, the actual credentials will be
    inferred by the rules specified in google.auth package.
    """
    return TestCredentials() if _config.testing else None


def build_semantic_retriever() -> genai.RetrieverServiceClient:
    credentials = _get_test_credentials()
    return genai.RetrieverServiceClient(
        credentials=credentials,
        client_info=gapic_v1.client_info.ClientInfo(user_agent=_USER_AGENT),
        client_options=client_options_lib.ClientOptions(
            api_endpoint=_config.api_endpoint
        ),
    )


def build_generative_service() -> genai.GenerativeServiceClient:
    credentials = _get_test_credentials()
    return genai.GenerativeServiceClient(
        credentials=credentials,
        client_info=gapic_v1.client_info.ClientInfo(user_agent=_USER_AGENT),
        client_options=client_options_lib.ClientOptions(
            api_endpoint=_config.api_endpoint
        ),
    )


def list_corpora(
    *,
    client: genai.RetrieverServiceClient,
) -> Iterator[Corpus]:
    for corpus in client.list_corpora(
        genai.ListCorporaRequest(page_size=_config.page_size)
    ):
        yield Corpus.from_corpus(corpus)


def get_corpus(
    *,
    corpus_id: str,
    client: genai.RetrieverServiceClient,
) -> Optional[Corpus]:
    try:
        corpus = client.get_corpus(
            genai.GetCorpusRequest(name=str(EntityName(corpus_id=corpus_id)))
        )
        return Corpus.from_corpus(corpus)
    except Exception as e:
        # If the corpus does not exist, the server returns a permission error.
        if not isinstance(e, gapi_exception.PermissionDenied):
            raise
        _logger.warning(f"Corpus {corpus_id} not found: {e}")
        return None


def create_corpus(
    *,
    corpus_id: Optional[str] = None,
    display_name: Optional[str] = None,
    client: genai.RetrieverServiceClient,
) -> Corpus:
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
    client: genai.RetrieverServiceClient,
) -> None:
    client.delete_corpus(
        genai.DeleteCorpusRequest(name=str(EntityName(corpus_id=corpus_id)), force=True)
    )


def list_documents(
    *,
    corpus_id: str,
    client: genai.RetrieverServiceClient,
) -> Iterator[Document]:
    for document in client.list_documents(
        genai.ListDocumentsRequest(
            parent=str(EntityName(corpus_id=corpus_id)), page_size=_DEFAULT_PAGE_SIZE
        )
    ):
        yield Document.from_document(document)


def get_document(
    *,
    corpus_id: str,
    document_id: str,
    client: genai.RetrieverServiceClient,
) -> Optional[Document]:
    try:
        document = client.get_document(
            genai.GetDocumentRequest(
                name=str(EntityName(corpus_id=corpus_id, document_id=document_id))
            )
        )
        return Document.from_document(document)
    except Exception as e:
        if not isinstance(e, gapi_exception.NotFound):
            raise
        _logger.warning(f"Document {document_id} in corpus {corpus_id} not found: {e}")
        return None


def create_document(
    *,
    corpus_id: str,
    document_id: Optional[str] = None,
    display_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    client: genai.RetrieverServiceClient,
) -> Document:
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
    client: genai.RetrieverServiceClient,
) -> None:
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
    client: genai.RetrieverServiceClient,
) -> List[genai.Chunk]:
    if metadatas is None:
        metadatas = [{} for _ in texts]
    if len(texts) != len(metadatas):
        raise ValueError(
            f"metadatas's length {len(metadatas)} and "
            f"texts's length {len(texts)} are mismatched"
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

        if len(batch_request.requests) >= _MAX_REQUEST_PER_CHUNK:
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
    client: genai.RetrieverServiceClient,
) -> None:
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
    client: genai.RetrieverServiceClient,
) -> List[genai.RelevantChunk]:
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
    client: genai.RetrieverServiceClient,
) -> List[genai.RelevantChunk]:
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
    id: str


@dataclass
class GroundedAnswer:
    answer: str
    attributed_passages: List[Passage]
    answerable_probability: Optional[float]


@dataclass
class GenerateAnswerError(Exception):
    finish_reason: genai.Candidate.FinishReason
    finish_message: str
    safety_ratings: MutableSequence[genai.SafetyRating]

    def __str__(self) -> str:
        return (
            f"finish_reason: {self.finish_reason.name} "
            f"finish_message: {self.finish_message} "
            f"safety ratings: {self.safety_ratings}"
        )


def generate_answer(
    *,
    prompt: str,
    passages: List[str],
    answer_style: int = genai.GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE,
    safety_settings: List[genai.SafetySetting] = [],
    temperature: Optional[float] = None,
    client: genai.GenerativeServiceClient,
) -> GroundedAnswer:
    # TODO: Consider passing in the corpus ID instead of the actual
    # passages.
    response = client.generate_answer(
        genai.GenerateAnswerRequest(
            contents=[
                genai.Content(parts=[genai.Part(text=prompt)]),
            ],
            model=_DEFAULT_GENERATE_SERVICE_MODEL,
            answer_style=answer_style,
            safety_settings=safety_settings,
            temperature=temperature,
            inline_passages=genai.GroundingPassages(
                passages=[
                    genai.GroundingPassage(
                        # IDs here takes alphanumeric only. No dashes allowed.
                        id=str(index),
                        content=genai.Content(parts=[genai.Part(text=chunk)]),
                    )
                    for index, chunk in enumerate(passages)
                ]
            ),
        )
    )

    if response.answer.finish_reason != genai.Candidate.FinishReason.STOP:
        raise GenerateAnswerError(
            finish_reason=response.answer.finish_reason,
            finish_message=response.answer.finish_message,
            safety_ratings=response.answer.safety_ratings,
        )

    assert len(response.answer.content.parts) == 1
    return GroundedAnswer(
        answer=response.answer.content.parts[0].text,
        attributed_passages=[
            Passage(
                text=passage.content.parts[0].text,
                id=passage.source_id.grounding_passage.passage_id,
            )
            for passage in response.answer.grounding_attributions
            if len(passage.content.parts) > 0
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

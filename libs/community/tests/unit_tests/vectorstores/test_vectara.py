import os
from typing import Generator, cast
from unittest.mock import MagicMock, mock_open, patch

import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores.vectara import (
    CoreDocument,
    File,
    Vectara,
    VectaraQueryConfig,
)


@pytest.fixture
def mock_env() -> Generator[None, None, None]:
    """Fixture to temporarily set/unset environment variables."""
    original_api_key = os.environ.pop("VECTARA_API_KEY", None)
    yield
    if original_api_key is not None:
        os.environ["VECTARA_API_KEY"] = original_api_key


def test_vectara_init_no_key(mock_env: None) -> None:
    """If no key is given or set in environment, constructor should raise ValueError."""
    with pytest.raises(ValueError, match="unable to find Vectara API key."):
        Vectara()


def test_vectara_init_key_in_constructor() -> None:
    """If a key is passed, it should succeed."""
    vectara = Vectara(vectara_api_key="test-key")
    assert vectara._vectara_api_key == "test-key"


def test_vectara_init_key_in_env(monkeypatch: MonkeyPatch) -> None:
    """If a key is provided in the environment, constructor should use it."""
    monkeypatch.setenv("VECTARA_API_KEY", "env-key")
    vectara = Vectara()
    assert vectara._vectara_api_key == "env-key"


def test_vectara_init_strip_trailing_slash() -> None:
    """Base URL should have trailing slash stripped."""
    vectara = Vectara(vectara_api_key="123", vectara_base_url="https://api.vectara.io/")
    assert vectara._base_url == "https://api.vectara.io"


def test_vectara_init_verify_ssl_true() -> None:
    """Check default verify_ssl is True."""
    vectara = Vectara(vectara_api_key="dummy_key")
    assert vectara._verify_ssl is True


def test_vectara_init_verify_ssl_false() -> None:
    """If verify_ssl=False is passed, ensure it's stored."""
    vectara = Vectara(vectara_api_key="dummy_key", vectara_verify_ssl=False)
    assert vectara._verify_ssl is False


@patch("requests.Session.post")
def test_vectara_usage_of_base_url_and_ssl(mock_post: MagicMock) -> None:
    """
    Confirm that calls to _session.post use self._base_url and _verify_ssl.
    We'll specifically check that the URL is correct and the verify param
    matches _verify_ssl.
    """
    vectara = Vectara(
        vectara_api_key="dummy_key",
        vectara_base_url="https://custom.domain/vectara/api/",
        vectara_verify_ssl=False,
    )

    vectara.vectara_query("Hello world", config=VectaraQueryConfig())
    mock_post.assert_called_once()
    _, call_kwargs = mock_post.call_args

    assert call_kwargs["url"].startswith("https://custom.domain/vectara/api")
    assert call_kwargs["verify"] is False


def test_get_post_headers() -> None:
    """Ensure _get_post_headers returns correct headers."""
    vectara = Vectara(vectara_api_key="dummy_key")
    headers = vectara._get_post_headers()
    assert headers["x-api-key"] == "dummy_key"
    assert headers["Content-Type"] == "application/json"
    assert headers["X-Source"] == "langchain"


def test_generate_doc_id() -> None:
    """Check MD5 generation for doc IDs."""
    vectara = Vectara("dummy")
    doc_id = vectara._generate_doc_id("hello world")
    # MD5 of "hello world" -> 5eb63bbbe01eeed093cb22bb8f5acdc3
    assert doc_id == "5eb63bbbe01eeed093cb22bb8f5acdc3"


def test_delete_doc_no_docid() -> None:
    """_delete_doc with no doc_id should raise ValueError."""
    vectara = Vectara(vectara_api_key="abc")
    with pytest.raises(ValueError, match="Document ID cannot be empty."):
        vectara._delete_doc("", "test_corpus")


def test_delete_doc_no_corpus() -> None:
    """_delete_doc with no corpus should raise ValueError."""
    vectara = Vectara(vectara_api_key="abc")
    with pytest.raises(ValueError, match="Corpus key cannot be empty."):
        vectara._delete_doc("docid", "")


@patch("requests.Session.delete")
def test_delete_doc_success(mock_delete: MagicMock) -> None:
    """_delete_doc returns True if response code=204."""
    mock_delete.return_value.status_code = 204
    mock_delete.return_value.json.return_value = {}

    vectara = Vectara(vectara_api_key="abc")
    result = vectara._delete_doc("docid", "corpus123")
    assert result is True
    mock_delete.assert_called_once()


@patch("requests.Session.delete")
def test_delete_doc_failure(mock_delete: MagicMock, caplog: LogCaptureFixture) -> None:
    """_delete_doc returns False and logs error if status != 204."""
    mock_delete.return_value.status_code = 400
    mock_delete.return_value.json.return_value = {"error": "some error"}

    vectara = Vectara(vectara_api_key="abc")
    result = vectara._delete_doc("docid", "corpus123")
    assert not result
    assert "Delete request failed for doc_id = docid" in caplog.text


@patch("requests.Session.post")
def test_index_doc_no_corpus(mock_post: MagicMock) -> None:
    """_index_doc with no corpus key should raise ValueError."""
    vectara = Vectara(vectara_api_key="abc")
    doc = CoreDocument(id="abc")
    with pytest.raises(ValueError, match="Corpus key cannot be empty."):
        vectara._index_doc(doc, corpus_key="")


@patch("requests.Session.post")
def test_index_doc_success(mock_post: MagicMock) -> None:
    """_index_doc returns 'SUCCEEDED' if status=201."""
    mock_post.return_value.status_code = 201
    mock_post.return_value.json.return_value = {}

    vectara = Vectara(vectara_api_key="abc")
    doc = CoreDocument(id="abc")
    result = vectara._index_doc(doc, corpus_key="mycorpus")
    assert result == "SUCCEEDED"


@patch("requests.Session.post")
def test_index_doc_already_exists(mock_post: MagicMock) -> None:
    """_index_doc returns 'ALREADY_EXISTS' if status=409 or 412."""
    mock_post.return_value.status_code = 409
    mock_post.return_value.json.return_value = {}

    vectara = Vectara(vectara_api_key="abc")
    doc = CoreDocument(id="abc")
    result = vectara._index_doc(doc, corpus_key="mycorpus")
    assert result == "ALREADY_EXISTS"


@patch("requests.Session.post")
def test_index_doc_error_message(mock_post: MagicMock) -> None:
    """_index_doc returns error message for other status codes."""
    mock_post.return_value.status_code = 400
    mock_post.return_value.json.return_value = {
        "field_errors": {"fieldA": "invalid format"},
        "messages": ["Something went wrong"],
    }

    vectara = Vectara(vectara_api_key="abc")
    doc = CoreDocument(id="abc")
    result = vectara._index_doc(doc, corpus_key="mycorpus")
    assert "fieldA: invalid format; Something went wrong" in result


def test_delete_no_corpus_key() -> None:
    """delete() with no corpus_key provided should raise ValueError."""
    vectara = Vectara(vectara_api_key="abc")
    with pytest.raises(ValueError, match="Corpus key cannot be empty."):
        vectara.delete(ids=["doc1", "doc2"])


@patch.object(Vectara, "_delete_doc", return_value=True)
def test_delete_ids(mock_del: MagicMock) -> None:
    """delete() with doc IDs calls _delete_doc for each doc."""
    vectara = Vectara(vectara_api_key="abc")
    success = vectara.delete(ids=["docA", "docB"], corpus_key="xyz")
    assert success is True
    assert mock_del.call_count == 2


@patch.object(Vectara, "_delete_doc", side_effect=[True, False])
def test_delete_ids_partial_failure(mock_del: MagicMock) -> None:
    """delete() returns False if any doc fails to delete."""
    vectara = Vectara(vectara_api_key="abc")
    success = vectara.delete(ids=["docA", "docB"], corpus_key="xyz")
    assert success is False


def test_add_files_not_exist(caplog: LogCaptureFixture) -> None:
    """add_files() logs error if file does not exist."""
    vectara = Vectara(vectara_api_key="abc")
    f = File(file_path="does_not_exist.pdf")
    doc_ids = vectara.add_files([f], "mycorpus")
    assert len(doc_ids) == 0
    assert "File does_not_exist.pdf does not exist, skipping" in caplog.text


@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="fake data")
@patch("requests.Session.post")
def test_add_files_success(
    mock_post: MagicMock, mock_file: MagicMock, mock_exists: MagicMock
) -> None:
    """add_files returns doc ID if response=201."""
    mock_post.return_value.status_code = 201
    mock_post.return_value.json.return_value = {"id": "mydoc123"}

    vectara = Vectara(vectara_api_key="abc")
    f = File(file_path="myfile.pdf")
    doc_ids = vectara.add_files([f], "mycorpus")
    assert doc_ids == ["mydoc123"]


@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="fake data")
@patch("requests.Session.post")
def test_add_files_failure(
    mock_post: MagicMock,
    mock_file: MagicMock,
    mock_exist: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """add_files logs error if response !=201."""
    mock_post.return_value.status_code = 400
    mock_post.return_value.json.return_value = {
        "field_errors": {"file": "Invalid"},
        "messages": ["Bad request"],
    }

    vectara = Vectara(vectara_api_key="abc")
    f = File(file_path="myfile.pdf")
    doc_ids = vectara.add_files([f], "mycorpus")
    assert doc_ids == []
    assert "File upload failed (400), reason: file: Invalid; Bad request" in caplog.text


def test_add_texts_no_corpus() -> None:
    """add_texts with no corpus key => ValueError."""
    vectara = Vectara(vectara_api_key="abc")
    with pytest.raises(ValueError, match="Missing required parameter: 'corpus_key'"):
        vectara.add_texts(["hello"], doc_type="core")


def test_add_texts_invalid_doc_type() -> None:
    """add_texts with invalid doc_type => ValueError."""
    vectara = Vectara(vectara_api_key="abc")
    with pytest.raises(
        ValueError, match="Invalid doc_type. Must be 'core' or 'structured'"
    ):
        vectara.add_texts(["hello"], corpus_key="xyz", doc_type="unsupported")


@patch.object(Vectara, "_index_doc", return_value="SUCCEEDED")
def test_add_texts_core(mock_index: MagicMock) -> None:
    """add_texts with doc_type=core => calls _index_doc once per text."""
    vectara = Vectara(vectara_api_key="abc")
    res = vectara.add_texts(["Hello", "World"], corpus_key="xyz", doc_type="core")
    assert len(res) == 2
    # Should call _index_doc 2 times
    assert mock_index.call_count == 2


@patch.object(Vectara, "_index_doc", return_value="ALREADY_EXISTS")
@patch.object(Vectara, "_delete_doc")
def test_add_texts_already_exists(
    mock_del: MagicMock, mock_index: MagicMock, caplog: LogCaptureFixture
) -> None:
    """If doc already exists, we try deleting then indexing again."""
    vectara = Vectara(vectara_api_key="abc")
    vectara.add_texts(["Hello"], corpus_key="xyz", doc_type="core")

    # The code logs "Unable to index document {doc_id}. Reason: ALREADY_EXISTS"
    assert "Unable to index document" in caplog.text
    assert "Reason: ALREADY_EXISTS" in caplog.text
    mock_del.assert_called_once()


@patch.object(Vectara, "_index_doc", return_value="ERROR: something")
def test_add_texts_index_error(
    mock_index: MagicMock, caplog: LogCaptureFixture
) -> None:
    """If indexing fails with a string error, we log it and skip the doc."""
    vectara = Vectara(vectara_api_key="abc")
    res = vectara.add_texts(["Hello"], corpus_key="xyz", doc_type="core")
    assert len(res) == 0
    assert "Unable to index document" in caplog.text


def test_get_query_body() -> None:
    """Check if _get_query_body returns the correct structure."""
    vectara = Vectara("abc")
    config = VectaraQueryConfig(stream_response=True, save_history=True, chat=True)
    body = vectara._get_query_body("Hello", config)
    assert body["query"] == "Hello"
    assert body["stream_response"] is True
    assert body["save_history"] is True
    assert body["chat"] == {"store": True}


@patch("requests.Session.post")
def test_vectara_query_fail(mock_post: MagicMock, caplog: LogCaptureFixture) -> None:
    """vectara_query returns empty list if status != 200."""
    mock_post.return_value.status_code = 400
    mock_post.return_value.json.return_value = {
        "field_errors": {"Q": "Bad"},
        "messages": ["Invalid request"],
    }
    vectara = Vectara("abc")
    res = vectara.vectara_query("What is love?", VectaraQueryConfig())
    assert res == []
    assert "Query failed (code 400), reason Q: Bad; Invalid request" in caplog.text


@patch("requests.Session.post")
def test_vectara_query_success(mock_post: MagicMock) -> None:
    """vectara_query returns documents if status=200."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "search_results": [
            {"text": "doc1 text", "score": 0.9, "document_metadata": {"foo": "bar"}},
            {"text": "doc2 text", "score": 0.8},
        ]
    }
    vectara = Vectara("abc")
    # Provide generation=None so we don't attempt to create a summary doc with None
    res = vectara.vectara_query("test", VectaraQueryConfig(generation=None))
    assert len(res) == 2
    assert res[0][0].page_content == "doc1 text"
    assert res[0][0].metadata == {"foo": "bar"}
    assert res[0][1] == 0.9


@patch("requests.Session.post")
def test_vectara_query_generation(mock_post: MagicMock) -> None:
    """
    If generation config or chat is set, we add an extra doc with summary or answer.
    """
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "search_results": [{"text": "doc1 text", "score": 0.9}],
        "summary": "This is the summary",
        "factual_consistency_score": 0.85,
    }
    vectara = Vectara("abc")
    cfg = VectaraQueryConfig()
    res = vectara.vectara_query("test", cfg)
    # Expect 2 items: doc, summary
    assert len(res) == 2
    summary_doc, summary_score = res[-1]
    assert summary_doc.page_content == "This is the summary"
    assert summary_doc.metadata == {"summary": True, "fcs": (0.85,)}


@patch("requests.Session.post")
def test_vectara_query_chat(mock_post: MagicMock) -> None:
    """If config.chat=True, we look for 'answer' + chat_id in the response."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "search_results": [],
        "answer": "Chat answer",
        "chat_id": "chat-123",
        "factual_consistency_score": 0.95,
    }
    vectara = Vectara("abc")
    cfg = VectaraQueryConfig(chat=True)
    res = vectara.vectara_query("test", cfg)
    # The only doc is the chat answer
    chat_doc, chat_score = res[-1]
    assert chat_doc.page_content == "Chat answer"
    assert chat_doc.metadata == {"chat_convo_id": "chat-123", "fcs": (0.95,)}


@patch.object(Vectara, "vectara_query", return_value=[(Document("doc"), 0.9)])
def test_similarity_search_with_score(mock_vq: MagicMock) -> None:
    vectara = Vectara("abc")
    res = vectara.similarity_search_with_score("hello")
    assert len(res) == 1
    assert mock_vq.called


@patch.object(
    Vectara, "similarity_search_with_score", return_value=[(Document("d"), 0.8)]
)
def test_similarity_search(mock_ssws: MagicMock) -> None:
    vectara = Vectara("abc")
    docs = vectara.similarity_search("hello")
    assert len(docs) == 1
    assert docs[0].page_content == "d"


def test_get_document_no_doc() -> None:
    vectara = Vectara("abc")
    with patch("requests.Session.get") as mock_get:
        mock_get.return_value.status_code = 404
        mock_get.return_value.json.return_value = {"messages": ["Not found"]}
        doc = vectara.get_document("docid", "mycorpus")
        assert doc is None


@patch("requests.Session.get")
def test_get_document_ok(mock_get: MagicMock) -> None:
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "metadata": {"doc_meta": "ABC"},
        "parts": [
            {"text": "hello part1", "metadata": {"k1": "v1"}},
            {"text": "part2 here", "metadata": {"k2": "v2"}},
        ],
    }
    vectara = Vectara("abc")
    doc = vectara.get_document("docid", "mycorpus")
    assert doc is not None
    assert doc.page_content == "hello part1\npart2 here"
    assert doc.metadata["doc_meta"] == "ABC"
    assert doc.metadata["k1"] == "v1"
    assert doc.metadata["k2"] == "v2"


def test_get_by_ids_no_corpus() -> None:
    """get_by_ids with no corpus_key => ValueError."""
    vectara = Vectara("abc")
    with pytest.raises(ValueError, match="Missing required parameter: 'corpus_key'."):
        vectara.get_by_ids(["docid1", "docid2"])


@patch.object(Vectara, "get_document", return_value=Document("abc"))
def test_get_by_ids_ok(mock_gd: MagicMock) -> None:
    vectara = Vectara("abc")
    docs = vectara.get_by_ids(["d1", "d2"], corpus_key="mycorpus")
    assert len(docs) == 2
    assert mock_gd.call_count == 2


def test_from_texts_missing_corpus() -> None:
    with pytest.raises(ValueError, match="Missing required parameter: 'corpus_key'"):
        Vectara.from_texts(["hi"], vectara_api_key="k")


@patch.object(Vectara, "add_texts")
def test_from_texts_ok(mock_add: MagicMock) -> None:
    vectara = Vectara.from_texts(["hi", "world"], vectara_api_key="k", corpus_key="C")
    assert isinstance(vectara, Vectara)
    mock_add.assert_called_once()


def test_from_documents_missing_corpus() -> None:
    docs = [Document("hi"), Document("test")]
    with pytest.raises(ValueError, match="Missing required parameter: 'corpus_key'"):
        Vectara.from_documents(docs, embedding=cast(Embeddings, None))


def test_rag() -> None:
    """Quick test for as_rag, returning a VectaraRAG."""
    vectara = Vectara("abc")
    config = VectaraQueryConfig()
    rag = vectara.as_rag(config)
    assert hasattr(rag, "invoke")
    assert hasattr(rag, "stream")


def test_as_chat() -> None:
    """Ensure as_chat sets config.chat=True."""
    vectara = Vectara("abc")
    config = VectaraQueryConfig(chat=False)
    chat_rag = vectara.as_chat(config)
    assert hasattr(chat_rag, "invoke")
    assert hasattr(chat_rag, "stream")
    assert chat_rag.config.chat is True


def test_as_retriever() -> None:
    """Ensure as_retriever returns a VectaraRetriever."""
    vectara = Vectara("abc")
    ret = vectara.as_retriever()
    from langchain_community.vectorstores.vectara import VectaraRetriever

    assert isinstance(ret, VectaraRetriever)

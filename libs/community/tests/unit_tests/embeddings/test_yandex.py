import os
from unittest import mock
from unittest.mock import MagicMock

import pytest

from langchain_community.embeddings import YandexGPTEmbeddings

YANDEX_MODULE_NAME2 = (
    "yandex.cloud.ai.foundation_models.v1.embedding.embedding_service_pb2_grpc"
)
YANDEX_MODULE_NAME = (
    "yandex.cloud.ai.foundation_models.v1.embedding.embedding_service_pb2"
)


@mock.patch.dict(os.environ, {"YC_API_KEY": "foo"}, clear=True)
def test_init() -> None:
    models = [
        YandexGPTEmbeddings(folder_id="bar"),  # type: ignore[call-arg]
        YandexGPTEmbeddings(  # type: ignore[call-arg]
            query_model_uri="emb://bar/text-search-query/latest",
            doc_model_uri="emb://bar/text-search-doc/latest",
        ),
        YandexGPTEmbeddings(  # type: ignore[call-arg]
            folder_id="bar",
            query_model_name="text-search-query",
            doc_model_name="text-search-doc",
        ),
    ]
    for embeddings in models:
        assert embeddings.model_uri == "emb://bar/text-search-query/latest"
        assert embeddings.doc_model_uri == "emb://bar/text-search-doc/latest"
        assert embeddings.model_name == "text-search-query"
        assert embeddings.doc_model_name == "text-search-doc"


@pytest.mark.parametrize(
    "api_key_or_token", [dict(api_key="bogus"), dict(iam_token="bogus")]
)
@pytest.mark.parametrize(
    "disable_logging",
    [dict(), dict(disable_request_logging=True), dict(disable_request_logging=False)],
)
@mock.patch.dict(os.environ, {}, clear=True)
def test_query_embedding_call(api_key_or_token: dict, disable_logging: dict) -> None:
    absent_yandex_module_stub = MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {
            YANDEX_MODULE_NAME: absent_yandex_module_stub,
            YANDEX_MODULE_NAME2: absent_yandex_module_stub,
            "grpc": MagicMock(),
        },
    ):
        stub = absent_yandex_module_stub.EmbeddingsServiceStub
        request_stub = absent_yandex_module_stub.TextEmbeddingRequest
        args = {"folder_id": "fldr", **api_key_or_token, **disable_logging}
        ygpt = YandexGPTEmbeddings(**args)
        grpc_call_mock = stub.return_value.TextEmbedding
        grpc_call_mock.return_value.embedding = [1, 2, 3]
        act_emb = ygpt.embed_query("nomatter")
        assert act_emb == [1, 2, 3]
        assert len(grpc_call_mock.call_args_list) == 1
        once_called_args = grpc_call_mock.call_args_list[0]
        act_model_uri = request_stub.call_args_list[0].kwargs["model_uri"]
        assert "fldr" in act_model_uri
        assert "query" in act_model_uri
        assert "doc" not in act_model_uri
        act_text = request_stub.call_args_list[0].kwargs["text"]
        assert act_text == "nomatter"
        act_metadata = once_called_args.kwargs["metadata"]
        assert act_metadata
        assert len(act_metadata) > 0
        if disable_logging.get("disable_request_logging"):
            assert ("x-data-logging-enabled", "false") in act_metadata


@pytest.mark.parametrize(
    "api_key_or_token", [dict(api_key="bogus"), dict(iam_token="bogus")]
)
@pytest.mark.parametrize(
    "disable_logging",
    [dict(), dict(disable_request_logging=True), dict(disable_request_logging=False)],
)
@mock.patch.dict(os.environ, {}, clear=True)
def test_doc_embedding_call(api_key_or_token: dict, disable_logging: dict) -> None:
    absent_yandex_module_stub = MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {
            YANDEX_MODULE_NAME: absent_yandex_module_stub,
            YANDEX_MODULE_NAME2: absent_yandex_module_stub,
            "grpc": MagicMock(),
        },
    ):
        stub = absent_yandex_module_stub.EmbeddingsServiceStub
        request_stub = absent_yandex_module_stub.TextEmbeddingRequest
        args = {"folder_id": "fldr", **api_key_or_token, **disable_logging}
        ygpt = YandexGPTEmbeddings(**args)
        grpc_call_mock = stub.return_value.TextEmbedding
        foo_emb = mock.Mock()
        foo_emb.embedding = [1, 2, 3]
        bar_emb = mock.Mock()
        bar_emb.embedding = [4, 5, 6]
        grpc_call_mock.side_effect = [foo_emb, bar_emb]
        act_emb = ygpt.embed_documents(["foo", "bar"])
        assert act_emb == [[1, 2, 3], [4, 5, 6]]
        assert len(grpc_call_mock.call_args_list) == 2
        for i, txt in enumerate(["foo", "bar"]):
            act_model_uri = request_stub.call_args_list[i].kwargs["model_uri"]
            act_text = request_stub.call_args_list[i].kwargs["text"]
            call_args = grpc_call_mock.call_args_list[i]
            act_metadata = call_args.kwargs["metadata"]
            assert "fldr" in act_model_uri
            assert "query" not in act_model_uri
            assert "doc" in act_model_uri
            assert act_text == txt
            assert act_metadata
            assert len(act_metadata) > 0
            if disable_logging.get("disable_request_logging"):
                assert ("x-data-logging-enabled", "false") in call_args.kwargs[
                    "metadata"
                ]

import os
from unittest import mock

import pytest

from langchain_community.embeddings import YandexGPTEmbeddings


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
    with mock.patch(
        "yandex.cloud.ai.foundation_models.v1.embedding.embedding_service_pb2_grpc.EmbeddingsServiceStub"
    ) as stub:
        args = {"folder_id": "fldr", **api_key_or_token, **disable_logging}
        ygpt = YandexGPTEmbeddings(**args)
        grpc_call_mock = stub.return_value.TextEmbedding
        grpc_call_mock.return_value.embedding = [1, 2, 3]
        act_emb = ygpt.embed_query("nomatter")
        assert act_emb == [1, 2, 3]
        assert len(grpc_call_mock.call_args_list) == 1
        once_called_args = grpc_call_mock.call_args_list[0]
        assert "fldr" in once_called_args.args[0].model_uri
        assert "query" in once_called_args.args[0].model_uri
        assert "doc" not in once_called_args.args[0].model_uri
        assert once_called_args.args[0].text == "nomatter"
        assert once_called_args.kwargs["metadata"]
        assert len(once_called_args.kwargs["metadata"]) > 0
        if disable_logging.get("disable_request_logging"):
            assert ("x-data-logging-enabled", "false") in once_called_args.kwargs[
                "metadata"
            ]


@pytest.mark.parametrize(
    "api_key_or_token", [dict(api_key="bogus"), dict(iam_token="bogus")]
)
@pytest.mark.parametrize(
    "disable_logging",
    [dict(), dict(disable_request_logging=True), dict(disable_request_logging=False)],
)
@mock.patch.dict(os.environ, {}, clear=True)
def test_doc_embedding_call(api_key_or_token: dict, disable_logging: dict) -> None:
    with mock.patch(
        "yandex.cloud.ai.foundation_models.v1.embedding.embedding_service_pb2_grpc.EmbeddingsServiceStub"
    ) as stub:
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
            call_args = grpc_call_mock.call_args_list[i]
            assert "fldr" in call_args.args[0].model_uri
            assert "query" not in call_args.args[0].model_uri
            assert "doc" in call_args.args[0].model_uri
            assert call_args.args[0].text == txt
            assert call_args.kwargs["metadata"]
            assert len(call_args.kwargs["metadata"]) > 0
            if disable_logging.get("disable_request_logging"):
                assert ("x-data-logging-enabled", "false") in call_args.kwargs[
                    "metadata"
                ]

import pytest

from langchain import embeddings
from tests.unit_tests import assert_all_importable

EXPECTED_ALL = [
    "CacheBackedEmbeddings",
]
EXPECTED_DEPRECATED_IMPORTS = [
    "OpenAIEmbeddings",
    "AzureOpenAIEmbeddings",
    "ClarifaiEmbeddings",
    "CohereEmbeddings",
    "DatabricksEmbeddings",
    "ElasticsearchEmbeddings",
    "FastEmbedEmbeddings",
    "HuggingFaceEmbeddings",
    "HuggingFaceInferenceAPIEmbeddings",
    "InfinityEmbeddings",
    "GradientEmbeddings",
    "JinaEmbeddings",
    "LlamaCppEmbeddings",
    "HuggingFaceHubEmbeddings",
    "MlflowAIGatewayEmbeddings",
    "MlflowEmbeddings",
    "ModelScopeEmbeddings",
    "TensorflowHubEmbeddings",
    "SagemakerEndpointEmbeddings",
    "HuggingFaceInstructEmbeddings",
    "MosaicMLInstructorEmbeddings",
    "SelfHostedEmbeddings",
    "SelfHostedHuggingFaceEmbeddings",
    "SelfHostedHuggingFaceInstructEmbeddings",
    "FakeEmbeddings",
    "DeterministicFakeEmbedding",
    "AlephAlphaAsymmetricSemanticEmbedding",
    "AlephAlphaSymmetricSemanticEmbedding",
    "SentenceTransformerEmbeddings",
    "GooglePalmEmbeddings",
    "MiniMaxEmbeddings",
    "VertexAIEmbeddings",
    "BedrockEmbeddings",
    "DeepInfraEmbeddings",
    "EdenAiEmbeddings",
    "DashScopeEmbeddings",
    "EmbaasEmbeddings",
    "OctoAIEmbeddings",
    "SpacyEmbeddings",
    "NLPCloudEmbeddings",
    "GPT4AllEmbeddings",
    "XinferenceEmbeddings",
    "LocalAIEmbeddings",
    "AwaEmbeddings",
    "HuggingFaceBgeEmbeddings",
    "ErnieEmbeddings",
    "JavelinAIGatewayEmbeddings",
    "OllamaEmbeddings",
    "QianfanEmbeddingsEndpoint",
    "JohnSnowLabsEmbeddings",
    "VoyageEmbeddings",
    "BookendEmbeddings",
]


def test_all_imports() -> None:
    assert set(embeddings.__all__) == set(EXPECTED_ALL)
    assert_all_importable(embeddings)


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(embeddings, import_)
            assert "langchain_community" in e
    with pytest.raises(AttributeError):
        getattr(embeddings, "foo")

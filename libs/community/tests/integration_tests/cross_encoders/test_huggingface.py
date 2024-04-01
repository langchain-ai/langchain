"""Test huggingface cross encoders."""

from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def _assert(encoder: HuggingFaceCrossEncoder) -> None:
    query = "I love you"
    texts = ["I love you", "I like you", "I don't like you", "I hate you"]
    output = encoder.score([(query, text) for text in texts])

    for i in range(len(texts) - 1):
        assert output[i] > output[i + 1]


def test_huggingface_cross_encoder() -> None:
    encoder = HuggingFaceCrossEncoder()
    _assert(encoder)


def test_huggingface_cross_encoder_with_designated_model_name() -> None:
    encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    _assert(encoder)

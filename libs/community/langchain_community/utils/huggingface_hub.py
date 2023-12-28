from __future__ import annotations

from importlib.metadata import version

from packaging.version import parse


def is_inference_client_supported() -> bool:
    """Return whether HuggingFace Hub Client library supports InferenceClient."""
    # InferenceAPI was deprecated 0.17.
    # See https://github.com/huggingface/huggingface_hub/commit/0a02b04e6cab31a906ddeaf61fce0d5df4b4f7be.
    _version = parse(version("hugingface_hub"))
    return not (_version.major == 0 and _version.minor < 17)

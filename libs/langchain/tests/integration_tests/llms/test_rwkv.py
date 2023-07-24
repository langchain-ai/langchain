# flake8: noqa
"""Test rwkv wrapper."""
import os
from urllib.request import urlretrieve

from langchain.llms import RWKV
import warnings
import pytest


def _download_model() -> str:
    """Download model.
    From https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth,
    """
    model_url = "https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth"
    tokenizer_url = (
        "https://github.com/BlinkDL/ChatRWKV/blob/main/v2/20B_tokenizer.json?raw=true"
    )
    local_filename = model_url.split("/")[-1]

    if not os.path.exists("20B_tokenizer.json"):
        urlretrieve(tokenizer_url, "20B_tokenizer.json")
    if not os.path.exists(local_filename):
        urlretrieve(model_url, local_filename)

    return local_filename


@pytest.mark.filterwarnings("ignore::UserWarning:")
def test_rwkv_inference() -> None:
    """Test valid gpt4all inference."""
    model_path = _download_model()
    llm = RWKV(model=model_path, tokens_path="20B_tokenizer.json", strategy="cpu fp32")
    output = llm("Say foo:")
    assert isinstance(output, str)

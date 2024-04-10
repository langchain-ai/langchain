# function to download hugging face repository models
import os

import pytest
import torch.cuda
from huggingface_hub import snapshot_download
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from libs.langchain.langchain.chains.llm import LLMChain

from langchain_community.llms.exllamav2 import ExLlamaV2


def download_GPTQ_model(model_name: str, models_dir: str = "./models/") -> str:
    """Download the model from hugging face repository.

    Params:
        model_name: str: the model name to download (repository name).

    Example: "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"
    """
    # Split the model name and create a directory name. Example:
    # "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ" ->
    # "TheBloke_CapybaraHermes-2.5-Mistral-7B-GPTQ"
    _model_name = model_name.split("/")
    _model_name = "_".join(_model_name)
    model_path = os.path.join(models_dir, _model_name)
    if _model_name not in os.listdir(models_dir):
        # download the model
        snapshot_download(
            repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False
        )

    return model_path


@pytest.mark.requires("exllamav2")
@pytest.mark.skipif(
    condition=not torch.cuda.is_available(),
    reason="CUDA is not available. ExllamaV2 requires CUDA.",
)
def test_exllamav2_inference():
    from exllamav2.generator import ExLlamaV2Sampler

    model_path = download_GPTQ_model("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")

    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.85
    settings.top_k = 50
    settings.top_p = 0.8
    settings.token_repetition_penalty = 1.05

    # Verbose is required to pass to the callback manager
    llm = ExLlamaV2(
        model_path=model_path,
        verbose=True,
        settings=settings,
        streaming=False,
        max_new_tokens=10,
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = "What Football team won the UCL in the year 2015?"

    out = llm_chain.invoke({"question": question})
    assert out is not None
    assert len(out) > 0


@pytest.mark.requires("exllamav2")
@pytest.mark.skipif(
    condition=not torch.cuda.is_available(),
    reason="CUDA is not available. ExllamaV2 requires CUDA.",
)
def test_exllamav2_streaming():
    from exllamav2.generator import ExLlamaV2Sampler

    model_path = download_GPTQ_model("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")

    callbacks = [StreamingStdOutCallbackHandler()]

    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.85
    settings.top_k = 50
    settings.top_p = 0.8
    settings.token_repetition_penalty = 1.05

    # Verbose is required to pass to the callback manager
    llm = ExLlamaV2(
        model_path=model_path,
        callbacks=callbacks,
        verbose=True,
        settings=settings,
        streaming=True,
        max_new_tokens=10,
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = (
        "What Football team won the UEFA Champions League in the year the iphone 6s "
        "was released?"
    )

    out = llm_chain.invoke({"question": question})
    assert out is not None
    assert len(out) > 0

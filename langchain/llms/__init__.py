"""Wrappers on top of large language models APIs."""
from langchain.llms.ai21 import AI21
from langchain.llms.cohere import Cohere
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.llms.manifest import ManifestWrapper
from langchain.llms.nlpcloud import NLPCloud
from langchain.llms.openai import OpenAI

__all__ = ["Cohere", "NLPCloud", "OpenAI", "HuggingFaceHub", "ManifestWrapper", "AI21"]

type_to_cls_dict = {
    "ai21": AI21,
    "cohere": Cohere,
    "hugginface_hub": HuggingFaceHub,
    "manifest": ManifestWrapper,
    "nlpcloud": NLPCloud,
    "openai": OpenAI,
}

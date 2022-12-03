"""Set up the package."""
from pathlib import Path

from setuptools import find_packages, setup

with open(Path(__file__).absolute().parents[0] / "langchain" / "VERSION") as _f:
    __version__ = _f.read().strip()

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

LLM_DEPENDENCIES = ["cohere", "openai", "nlpcloud", "huggingface_hub"]
OTHER_DEPENDENCIES = [
    "elasticsearch",
    "google-search-results",
    "wikipedia",
    "faiss-cpu",
    "sentence_transformers",
    "transformers",
    "spacy",
    "nltk",
]


setup(
    name="langchain",
    version=__version__,
    packages=find_packages(),
    description="Building applications with LLMs through composability",
    install_requires=["pydantic", "sqlalchemy", "numpy", "requests", "pyyaml"],
    long_description=long_description,
    license="MIT",
    url="https://github.com/hwchase17/langchain",
    include_package_data=True,
    long_description_content_type="text/markdown",
    extras_require={
        "llms": LLM_DEPENDENCIES,
        "all": LLM_DEPENDENCIES + OTHER_DEPENDENCIES,
    },
)

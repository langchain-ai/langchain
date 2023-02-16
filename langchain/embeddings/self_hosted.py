"""Wrapper around HuggingFace embedding models to perform inference on a self-hosted remote hardware."""
from typing import Any, Callable, List

from pydantic import BaseModel, Extra

from langchain.embeddings.base import Embeddings
from langchain.llms import SelfHostedPipeline


def _embed_documents(pipeline: Any, *args: Any, **kwargs: Any) -> List[List[float]]:
    """Inference function to send to the remote hardware. Accepts a sentence_transformer model_id and
    returns a list of embeddings for each document in the batch.
    """
    return pipeline.encode(*args, **kwargs)


class SelfHostedEmbeddings(SelfHostedPipeline, Embeddings, BaseModel):
    """Wrapper around sentence_transformers embedding models to perform inference on self-hosted remote hardware.
    Supported hardware includes auto-launched instances on AWS, GCP, Azure, and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another cloud like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example using a model load function:
        .. code-block:: python

            from langchain.embeddings import SelfHostedHFEmbeddings
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import runhouse as rh

            gpu = rh.cluster(name="rh-a10x", instance_type="A100:1")
            def get_pipeline():
                model_id = "facebook/bart-large"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(model_id)
                pipe = pipeline(
                    "feature-extraction", model=model, tokenizer=tokenizer
                )
            hf = SelfHostedHuggingFaceEmbeddings(
                model_load_fn=get_pipeline,
                hardware=gpu
                model_reqs=["./", "torch", "transformers"],
            )
    Example passing in a pipeline path:
        .. code-block:: python

            from langchain.embeddings import SelfHostedHFEmbeddings
            import runhouse as rh
            from transformers import pipeline

            gpu = rh.cluster(name="rh-a10x", instance_type="A100:1")
            pipeline = pipeline(model="bert-base-uncased", task="feature-extraction")
            rh.blob(pickle.dumps(pipeline), path="models/pipeline.pkl").save().to(gpu, path="models")
            llm = SelfHostedHFEmbeddings.from_pipeline(
                pipeline="models/pipeline.pkl",
                hardware=gpu,
                model_reqs=["./", "torch", "transformers"],
            )
    """

    inference_fn: Callable = _embed_documents  #: :meta private:
    """Inference function to send to the remote hardware."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.client(self.pipeline_ref, texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.client(self.pipeline_ref, text)
        return embedding.tolist()

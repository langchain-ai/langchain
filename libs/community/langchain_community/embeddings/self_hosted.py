from typing import Any, List

import runhouse as rh
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Extra

from langchain_community.llms.self_hosted import SelfHostedPipeline


class EmbeddingsClass:
    def __init__(self, model_id: str, instruct: bool = False, device: int = 0):
        super().__init__()
        self.model_id, self.instruct, self.device = model_id, instruct, device
        self.client = None

    def load_embedding_model(self) -> Any:
        import sentence_transformers

        self.client = sentence_transformers.SentenceTransformer(self.model_id)

    def embed_documents(self, *args: Any, **kwargs: Any) -> List[List[float]]:
        """Inference function to send to the remote hardware.

        Accepts a sentence_transformer model_id and
        returns a list of embeddings for each document in the batch.
        """
        return self.client(*args, **kwargs)


class SelfHostedEmbeddings(SelfHostedPipeline, Embeddings):
    """Custom embedding models on self-hosted remote hardware.

    Supported hardware includes auto-launched instances on AWS, GCP, Azure,
    and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another
    cloud like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example using a model load function:
        .. code-block:: python

            from langchain_community.embeddings import SelfHostedEmbeddings
            import runhouse as rh

            class MyEmbeddingsClass:
                def __init__(self,
                            model_id: str,
                            instruct: bool = False,
                            device: int = 0):
                    ... construction implementation here ...

                def load_embedding_model(self):
                    ... load_embedding_model implementation here ...

                def embed_documents(self, *args: Any, **kwargs: Any) ->
                                                                    List[List[float]]:
                    .. embed_documents implementation here ...

            gpu = rh.cluster(name='rh-a10x', instance_type='g5.4xlarge', provider='aws')
            gpu.run(commands=["pip install langchain"])
            embedding_env = rh.env(name="embeddings_env",
                                   reqs=["transformers",
                                        "torch",
                                        "accelerate",
                                        "huggingface-hub",
                                        "sentence_transformers"],
                                   secrets=["huggingface"]
                                   # need for downloading models from huggingface
                                  ).to(system=gpu)
            hf = SelfHostedEmbeddings(embeddings_cls=MyEmbeddingsClass,
                                      hardware=gpu,
                                      env=embedding_env)
            # if embeddings_cls is not provided,
                    the default embeddings_cls will be used.
    """

    inference_kwargs: Any = None
    """Any kwargs to pass to the model's inference function."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow
        arbitrary_types_allowed = True

    def __init__(
        self,
        embeddings_cls: Any = EmbeddingsClass,
        load_fn_kwargs: Any = None,
        **kwargs: Any,
    ):
        """Init the pipeline with an auxiliary function.

        The load function must be in global scope to be imported
        and run on the server, i.e. in a module and not a REPL or closure.
        Then, initialize the remote inference function.
        """
        gpu, embeddings_env = kwargs.get("hardware"), kwargs.get("env")
        model_id, task = (
            load_fn_kwargs.get("model_id", None),
            load_fn_kwargs.get("task", None),
        )
        super().__init__(hardware=gpu, env=embeddings_env, model_id=model_id, task=task)
        EmbeddingsPipeline_remote = rh.module(embeddings_cls).to(
            system=gpu, env=embeddings_env
        )
        self.EmbeddingsPipeline_remote_instance = EmbeddingsPipeline_remote(
            model_id=model_id
        )
        _load_fn_kwargs = self.load_fn_kwargs or {}
        self.EmbeddingsPipeline_remote_instance.load_embedding_model.remote(
            **_load_fn_kwargs
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.s

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.EmbeddingsPipeline_remote_instance.embed_documents(texts)
        if not isinstance(embeddings, list):
            return embeddings.tolist()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embeddings = self.EmbeddingsPipeline_remote_instance.embed_documents(text)
        if not isinstance(embeddings, list):
            return embeddings.tolist()
        return embeddings

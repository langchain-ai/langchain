import importlib.util
import os
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict


class QuantizedBgeEmbeddings(BaseModel, Embeddings):
    """Leverage Itrex runtime to unlock the performance of compressed NLP models.

    Please ensure that you have installed intel-extension-for-transformers.

    Input:
        model_name: str = Model name.
        max_seq_len: int = The maximum sequence length for tokenization. (default 512)
        pooling_strategy: str =
            "mean" or "cls", pooling strategy for the final layer. (default "mean")
        query_instruction: Optional[str] =
            An instruction to add to the query before embedding. (default None)
        document_instruction: Optional[str] =
            An instruction to add to each document before embedding. (default None)
        padding: Optional[bool] =
            Whether to add padding during tokenization or not. (default True)
        model_kwargs: Optional[Dict] =
            Parameters to add to the model during initialization. (default {})
        encode_kwargs: Optional[Dict] =
            Parameters to add during the embedding forward pass. (default {})
        onnx_file_name: Optional[str] =
            File name of onnx optimized model which is exported by itrex.
            (default "int8-model.onnx")

    Example:
        .. code-block:: python

            from langchain_community.embeddings import QuantizedBgeEmbeddings

            model_name = "Intel/bge-small-en-v1.5-sts-int8-static-inc"
            encode_kwargs = {'normalize_embeddings': True}
            hf = QuantizedBgeEmbeddings(
                model_name,
                encode_kwargs=encode_kwargs,
                query_instruction="Represent this sentence for searching relevant passages: "
            )
    """  # noqa: E501

    def __init__(
        self,
        model_name: str,
        *,
        max_seq_len: int = 512,
        pooling_strategy: str = "mean",  # "mean" or "cls"
        query_instruction: Optional[str] = None,
        document_instruction: Optional[str] = None,
        padding: bool = True,
        model_kwargs: Optional[Dict] = None,
        encode_kwargs: Optional[Dict] = None,
        onnx_file_name: Optional[str] = "int8-model.onnx",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # check sentence_transformers python package
        if importlib.util.find_spec("intel_extension_for_transformers") is None:
            raise ImportError(
                "Could not import intel_extension_for_transformers python package. "
                "Please install it with "
                "`pip install -U intel-extension-for-transformers`."
            )

        # check torch python package
        if importlib.util.find_spec("torch") is None:
            raise ImportError(
                "Could not import torch python package. "
                "Please install it with `pip install -U torch`."
            )

        # check onnx python package
        if importlib.util.find_spec("onnx") is None:
            raise ImportError(
                "Could not import onnx python package. "
                "Please install it with `pip install -U onnx`."
            )

        self.model_name_or_path = model_name
        self.max_seq_len = max_seq_len
        self.pooling = pooling_strategy
        self.padding = padding
        self.encode_kwargs = encode_kwargs or {}
        self.model_kwargs = model_kwargs or {}

        self.normalize = self.encode_kwargs.get("normalize_embeddings", False)
        self.batch_size = self.encode_kwargs.get("batch_size", 32)

        self.query_instruction = query_instruction
        self.document_instruction = document_instruction
        self.onnx_file_name = onnx_file_name

        self.load_model()

    def load_model(self) -> None:
        from huggingface_hub import hf_hub_download
        from intel_extension_for_transformers.transformers import AutoModel
        from transformers import AutoConfig, AutoTokenizer

        self.hidden_size = AutoConfig.from_pretrained(
            self.model_name_or_path
        ).hidden_size
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
        )
        onnx_model_path = os.path.join(self.model_name_or_path, self.onnx_file_name)  # type: ignore[arg-type]
        if not os.path.exists(onnx_model_path):
            onnx_model_path = hf_hub_download(
                self.model_name_or_path, filename=self.onnx_file_name
            )
        self.transformer_model = AutoModel.from_pretrained(
            onnx_model_path, use_embedding_runtime=True
        )

    model_config = ConfigDict(
        extra="allow",
        protected_namespaces=(),
    )

    def _embed(self, inputs: Any) -> Any:
        import torch

        engine_input = [value for value in inputs.values()]
        outputs = self.transformer_model.generate(engine_input)
        if "last_hidden_state:0" in outputs:
            last_hidden_state = outputs["last_hidden_state:0"]
        else:
            last_hidden_state = [out for out in outputs.values()][0]
        last_hidden_state = torch.tensor(last_hidden_state).reshape(
            inputs["input_ids"].shape[0], inputs["input_ids"].shape[1], self.hidden_size
        )
        if self.pooling == "mean":
            emb = self._mean_pooling(last_hidden_state, inputs["attention_mask"])
        elif self.pooling == "cls":
            emb = self._cls_pooling(last_hidden_state)
        else:
            raise ValueError("pooling method no supported")

        if self.normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

    @staticmethod
    def _cls_pooling(last_hidden_state: Any) -> Any:
        return last_hidden_state[:, 0]

    @staticmethod
    def _mean_pooling(last_hidden_state: Any, attention_mask: Any) -> Any:
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "Unable to import torch, please install with `pip install -U torch`."
            ) from e
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        inputs = self.transformer_tokenizer(
            texts,
            max_length=self.max_seq_len,
            truncation=True,
            padding=self.padding,
            return_tensors="pt",
        )
        return self._embed(inputs).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text documents using the Optimized Embedder model.

        Input:
            texts: List[str] = List of text documents to embed.
        Output:
            List[List[float]] = The embeddings of each text document.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Unable to import pandas, please install with `pip install -U pandas`."
            ) from e
        docs = [
            self.document_instruction + d if self.document_instruction else d
            for d in texts
        ]

        # group into batches
        text_list_df = pd.DataFrame(docs, columns=["texts"]).reset_index()

        # assign each example with its batch
        text_list_df["batch_index"] = text_list_df["index"] // self.batch_size

        # create groups
        batches = list(text_list_df.groupby(["batch_index"])["texts"].apply(list))

        vectors = []
        for batch in batches:
            vectors += self._embed_text(batch)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        if self.query_instruction:
            text = self.query_instruction + text
        return self._embed_text([text])[0]

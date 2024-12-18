from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict


class QuantizedBiEncoderEmbeddings(BaseModel, Embeddings):
    """Quantized bi-encoders embedding models.

    Please ensure that you have installed optimum-intel and ipex.

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

    Example:

    from langchain_community.embeddings import QuantizedBiEncoderEmbeddings

    model_name = "Intel/bge-small-en-v1.5-rag-int8-static"
    encode_kwargs = {'normalize_embeddings': True}
    hf = QuantizedBiEncoderEmbeddings(
        model_name,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this sentence for searching relevant passages: "
    )
    """

    def __init__(
        self,
        model_name: str,
        max_seq_len: int = 512,
        pooling_strategy: str = "mean",  # "mean" or "cls"
        query_instruction: Optional[str] = None,
        document_instruction: Optional[str] = None,
        padding: bool = True,
        model_kwargs: Optional[Dict] = None,
        encode_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
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

        self.load_model()

    def load_model(self) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Unable to import transformers, please install with "
                "`pip install -U transformers`."
            ) from e
        try:
            from optimum.intel import IPEXModel

            self.transformer_model = IPEXModel.from_pretrained(
                self.model_name_or_path, **self.model_kwargs
            )
        except Exception as e:
            raise Exception(
                f"""
Failed to load model {self.model_name_or_path}, due to the following error:
{e}
Please ensure that you have installed optimum-intel and ipex correctly,using:

pip install optimum[neural-compressor]
pip install intel_extension_for_pytorch

For more information, please visit:
* Install optimum-intel as shown here: https://github.com/huggingface/optimum-intel.
* Install IPEX as shown here: https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu&version=v2.2.0%2Bcpu.
"""
            )
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
        )
        self.transformer_model.eval()

    model_config = ConfigDict(
        extra="allow",
        protected_namespaces=(),
    )

    def _embed(self, inputs: Any) -> Any:
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "Unable to import torch, please install with `pip install -U torch`."
            ) from e
        with torch.inference_mode():
            outputs = self.transformer_model(**inputs)
            if self.pooling == "mean":
                emb = self._mean_pooling(outputs, inputs["attention_mask"])
            elif self.pooling == "cls":
                emb = self._cls_pooling(outputs)
            else:
                raise ValueError("pooling method no supported")

            if self.normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb

    @staticmethod
    def _cls_pooling(outputs: Any) -> Any:
        if isinstance(outputs, dict):
            token_embeddings = outputs["last_hidden_state"]
        else:
            token_embeddings = outputs[0]
        return token_embeddings[:, 0]

    @staticmethod
    def _mean_pooling(outputs: Any, attention_mask: Any) -> Any:
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "Unable to import torch, please install with `pip install -U torch`."
            ) from e
        if isinstance(outputs, dict):
            token_embeddings = outputs["last_hidden_state"]
        else:
            # First element of model_output contains all token embeddings
            token_embeddings = outputs[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
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
        try:
            from tqdm import tqdm
        except ImportError as e:
            raise ImportError(
                "Unable to import tqdm, please install with `pip install -U tqdm`."
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
        for batch in tqdm(batches, desc="Batches"):
            vectors += self._embed_text(batch)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        if self.query_instruction:
            text = self.query_instruction + text
        return self._embed_text([text])[0]

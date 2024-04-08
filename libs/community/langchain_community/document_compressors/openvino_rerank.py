from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.pydantic_v1 import Field


class RerankRequest:
    def __init__(self, query: Any = None, passages: Any = None):
        self.query = query
        self.passages = passages if passages is not None else []


class OpenVINOReranker(BaseDocumentCompressor):
    """
    OpenVINO rerank models.
    """

    ov_model: Any
    """OpenVINO model object."""
    tokenizer: Any
    """Tokenizer for embedding model."""
    model_name_or_path: str
    """HuggingFace model id."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments passed to the model."""
    top_n: int = 4
    """return Top n texts."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        try:
            from optimum.intel.openvino import OVModelForSequenceClassification
        except ImportError as e:
            raise ValueError(
                "Could not import optimum-intel python package. "
                "Please install it with: "
                "pip install -U 'optimum[openvino,nncf]'"
            ) from e

        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise ValueError(
                "Could not import huggingface_hub python package. "
                "Please install it with: "
                "`pip install -U huggingface_hub`."
            ) from e

        def require_model_export(
            model_id: str, revision: Any = None, subfolder: Any = None
        ) -> bool:
            model_dir = Path(model_id)
            if subfolder is not None:
                model_dir = model_dir / subfolder
            if model_dir.is_dir():
                return (
                    not (model_dir / "openvino_model.xml").exists()
                    or not (model_dir / "openvino_model.bin").exists()
                )
            hf_api = HfApi()
            try:
                model_info = hf_api.model_info(model_id, revision=revision or "main")
                normalized_subfolder = (
                    None if subfolder is None else Path(subfolder).as_posix()
                )
                model_files = [
                    file.rfilename
                    for file in model_info.siblings
                    if normalized_subfolder is None
                    or file.rfilename.startswith(normalized_subfolder)
                ]
                ov_model_path = (
                    "openvino_model.xml"
                    if subfolder is None
                    else f"{normalized_subfolder}/openvino_model.xml"
                )
                return (
                    ov_model_path not in model_files
                    or ov_model_path.replace(".xml", ".bin") not in model_files
                )
            except Exception:
                return True

        if require_model_export(self.model_name_or_path):
            # use remote model
            self.ov_model = OVModelForSequenceClassification.from_pretrained(
                self.model_name_or_path, export=True, **self.model_kwargs
            )
        else:
            # use local model
            self.ov_model = OVModelForSequenceClassification.from_pretrained(
                self.model_name_or_path, **self.model_kwargs
            )

        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Unable to import transformers, please install with "
                "`pip install -U transformers`."
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def rerank(self, request: Any) -> Any:
        query = request.query
        passages = request.passages

        query_passage_pairs = [[query, passage["text"]] for passage in passages]
        input_tensors = self.tokenizer(
            query_passage_pairs, padding=True, truncation=True, return_tensors="pt"
        )

        outputs = self.ov_model(**input_tensors, return_dict=True)
        if outputs[0].shape[1] > 1:
            scores = outputs[0][:, 1]
        else:
            scores = outputs[0].flatten()

        scores = list(1 / (1 + np.exp(-scores)))

        # Combine scores with passages, including metadata
        for score, passage in zip(scores, passages):
            passage["score"] = score

        # Sort passages based on scores
        passages.sort(key=lambda x: x["score"], reverse=True)

        return passages

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        passages = [
            {"id": i, "text": doc.page_content} for i, doc in enumerate(documents)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        rerank_response = self.rerank(rerank_request)[: self.top_n]
        final_results = []
        for r in rerank_response:
            doc = Document(
                page_content=r["text"],
                metadata={"id": r["id"], "relevance_score": r["score"]},
            )
            final_results.append(doc)
        return final_results

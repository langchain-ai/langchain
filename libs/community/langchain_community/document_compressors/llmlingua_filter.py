# LLM Lingua Document Compressor

import re
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import (
    BaseDocumentCompressor,
)
from langchain_core.pydantic_v1 import root_validator

DEFAULT_LLM_LINGUA_INSTRUCTION = (
    "Given this documents, please answer the final question"
)


class LLMLinguaCompressor(BaseDocumentCompressor):
    """
    Compress using LLMLingua Project.

    https://github.com/microsoft/LLMLingua
    """

    # Pattern to match ref tags at the beginning or end of the string,
    # allowing for malformed tags
    _pattern_beginning: Pattern = re.compile(r"\A(?:<#)?(?:ref)?(\d+)(?:#>?)?")
    _pattern_ending: Pattern = re.compile(r"(?:<#)?(?:ref)?(\d+)(?:#>?)?\Z")

    model_name: str = "NousResearch/Llama-2-7b-hf"
    """The hugging face model to use"""
    device_map: str = "cuda"
    """The device to use for llm lingua"""
    target_token: int = 300
    """The target number of compressed tokens"""
    rank_method: str = "longllmlingua"
    """The ranking method to use"""
    model_config: dict = {}
    """Custom configuration for the model"""
    open_api_config: dict = {}
    """open_api configuration"""
    instruction: str = DEFAULT_LLM_LINGUA_INSTRUCTION
    """The instruction for the LLM"""
    additional_compress_kwargs: dict = {
        "condition_compare": True,
        "condition_in_question": "after",
        "context_budget": "+100",
        "reorder_context": "sort",
        "dynamic_context_compression_ratio": 0.4,
    }
    """Extra compression arguments"""
    lingua: Any
    """The instance of the llm linqua"""

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            from llmlingua import PromptCompressor
        except ImportError:
            raise ImportError(
                "Could not import llmlingua python package. "
                "Please install it with `pip install llmlingua`."
            )
        if not values.get("lingua"):
            values["lingua"] = PromptCompressor(
                model_name=values.get("model_name", {}),
                device_map=values.get("device_map", {}),
                model_config=values.get("model_config", {}),
                open_api_config=values.get("open_api_config", {}),
            )
        return values

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    @staticmethod
    def _format_context(docs: Sequence[Document]) -> List[str]:
        """
        Format the output of the retriever by including
        special ref tags for tracking the metadata after compression
        """
        formatted_docs = []
        for i, doc in enumerate(docs):
            content = doc.page_content.replace("\n\n", "\n")
            doc_string = f"\n\n<#ref{i}#> {content} <#ref{i}#>\n\n"
            formatted_docs.append(doc_string)
        return formatted_docs

    def extract_ref_id_tuples_and_clean(
        self, contents: List[str]
    ) -> List[Tuple[str, int]]:
        """
        Extracts reference IDs from the contents and cleans up the ref tags.

        This function processes a list of strings, searching for reference ID tags
        at the beginning and end of each string. When a ref tag is found, it is
        removed from the string, and its ID is recorded. If no ref ID is found,
        a generic ID of "-1" is assigned.

        The search for ref tags is performed only at the beginning and
        end of the string, with the assumption that there will
        be at most one ref ID per string. Malformed ref tags are
        handled gracefully.

        Args:
            contents (List[str]): A list of contents to be processed.

        Returns:
            List[Tuple[str, int]]: The cleaned string and the associated ref ID.

        Examples:
            >>> strings_list = [
                    '<#ref0#> Example content <#ref0#>',
                    'Content with no ref ID.'
                ]
            >>> extract_ref_id_tuples_and_clean(strings_list)
            [('Example content', 0), ('Content with no ref ID.', -1)]
        """
        ref_id_tuples = []
        for content in contents:
            clean_string = content.strip()
            if not clean_string:
                continue

            # Search for ref tags at the beginning and the end of the string
            ref_id = None
            for pattern in [self._pattern_beginning, self._pattern_ending]:
                match = pattern.search(clean_string)
                if match:
                    ref_id = match.group(1)
                    clean_string = pattern.sub("", clean_string).strip()
            # Convert ref ID to int or use -1 if not found
            ref_id_to_use = int(ref_id) if ref_id and ref_id.isdigit() else -1
            ref_id_tuples.append((clean_string, ref_id_to_use))

        return ref_id_tuples

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using BAAI/bge-reranker models.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []

        compressed_prompt = self.lingua.compress_prompt(
            context=self._format_context(documents),
            instruction=self.instruction,
            question=query,
            target_token=self.target_token,
            rank_method=self.rank_method,
            concate_question=False,
            add_instruction=True,
            **self.additional_compress_kwargs,
        )
        compreseed_context = compressed_prompt["compressed_prompt"].split("\n\n")[1:]

        extracted_metadata = self.extract_ref_id_tuples_and_clean(compreseed_context)

        compressed_docs: List[Document] = []

        for context, index in extracted_metadata:
            if index == -1 or index >= len(documents):
                doc = Document(page_content=context)
            else:
                doc = Document(page_content=context, metadata=documents[index].metadata)
            compressed_docs.append(doc)

        return compressed_docs

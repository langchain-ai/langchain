"""**text_splitter** is an integration of text splitter other then .

Large language models (LLMs) can be used for many tasks,
but often have a limited context size that can be smaller than
documents you might want to use. To use documents of larger length,
you often have to split your text into chunks to fit within this context size,
This crate provides methods for splitting longer pieces of text
into smaller chunks, aiming to maximize a desired chunk size,
but still splitting at semantically sensible boundaries whenever possible


**Class hierarchy:**

.. code-block::
     <name>TextSplitter  # Examples: SemanticCharacterTextSplitter, SemanticTiktokenTextSplitter

"""

from langchain_community.text_splitter.semantic_text_splitter import (
    SemanticCharacterTextSplitter,
    SemanticTiktokenTextSplitter,
)

__all__ = [
    "SemanticCharacterTextSplitter",
    "SemanticTiktokenTextSplitter",
]

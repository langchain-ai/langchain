"""**Text Splitters** are classes for splitting text.


**Class hierarchy:**

.. code-block::

    BaseTextSplitter, BaseDocumentTransformer --> TextSplitter --> <name>TextSplitter  # Example: CharacterTextSplitter
                                                 RecursiveCharacterTextSplitter -->  <name>TextSplitter

Note: **MarkdownHeaderTextSplitter** and **HTMLHeaderTextSplitter derive from BaseTextSplitter only.


**Main helpers:**

.. code-block::

    Document, Tokenizer, Language, LineType, HeaderType

"""  # noqa: E501

from langchain.text_splitters import (
    TS,
    CharacterTextSplitter,
    ElementType,
    HeaderType,
    HTMLHeaderTextSplitter,
    Language,
    LatexTextSplitter,
    LineType,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    NLTKTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    SentenceTransformersTokenTextSplitter,
    SpacyTextSplitter,
    TextSplitter,
    Tokenizer,
    TokenTextSplitter,
    split_text_on_tokens,
)

__all__ = [
    "CharacterTextSplitter",
    "ElementType",
    "HTMLHeaderTextSplitter",
    "HeaderType",
    "Language",
    "LatexTextSplitter",
    "LineType",
    "MarkdownHeaderTextSplitter",
    "MarkdownTextSplitter",
    "NLTKTextSplitter",
    "PythonCodeTextSplitter",
    "RecursiveCharacterTextSplitter",
    "RecursiveJsonSplitter",
    "SentenceTransformersTokenTextSplitter",
    "SpacyTextSplitter",
    "TS",
    "TextSplitter",
    "TokenTextSplitter",
    "Tokenizer",
    "split_text_on_tokens",
]

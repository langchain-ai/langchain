"""**Text Splitters** are classes for splitting text.


**Class hierarchy:**

.. code-block::

    BaseTextSplitter, BaseDocumentTransformer --> TextSplitter --> <name>TextSplitter  # Example: CharacterTextSplitter
                                                 RecursiveCharacterTextSplitter -->  <name>TextSplitter
    BaseTextToDocumentsSplitter --> HTMLHeaderTextSplitter, MarkdownHeaderTextSplitter


**Main helpers:**

.. code-block::

    Document, Tokenizer, Language, LineType, HeaderType

"""  # noqa: E501

from langchain.text_splitters.base import TS, TextSplitter
from langchain.text_splitters.character import CharacterTextSplitter
from langchain.text_splitters.html_header import ElementType, HTMLHeaderTextSplitter
from langchain.text_splitters.markdown import (
    HeaderType,
    LineType,
    MarkdownHeaderTextSplitter,
)
from langchain.text_splitters.nltk import NLTKTextSplitter
from langchain.text_splitters.recursive_character import (
    LatexTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.text_splitters.spacy import SpacyTextSplitter
from langchain.text_splitters.token import (
    SentenceTransformersTokenTextSplitter,
    Tokenizer,
    TokenTextSplitter,
    split_text_on_tokens,
)
from langchain.text_splitters.utils import Language

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
    "SentenceTransformersTokenTextSplitter",
    "SpacyTextSplitter",
    "TS",
    "TextSplitter",
    "TokenTextSplitter",
    "Tokenizer",
    "split_text_on_tokens",
]

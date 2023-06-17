from abc import ABC, abstractmethod
from typing import List

from langchain.chains.combine_documents.base import format_document
from langchain.docstore.document import Document
from langchain.load.serializable import Serializable
from langchain.prompts.base import BasePromptTemplate


class ContextBuilder(Serializable):
    """A vectorstore can be used to provide an LLM with extra context by including it in the prompt.

    by default context is created by joining all the documents returned from the store seperated by a seperator
    e.g
    <page_content>
    <document_separator>
    <page_content>
    <document_separator>
    <page_content>


    However often the logic we want to apply is much more bespoke.
    For exmaple imagine we are storing information about ppl in a vector store.

    The first entry would look like:
    name,date of birth,job
    Paul,23-12-1985   ,cleaner

    another entry would look like:
    name,date of birth,job
    John,23-12-1985   ,driver


    Obviously both of these embeddings contain the same headers.

    By default the context would be created as such

    name,date of birth,job
    Paul,23-12-1985   ,cleaner
    <document_separator>
    name,date of birth,job
    John,23-12-1985   ,driver


    However we want to reduce the header duplication and also just create it as one table for the LLM to understand

    name,date of birth,job
    Paul,23-12-1985   ,cleaner
    John,23-12-1985   ,driver

    """

    @abstractmethod
    def create_context(
        self, docs: List[Document], document_prompt: BasePromptTemplate
    ) -> str:
        """create a string context from the documents"""


class DefaultContextBuilder(ContextBuilder):
    document_separator: str = "\n\n"
    """The string with which to join the formatted documents"""

    def create_context(self, docs, document_prompt: BasePromptTemplate) -> str:
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return self.document_separator.join(doc_strings)

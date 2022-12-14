"""Combining documents by mapping a chain over them first, then combining results."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from transformers import GPT2TokenizerFast


class MapReduceDocumentsChain(BaseCombineDocumentsChain, BaseModel):
    """Combining documents by mapping a chain over them, then combining results."""

    llm_chain: LLMChain
    """Chain to apply to each document individually.."""
    combine_document_chain: BaseCombineDocumentsChain
    """Chain to use to combine results of applying llm_chain to documents."""
    document_variable_name: str
    """The variable name in the llm_chain to put the documents in.
    If only one variable in the llm_chain, this need not be provided."""
    question_prompt: PromptTemplate
    combine_prompt: PromptTemplate
    document_prompt: PromptTemplate

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def get_default_document_variable_name(cls, values: Dict) -> Dict:
        """Get default document variable name, if not provided."""
        if "document_variable_name" not in values:
            llm_chain_variables = values["llm_chain"].prompt.input_variables
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                raise ValueError(
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain input_variables"
                )
        else:
            llm_chain_variables = values["llm_chain"].prompt.input_variables
            if values["document_variable_name"] not in llm_chain_variables:
                raise ValueError(
                    f"document_variable_name {values['document_variable_name']} was "
                    f"not found in llm_chain input_variables: {llm_chain_variables}"
                )
        return values

    def compute_tokens(self, results: List, docs: List[Document], **kwargs: Any) -> int:
        """Checks to see if, after combining a set of documents into one text, if it is too long."""
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            # This uses metadata from the docs, and the textual results from `results`
            for i, r in enumerate(results)
        ]
        doc_dicts = []
        for doc in result_docs:
            base_info = {"page_content": doc.page_content}
            base_info.update(doc.metadata)
            document_info = {
                k: base_info[k] for k in self.document_prompt.input_variables
                # TODO: how does this work with plain QA, as opposed to this which is designed to work with
                # QA with sources right now?
            }
            doc_dicts.append(document_info)
        # Format each document according to the prompt
        doc_strings = [self.document_prompt.format(**doc) for doc in doc_dicts]
        combined_texts = "\n\n".join(doc_strings)

        # tokenize the text and compute num tokens
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenized_text = tokenizer.tokenize(combined_texts)
        num_tokens = len(tokenized_text)
        return num_tokens

    def combine_docs(self, docs: List[Document],
                     token_max=3000,
                     **kwargs: Any) -> str:
        """
        Combine by mapping first chain over all, then
        if NUM OF TOKENS OF combined_texts is too large for context window,
        combine some text and run self.llm_chain.apply again,
        keep doing this in a while loop until the num of tokens in the combined text fits, then
        stuff it into a final chain once the NUM OF TOKENS OF combined_texts is within the limit.
        """

        results = self.llm_chain.apply(
            # FYI - this is parallelized and so it is fast.
            [{**{self.document_variable_name: d.page_content}, **kwargs} for d in docs]
        )
        question_result_key = self.llm_chain.output_key

        num_tokens = self.compute_tokens(results, docs, **kwargs)
        print(num_tokens)

        # and if NUM OF TOKENS OF combined_texts is too large for context window,
        # then combine some texts and keep running self.llm_chain.apply again until it fits.
        while num_tokens > token_max:
            # docs is a List[Document] where each Document has .page_content w/ the text
            alld = []
            for i in range(len(results)):
                if (i % 2) != 0:
                    alld.append(Document(page_content="\n\n".join([results[i][question_result_key],
                                                                   results[i - 1][question_result_key]]),
                                         # TODO: combine metadata from [i] and [i-1]?
                                         metadata=docs[i].metadata))

            results = self.llm_chain.apply(
                [{**{self.document_variable_name: d.page_content}, **kwargs} for d in alld]
            )
            num_tokens = self.compute_tokens(results, docs, **kwargs)
            print(num_tokens)

        print("Starting final reduce_chain / merging into final answer")
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            # This uses metadata from the docs, and the textual results from `results`
            for i, r in enumerate(results)
        ]
        print("Combining docs.")
        out = self.combine_document_chain.combine_docs(result_docs, **kwargs)
        print("Done combining docs.")
        return out

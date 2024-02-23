"""Chain that combines documents by stuffing into context."""

from typing import Any, Dict, List, Optional, Tuple

from langchain.callbacks.manager import Callbacks
from langchain.chains.combine_documents.base import (
    BaseCombineDocumentsChain,
)
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import Extra, Field, root_validator
from langchain.schema import BasePromptTemplate, format_document
from collections import Counter
import re

# Define the Unicode space regex pattern
unicode_space_pattern = re.compile(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000]+')

# Example Unicode string with different space characters
unicode_string = "This\u00A0is\u0020a\u2003test\u3000string"



def _get_default_document_prompt() -> PromptTemplate:
    return PromptTemplate(input_variables=["page_content"], template="{page_content}")

def find_odd_one_out(lst):
    counter = Counter(lst)
    return counter.most_common()

def _extract_page_number(s: Document) -> int:
    # Use regex to find the page number in the string
    import re
    match = re.search(r'page (\d+) of \d+', s.page_content.lower())
    if match:
        # If a page number is found, return it as an integer
        return int(match.group(1))
    else:
        # If no page number is found, return a large number to sort this item last
        return float('inf')

def replace_sep_text(doc:Document, sep=';', oksep='|') -> Document:
    doc.page_content = doc.page_content.replace(sep, oksep)
    # Replace all Unicode space characters with ASCII space using compiled regex pattern
    ascii_string = unicode_space_pattern.sub(' ', doc.page_content)
    doc.page_content = ascii_string
    return doc


def _extract_page_number_meta(s: Document) -> int:
    # Use regex to find the page number in the string
    page = s.metadata.get('page', None)
    if isinstance(page, int) or (isinstance(page, str) and page.isdigit()):
        # If a page number is found, return it as an integer
        return int(page)+1
    else:
        # If no page number is found, return a large number to sort this item last
        return float('inf')

class StuffDocumentsChain(BaseCombineDocumentsChain):
    """Chain that combines documents by stuffing into context.

    This chain takes a list of documents and first combines them into a single string.
    It does this by formatting each document into a string with the `document_prompt`
    and then joining them together with `document_separator`. It then adds that new
    string to the inputs with the variable name set by `document_variable_name`.
    Those inputs are then passed to the `llm_chain`.

    Example:
        .. code-block:: python

            from langchain.chains import StuffDocumentsChain, LLMChain
            from langchain.prompts import PromptTemplate
            from langchain.llms import OpenAI

            # This controls how each document will be formatted. Specifically,
            # it will be passed to `format_document` - see that function for more
            # details.
            document_prompt = PromptTemplate(
                input_variables=["page_content"],
                 template="{page_content}"
            )
            document_variable_name = "context"
            llm = OpenAI()
            # The prompt here should take as an input variable the
            # `document_variable_name`
            prompt = PromptTemplate.from_template(
                "Summarize this content: {context}"
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_prompt=document_prompt,
                document_variable_name=document_variable_name
            )
    """

    llm_chain: LLMChain
    """LLM chain which is called with the formatted document string,
    along with any other inputs."""
    document_prompt: BasePromptTemplate = Field(
        default_factory=_get_default_document_prompt
    )
    """Prompt to use to format each document, gets passed to `format_document`."""
    document_variable_name: str
    """The variable name in the llm_chain to put the documents in.
    If only one variable in the llm_chain, this need not be provided."""
    document_separator: str = "\n\n"
    """The string with which to join the formatted documents"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def get_default_document_variable_name(cls, values: Dict) -> Dict:
        """Get default document variable name, if not provided.

        If only one variable is present in the llm_chain.prompt,
        we can infer that the formatted documents should be passed in
        with this variable name.
        """
        llm_chain_variables = values["llm_chain"].prompt.input_variables
        if "document_variable_name" not in values:
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                raise ValueError(
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain_variables"
                )
        else:
            if values["document_variable_name"] not in llm_chain_variables:
                raise ValueError(
                    f"document_variable_name {values['document_variable_name']} was "
                    f"not found in llm_chain input_variables: {llm_chain_variables}"
                )
        return values

    @property
    def input_keys(self) -> List[str]:
        extra_keys = [
            k for k in self.llm_chain.input_keys if k != self.document_variable_name
        ]
        return super().input_keys + extra_keys

    def _get_inputs(self, docs: List[Document], **kwargs: Any) -> dict:
        """Construct inputs from kwargs and docs.

        Format and the join all the documents together into one input with name
        `self.document_variable_name`. The pluck any additional variables
        from **kwargs.

        Args:
            docs: List of documents to format and then join into single input
            **kwargs: additional inputs to chain, will pluck any other required
                arguments from here.

        Returns:
            dictionary of inputs to LLMChain
        """
        # Format each document according to the prompt
        doc_strings = [format_document(doc, self.document_prompt) for doc in docs]
        # Join the documents together to put them in the prompt.
        inputs = {
            k: v
            for k, v in kwargs.items()
            if k in self.llm_chain.prompt.input_variables
        }
        inputs[self.document_variable_name] = self.document_separator.join(doc_strings)
        return inputs

    def prompt_length(self, docs: List[Document], **kwargs: Any) -> Optional[int]:
        """Return the prompt length given the documents passed in.

        This can be used by a caller to determine whether passing in a list
        of documents would exceed a certain prompt length. This useful when
        trying to ensure that the size of a prompt remains below a certain
        context limit.

        Args:
            docs: List[Document], a list of documents to use to calculate the
                total prompt length.

        Returns:
            Returns None if the method does not depend on the prompt length,
            otherwise the length of the prompt in tokens.
        """
        inputs = self._get_inputs(docs, **kwargs)
        prompt = self.llm_chain.prompt.format(**inputs)
        return self.llm_chain.llm.get_num_tokens(prompt)

    def apply_predict_docs_input(self, partial_doc, **kwargs: Any):
        inputs = self._get_inputs(partial_doc, **kwargs)
        # Call predict on the LLM.
        return self.llm_chain.predict(callbacks=callbacks, **inputs)

    def count_semicolons(self, s):
        lines = s.split('\n')

        return (any(set([line.count(';SEN') for line in lines])),
                [line.count(';') for line in lines], lines)

    def combine_docs(
            self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Stuff all documents into one prompt and pass to LLM.

        Args:
            docs: List of documents to join together into one variable
            callbacks: Optional callbacks to pass along
            **kwargs: additional parameters to use to get inputs to LLMChain.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        if not docs:
            return '', {}
        sizedocs = len(docs)
        predict2 = ''
        print('sorting by page no')
        if callbacks:
            predict_function = callbacks.metadata.get('callback_oneshot', None)

            if callbacks.metadata.get('sortByPage', True):
                docs = sorted(docs, key=_extract_page_number_meta)
                list(map(replace_sep_text, docs))
            # replace_sep_text(docs)
        split_data_to_pages = 2
        if callbacks:
            split_data_to_pages = callbacks.metadata.get('splitData', 0)
            if split_data_to_pages>len(docs):
                split_data_to_pages = len(docs)
        origdocs = docs
        if split_data_to_pages>1 and sizedocs >= 2:
            splitEnd = int(sizedocs / split_data_to_pages)
            splitStart = 0
            total_semi = []
            while splitStart<sizedocs:
                docs2 = docs[splitStart:splitStart +splitEnd]
                splitStart+=splitEnd
                if not docs2:
                    break
                inputs2 = self._get_inputs(docs2, **kwargs)
                predict_temp = self.llm_chain.predict(callbacks=callbacks, **inputs2)
                hasSENField, c_semi, slines = self.count_semicolons(predict_temp)

                if not any([c>10 for c in c_semi]) and len(set(c_semi))==1:
                    total_semi += c_semi
                    predict2+='\n'+predict_temp

            if len(set(total_semi)) > 1 or hasSENField:
                print('processing in one go')
                predict2 = ''
                docs = origdocs
            else:
                docs = []
        if docs:
            inputs = self._get_inputs(docs, **kwargs)
            # Call predict on the LLM.
            if predict_function:
                predict1 = predict_function(inputs['question']+'\n\n\''+inputs['context']+"\n A:'")
            else:
                predict1 = self.llm_chain.predict(callbacks=callbacks, **inputs)
            hasSENField, c_semi, slines = self.count_semicolons(predict1)
            count_list = find_odd_one_out(c_semi)
            if len(count_list)!=1:
                odd_one = count_list[-1][0]
                idxm = c_semi.index(odd_one)
                alt_idx = idxm +1 if idxm < len(c_semi)-1 else idxm-1
                if odd_one == count_list[0][0] -1:
                    if hasSENField:
                        slines[idxm]+=';ERROR'
                    else:
                        slines[idxm] += ';'
                    # semi_pos = [i for i, char in enumerate(lines[idxm]) if char == ';']
                    # semi_pos_Alt = [i for i, char in enumerate(lines[alt_idx]) if char == ';']
                    predict1 = '\n'.join(slines)

            predict2 = predict2.strip()

            return predict1.strip(), {}
            # + '\n\n|;|' + predict2, {}
        return predict2, {}

    async def acombine_docs(
            self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Async stuff all documents into one prompt and pass to LLM.

        Args:
            docs: List of documents to join together into one variable
            callbacks: Optional callbacks to pass along
            **kwargs: additional parameters to use to get inputs to LLMChain.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        inputs = self._get_inputs(docs, **kwargs)
        # Call predict on the LLM.
        return await self.llm_chain.apredict(callbacks=callbacks, **inputs), {}

    @property
    def _chain_type(self) -> str:
        return "stuff_documents_chain"

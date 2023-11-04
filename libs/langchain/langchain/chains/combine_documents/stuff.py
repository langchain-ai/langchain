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


def _get_default_document_prompt() -> PromptTemplate:
    return PromptTemplate(input_variables=["page_content"], template="{page_content}")

def _extract_page_number(s:Document)->int:
    # Use regex to find the page number in the string
    import re
    match = re.search(r'page (\d+) of \d+', s.page_content.lower())
    if match:
        # If a page number is found, return it as an integer
        return int(match.group(1))
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
        
        return (any( set([line.count(';SEN') for line in lines]) ),
        [line.count(';') for line in lines ])



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
        sizedocs = len(docs)
        predict2 = ''
        print('sorting by page no')
        if callbacks and callbacks.metadata.get('sortByPage', True):
            docs = docs = sorted(docs, key=_extract_page_number)
        if (not callbacks or callbacks.metadata.get('splitData', True)) and sizedocs>=2:
            splitEnd = int(sizedocs/2)
            splitStart = splitEnd
            docs2 =   docs[splitStart:]
            docs = docs[:splitEnd]
            inputs2 = self._get_inputs(docs2, **kwargs)
            predict2 = self.llm_chain.predict(callbacks=callbacks, **inputs2)
            hasSENField, c_semi = self.count_semicolons(predict2)
            if len(set(c_semi))>1 or hasSENField:
                print('processing in one go')
                predict2=''
                docs = docs + docs2
        inputs = self._get_inputs(docs, **kwargs)
        # Call predict on the LLM.
        predict1 = self.llm_chain.predict(callbacks=callbacks, **inputs)
        predict2 = predict2.strip()
        if predict2 and predict2.startswith('Answer'):
            print('halucination answer')
            predict2=''
        return predict1.strip() +'\n\n|;|'+predict2, {}

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

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    Optional,
    Sequence,
    cast,
)

from langchain.chains import LLMChain
from langchain.document_transformers.runnable_document_transformer import (
    RunnableGeneratorDocumentTransformer,
)

# Note: Import directly from langchain_core is not stable and generate some errors
# from langchain_core.documents import Document
# from langchain_core.language_models import BaseLanguageModel
# from langchain_core.output_parsers import BaseOutputParser, NumberedListOutputParser
# from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import NumberedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser, Document
from langchain.schema.language_model import BaseLanguageModel


def _default_get_input(doc: Document) -> Dict[str, Any]:
    """Return the context chain input."""
    return {
        "context": doc.page_content,
    }


_default_template = """
Given a text input, generate {nb_of_questions} questions from it in the same language. 
Context:
```
{context}
```
{format_instructions}"""

_default_parser = NumberedListOutputParser()


def _get_default_chain_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        template=_default_template,
        output_parser=_default_parser,
        partial_variables={
            "format_instructions": _default_parser.get_format_instructions()
        },
    )


class GenerateQuestionsTransformer(RunnableGeneratorDocumentTransformer):
    """Generate questions for each Documents."""

    llm_chain: LLMChain
    get_input: Callable[[Document], dict] = _default_get_input
    nb_of_questions: int = 3

    """Callable for constructing the chain input from the query and a Document."""

    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        _callbacks = kwargs.get("callbacks", None)
        max_retry = 3
        for doc in documents:  # PPR: work with batch?
            _input = {
                **self.get_input(doc),
                **{"nb_of_questions": self.nb_of_questions},
            }
            output = cast(
                Sequence[str], self.llm_chain.predict(callbacks=_callbacks, **_input)
            )
            if not output:
                if max_retry := max_retry - 1:
                    continue
                else:
                    return
            for question in output:
                yield Document(page_content=question, metadata=doc.metadata)

    async def _alazy_transform_documents(  # type: ignore
        self,
        documents: AsyncIterator[Document],
        **kwargs: Any,
    ) -> AsyncIterator[Document]:
        """Compress page content of raw documents asynchronously."""
        _callbacks = kwargs.get("callbacks", None)
        async for doc in documents:  # PPR: work with batch?
            _input = {
                **self.get_input(doc),
                **{"nb_of_questions": self.nb_of_questions},
            }
            output = cast(
                Sequence[str],
                await self.llm_chain.apredict(callbacks=_callbacks, **_input),
            )
            if not output:
                continue
            for question in output:
                yield Document(page_content=question, metadata=doc.metadata)

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        get_input: Optional[Callable[[Document], dict]] = None,
        nb_of_questions: int = 3,
        llm_chain_kwargs: Optional[dict] = None,
    ) -> "GenerateQuestionsTransformer":
        """Initialize from LLM."""
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else _default_get_input
        assert _prompt.output_parser
        llm_chain = LLMChain(
            llm=llm,
            prompt=_prompt,
            output_parser=cast(BaseOutputParser, _prompt.output_parser),
            **(llm_chain_kwargs or {}),
        )
        return cls(
            llm_chain=llm_chain, get_input=_get_input, nb_of_questions=nb_of_questions
        )

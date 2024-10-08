from typing import Any, Callable, Dict, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.schema.document import Document
from langchain.schema.retriever import BaseRetriever

__all__ = ["TokenLimitRetriver"]


class TokenLimitRetrieverMixin(BaseModel):
    token_limit: int
    token_cutoff_strategy: str
    remaining_token_fillup_callback: Optional[Callable[[str, int], str]] = None
    # remaining_token_fillup_callback is a callback that take the final document
    # and number of token remaining
    tokeniser_callback: Callable
    retriever: BaseRetriever

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        assert callable(values["remaining_token_fillup_callback"]) ^ (
            values["token_cutoff_strategy"] == "complete_document"
        ), (
            'cannot set token_cutoff_strategy to be "complete_document" '
            "and set cufoff function at the same time"
            + f"{values['token_cutoff_strategy']} and "
            + f"{values['remaining_token_fillup_callback']}"
        )
        assert values["token_cutoff_strategy"] in [
            "complete_document",
            "partial_document",
        ]
        return values


class TokenLimitRetriver(BaseRetriever, TokenLimitRetrieverMixin):
    """
    Wrap another retriever and limit the token.
    tokenizer callback is require to count the token

    remaining_token_fillup_callback is called when there are token
    quote left for part of the final document. A new document containing
    partial of the final document will be appended to end of
    the retrieved list of documents.


    Args:
        token_limit (int): The maximum number of tokens that must not exceed
        token_cutoff_strategy (Literal['complete_doccument', 'partial_document']):
        specify strategy to cutoff,whether whole document or part of the final document
        retriever (RetrieverLike): retriever to be wrapped
        remaining_token_fillup_callback (Optional[Callable[[str,int], str]]):
        A callable to chops the final document if the final document exceeds token count
        tokeniser_callback (Callable): A callable that calculate the
        number of tokens in a document

    Example:


    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs
    ) -> List[Document]:
        docs = self.retriever._get_relevant_documents(
            query, run_manager=run_manager, **kwargs
        )
        cummulative_token_len = 0
        len_doc_token = 0
        i = 0
        len_doc_token = len(self.tokeniser_callback(docs[i].page_content))
        while cummulative_token_len + len_doc_token <= self.token_limit:
            cummulative_token_len += len_doc_token

            # check next
            i += 1
            if i == len(docs):
                break
            len_doc_token = len(self.tokeniser_callback(docs[i].page_content))

        tokens_remaining = self.token_limit - cummulative_token_len
        to_return = docs[:i]
        if self.token_cutoff_strategy == "partial_document":
            if tokens_remaining > 0 and i < len(docs):
                # chop final document to fit the remaining token
                to_return.append(
                    Document(
                        self.remaining_token_fillup_callback(
                            docs[i].page_content, tokens_remaining
                        )
                    )
                )

        return to_return

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        docs = await self.retriever._aget_relevant_documents(
            query, run_manager=run_manager, **kwargs
        )
        cummulative_token_len = 0
        len_doc_token = 0
        i = 0
        len_doc_token = len(self.tokeniser_callback(docs[i].page_content))
        while cummulative_token_len + len_doc_token <= self.token_limit:
            cummulative_token_len += len_doc_token
            i += 1
            if i == len(docs):
                break
            len_doc_token = len(self.tokeniser_callback(docs[i].page_content))

        tokens_remaining = self.token_limit - cummulative_token_len
        to_return = docs[:i]
        if self.token_cutoff_strategy == "partial_document":
            if tokens_remaining > 0 and i < len(docs):
                # chop final document to fit the remaining token
                to_return.append(
                    Document(
                        self.remaining_token_fillup_callback(
                            docs[i].page_content, tokens_remaining
                        )
                    )
                )

        return to_return

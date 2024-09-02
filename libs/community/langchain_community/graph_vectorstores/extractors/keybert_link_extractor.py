from typing import Any, Dict, Iterable, Optional, Set, Union

from langchain_core._api import beta
from langchain_core.documents import Document
from langchain_core.graph_vectorstores.links import Link

from langchain_community.graph_vectorstores.extractors.link_extractor import (
    LinkExtractor,
)

KeybertInput = Union[str, Document]


@beta()
class KeybertLinkExtractor(LinkExtractor[KeybertInput]):
    def __init__(
        self,
        *,
        kind: str = "kw",
        embedding_model: str = "all-MiniLM-L6-v2",
        extract_keywords_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Extract keywords using KeyBERT <https://maartengr.github.io/KeyBERT/>.

        Example:

            .. code-block:: python

                extractor = KeybertLinkExtractor()

                results = extractor.extract_one(PAGE_1)

        Args:
            kind: Kind of links to produce with this extractor.
            embedding_model: Name of the embedding model to use with KeyBERT.
            extract_keywords_kwargs: Keyword arguments to pass to KeyBERT's
                `extract_keywords` method.
        """
        try:
            import keybert

            self._kw_model = keybert.KeyBERT(model=embedding_model)
        except ImportError:
            raise ImportError(
                "keybert is required for KeybertLinkExtractor. "
                "Please install it with `pip install keybert`."
            ) from None

        self._kind = kind
        self._extract_keywords_kwargs = extract_keywords_kwargs or {}

    def extract_one(self, input: KeybertInput) -> Set[Link]:  # noqa: A002
        keywords = self._kw_model.extract_keywords(
            input if isinstance(input, str) else input.page_content,
            **self._extract_keywords_kwargs,
        )
        return {Link.bidir(kind=self._kind, tag=kw[0]) for kw in keywords}

    def extract_many(
        self,
        inputs: Iterable[KeybertInput],
    ) -> Iterable[Set[Link]]:
        inputs = list(inputs)
        if len(inputs) == 1:
            # Even though we pass a list, if it contains one item, keybert will
            # flatten it. This means it's easier to just call the special case
            # for one item.
            yield self.extract_one(inputs[0])
        elif len(inputs) > 1:
            strs = [i if isinstance(i, str) else i.page_content for i in inputs]
            extracted = self._kw_model.extract_keywords(
                strs, **self._extract_keywords_kwargs
            )
            for keywords in extracted:
                yield {Link.bidir(kind=self._kind, tag=kw[0]) for kw in keywords}

"Extract keywords using jieba"
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union
from langchain_core._api import beta
from langchain_core.documents import Document
from langchain_community.graph_vectorstores.extractors.link_extractor import (
    LinkExtractor
)
from langchain_community.graph_vectorstores.links import Link

KeybertInput = Union[str, Document]

@beta()
class JiebaLinkExtractor(LinkExtractor[KeybertInput]):
    """
    Extract keywords using jieba.analyse(https://github.com/fxsjy/jieba)
    `jieba` is a popular Chinese text segmentation library and the `analyse` module
    can extract keywords from Chinese document.
    The JiebaLinkExtractor uses jieba.analyse to create links between documents
    that have keywords in common.
    Example::
        extractor = JiebaLinkExtractor()
        results = extractor.extract_one("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")

        
    How to apply JiebaLinkExtractor to 
    langchain_community.graph_vectorstores.extractors.LinkExtractorTransformer
    Example::
        from langchain_community.graph_vectorstores.extractors import LinkExtractorTransformer
        pipeline = LinkExtractorTransformer([JiebaLinkExtractor()])
        results = pipeline.transform_documents(docs)


    Arguments:
        kind: The kind of the link.
        analyzer: The analyzer to use. Must be one of `tfidf`, `textrank`, or `mixed`.
        weight: Only works when the analyzer is `mixed`. 
        Must be a tuple of length 2 and sum to 1.
        extract_keywords_kwargs: Keyword arguments to pass to jieba.analyse.extract_keywords.
        stop_words_path: Path to a file containing stop words.
    Example::
        extractor = JiebaLinkExtractor(
            analyzer='mixed', weight=(0.3, 0.7), 
            extract_keywords_kwargs={'topK': 5, 
            'withWeight': True, 
            'allowPOS': ('n','nr','ns','nt','nz')}
        )
        results = extractor.extract_one("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
    """
    def __init__(
            self,
            *,
            kind: str = 'kw',
            analyzer: str = 'tfidf',
            weight: Tuple[float, float] = (0.5, 0.5),
            extract_keywords_kwargs: Optional[Dict[str, Any]] = None,
            stop_words_path: Optional[str] = None
        ):
        try:
            from jieba.analyse import set_stop_words
            self._kind = kind
            self._analyzer = analyzer
            self._weight = weight
            if extract_keywords_kwargs is None:
                extract_keywords_kwargs = {
                    "topK": 3, 
                    "withWeight": False, 
                    "allowPOS": ('n','nr','ns','nt','nz','a','an','v','vn'), 
                    "withFlag":False
                }
            self._extract_keywords_kwargs = extract_keywords_kwargs
            if stop_words_path is not None:
                set_stop_words(stop_words_path)
        except ImportError as exc:
            raise ImportError(
                "jieba is required for JiebaLinkExtractor. "
                "Please install it with `pip install jieba`."
            ) from exc
    def extract_one(
            self,
            input: KeybertInput
        ) -> Set[Link]:
        from jieba.analyse.tfidf import TFIDF
        from jieba.analyse.textrank import TextRank
        if isinstance(input, Document):
            input = input.page_content
        if self._analyzer == 'tfidf':
            keywords = TFIDF().extract_tags(input, **self._extract_keywords_kwargs)
        elif self._analyzer == 'textrank':
            keywords = TextRank().extract_tags(input, **self._extract_keywords_kwargs)
        elif self._analyzer == 'mixed':
            self._extract_keywords_kwargs['withWeight'] = True
            if len(self._weight) != 2 or self._weight[0]+self._weight[1] != 1:
                raise ValueError("when analyzer is 'mixed', "
                "weight must be a tuple of length 2 and sum to 1")
            keywords_tfidf:Dict[str,Tuple[str, float]] = {
                keyword[0]:keyword for keyword in TFIDF().extract_tags(
                    input, **self._extract_keywords_kwargs
                )
            }
            keywords_textrank:Dict[str,Tuple[str, float]] = {
                keyword[0]:keyword for keyword in TextRank().extract_tags(
                    input, **self._extract_keywords_kwargs
                )
            }
            mixed_keywords = {}
            for key, value in keywords_tfidf.items():
                if key in keywords_textrank:
                    mixed_keywords[key] = (
                        key, value[1] * self._weight[0]
                        + keywords_textrank[key][1] * self._weight[1]
                    )
                else:
                    mixed_keywords[key] = (key, value[1])
            for key, value in keywords_textrank.items():
                if key not in keywords_tfidf:
                    mixed_keywords[key] = (key, value[1])
            keywords = sorted(mixed_keywords.values(), key=lambda x: x[1], reverse=True)
            keywords = keywords[:self._extract_keywords_kwargs['topK']]
        else:
            raise ValueError("analyzer must be 'tfidf','textrank' or 'mixed'")
        if self._extract_keywords_kwargs['withWeight']:
            keywords = [kw[0] for kw in keywords]
        return {Link.bidir(kind=self._kind, tag=kw) for kw in keywords}

    def extract_many(
            self,
            inputs: Iterable[KeybertInput],
        ) -> Iterable[Set[Link]]:
        for input in inputs:
            yield self.extract_one(input)

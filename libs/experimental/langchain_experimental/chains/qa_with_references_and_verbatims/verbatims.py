import logging
import re
from typing import List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel
from langchain.schema import BaseOutputParser

logger = logging.getLogger(__name__)


def _extract_original_verbatim(verbatim: str, page_content: str) -> Optional[str]:
    """The exact format of verbatim may be changed by the LLM.
    Extract only the words of the verbatim, and try to find a sequence
    of same words in the original document.
    """
    only_words = filter(len, re.split(r"[^\w]+", verbatim))
    regex_for_words_in_same_oder = r"(?i)" + r"\b\b[^\w]+".join(only_words)
    regex_for_words_in_same_oder += r"\b\s*[.!?:;]?"  # Optional end of sentence
    match = re.search(regex_for_words_in_same_oder, page_content, re.IGNORECASE)
    if match:
        return match[0].strip()
    return None  # No verbatim found in the original document


class VerbatimsFromDoc(BaseModel):
    ids: List[str]
    """ The position of the document in the original list """
    verbatims: List[str]
    """ All verbatims for this document """

    def original_verbatims(self, page_content: str) -> List[str]:
        result = []
        for j, verbatim in enumerate(self.verbatims):
            original_verbatim = _extract_original_verbatim(
                verbatim=verbatim, page_content=page_content
            )
            if original_verbatim:
                result.append(original_verbatim)
            else:
                logger.debug(f'Ignore verbatim "{verbatim}"')
        return result

    def __str__(self) -> str:
        return "\n".join(
            [f'"{v}"' for v in [re.sub(r"\s", " ", v) for v in self.verbatims]]
        )


class Verbatims(BaseModel):
    """
    Response, references and verbatims object.
    """

    response: str
    """ The response """
    documents: List[VerbatimsFromDoc]
    """ The list of documents and verbatims"""


verbatims_from_doc_parser: BaseOutputParser = PydanticOutputParser(
    pydantic_object=VerbatimsFromDoc
)
verbatims_parser: BaseOutputParser = PydanticOutputParser(pydantic_object=Verbatims)
empty_value: str = Verbatims(
    response="I don't known", documents=[]
).json()  # Pydantic 1

""" A parser for this object """

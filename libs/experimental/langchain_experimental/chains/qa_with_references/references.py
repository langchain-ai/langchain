import logging
import re
from typing import Set

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Extra
from langchain.schema import BaseOutputParser

logger = logging.getLogger(__name__)

# To optimize the consumption of tokens, it's better to use only 'text', without json.
# Else the schema consume ~300 tokens and the response 20 tokens by step
_OPTIMIZE = True  # Experimental


class References(BaseModel):
    """
    Response and referenced documents.
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    response: str
    """ The response """
    documents_ids: Set[str] = set()
    """ The list of documents used to response """

    def __str__(self) -> str:
        if _OPTIMIZE:
            return f'{self.response}\nIDS:{",".join(map(str, self.documents_ids))}'
        else:
            return self.json()  # Pydantic 1


references_parser: BaseOutputParser
if _OPTIMIZE:

    class _ReferencesParser(BaseOutputParser):
        """An optimised parser for Reference.
        It's more effective than the pydantic approach
        """

        @property
        def lc_serializable(self) -> bool:
            return True

        @property
        def _type(self) -> str:
            """Return the type key."""
            return "reference_parser"

        def get_format_instructions(self) -> str:
            return (
                "Your response should be in the form:\n"
                "Answer: the response.\n"
                "At the start of a new line 'IDS:' followed by a comma-separated "
                "list of document identifiers used "
                "in the response. The ids must have the form _idx_<number>.\n"
                # "\n"
                # "Answer: my response\n"
                # "IDS: _idx_1, _idx_2\n\n"
            )

        def parse(self, text: str) -> References:
            regex = r"(?i)(?:Answer:)?(.*)\sIDS:(.*)"
            match = re.search(regex, text)
            if match:
                ids: Set[str] = set()
                for str_doc_id in match[2].split(","):
                    m = re.match(r"\s*(?:_idx_)?(\d+)\s*", str_doc_id.strip())
                    if m:
                        ids.add(m[1])

                return References(response=match[1].strip(), documents_ids=ids)
            else:
                raise ValueError(f"Could not parse output: {text}")

    references_parser = _ReferencesParser()
    empty_value = "I don't know"
else:
    references_parser = PydanticOutputParser(pydantic_object=References)
    empty_value = References(
        response="I don't know", documents_ids=set()
    ).json()  # Pydantic 1

"""Util that calls bibtexparser."""
import logging
from typing import Any, Dict, List, Mapping

from langchain.pydantic_v1 import BaseModel, Extra, root_validator

logger = logging.getLogger(__name__)

OPTIONAL_FIELDS = [
    "annotate",
    "booktitle",
    "editor",
    "howpublished",
    "journal",
    "keywords",
    "note",
    "organization",
    "publisher",
    "school",
    "series",
    "type",
    "doi",
    "issn",
    "isbn",
]


class BibtexparserWrapper(BaseModel):
    """Wrapper around bibtexparser.

    To use, you should have the ``bibtexparser`` python package installed.
    https://bibtexparser.readthedocs.io/en/master/

    This wrapper will use bibtexparser to load a collection of references from
    a bibtex file and fetch document summaries.
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import bibtexparser  # noqa
        except ImportError:
            raise ImportError(
                "Could not import bibtexparser python package. "
                "Please install it with `pip install bibtexparser`."
            )

        return values

    def load_bibtex_entries(self, path: str) -> List[Dict[str, Any]]:
        """Load bibtex entries from the bibtex file at the given path."""
        import bibtexparser

        with open(path) as file:
            entries = bibtexparser.load(file).entries
        return entries

    def get_metadata(
        self, entry: Mapping[str, Any], load_extra: bool = False
    ) -> Dict[str, Any]:
        """Get metadata for the given entry."""
        publication = entry.get("journal") or entry.get("booktitle")
        if "url" in entry:
            url = entry["url"]
        elif "doi" in entry:
            url = f'https://doi.org/{entry["doi"]}'
        else:
            url = None
        meta = {
            "id": entry.get("ID"),
            "published_year": entry.get("year"),
            "title": entry.get("title"),
            "publication": publication,
            "authors": entry.get("author"),
            "abstract": entry.get("abstract"),
            "url": url,
        }
        if load_extra:
            for field in OPTIONAL_FIELDS:
                meta[field] = entry.get(field)
        return {k: v for k, v in meta.items() if v is not None}

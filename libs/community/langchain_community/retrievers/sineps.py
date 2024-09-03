"""
Provides similar query retrieval functionality as SelfQueryRetriever 
but uses Sineps' Filter Extractor for query construction.
For date-type fields, Sineps support the current_date placeholder 
to indicate the date when the query is made.
Please refer to the Sineps' documentation(https://docs.sineps.io/docs/docs/guides_and_concepts/filter_extractor) 
for more information.
"""

import logging
import os
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Tuple

import sineps
from langchain.retrievers.self_query.base import _get_builtin_translator
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    FilterDirective,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class SinepsAttributeInfo(BaseModel):
    name: str
    type: str
    description: str
    values: List[str] = None

    def to_dict(self):
        if self.values is None:
            return {
                "name": self.name,
                "type": self.type,
                "description": self.description,
            }
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "values": self.values,
        }


def _calculate_date(
    sineps_date_representation: str, current_date: datetime.date
) -> datetime.date:
    match = re.match(
        r"\$\(\s*current_date\s*-\s*(\d+)([ymd])\s*\)",
        sineps_date_representation.strip(),
    )
    if not match:
        raise ValueError("Invalid date format")

    value, unit = int(match.group(1)), match.group(2)

    if unit == "y":
        result_date = current_date.replace(year=current_date.year - value)
    elif unit == "m":
        new_month = current_date.month - value
        new_year = current_date.year
        while new_month <= 0:
            new_month += 12
            new_year -= 1
        result_date = current_date.replace(year=new_year, month=new_month)
    elif unit == "d":
        result_date = current_date - timedelta(days=value)
    else:
        raise ValueError("Invalid date unit")

    return result_date


def _translate_sineps_filter_node(
    node: dict, attribute: str, current_date: date = None
) -> FilterDirective:
    if not node:
        return None
    if node.type == "ConjunctedFilter":
        arguments = [
            _translate_sineps_filter_node(arg, attribute, current_date)
            for arg in node.filters
        ]
        return Operation(
            operator=Operator.AND if node.conjunction == "AND" else Operator.OR,
            arguments=[arg for arg in arguments if arg is not None],
        )
    elif node.type == "Filter":
        if current_date is None:
            current_date = date.today()
        value = node.value
        if value.startswith("$("):
            value = _calculate_date(value, current_date)
        else:
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass
        if node.operator == "NOT CONTAIN":
            return Operation(
                operator=Operator.NOT,
                arguments=[
                    Comparison(
                        comparator=Comparator.CONTAIN,
                        attribute=attribute,
                        value=value,
                    )
                ],
            )
        if node.operator == "=":
            comparator = Comparator.EQ
        elif node.operator == "!=":
            comparator = Comparator.NE
        elif node.operator == "<":
            comparator = Comparator.LT

        elif node.operator == "<=":
            comparator = Comparator.LTE
        elif node.operator == ">":
            comparator = Comparator.GT
        elif node.operator == ">=":
            comparator = Comparator.GTE
        elif node.operator == "CONTAIN":
            comparator = Comparator.CONTAIN

        return Comparison(
            comparator=comparator,
            attribute=attribute,
            value=value,
        )


class SinepsSelfQueryRetriever(BaseRetriever):
    """Provides similar query retrieval functionality as SelfQueryRetriever
    but uses Sineps' Filter Extractor for query construction."""

    sineps_api_key: str = None
    vectorstore: VectorStore
    """The underlying vector store from which documents will be retrieved."""
    search_type: str = "similarity"
    """The search type to perform on the vector store."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass in to the vector store search."""
    structured_query_translator: Visitor
    """Translator for turning internal query language into vectorstore search params."""
    sineps_metadata_field_info: List[SinepsAttributeInfo]
    verbose: bool = False

    @root_validator(pre=True)
    def validate_translator(cls, values: Dict) -> Dict:
        """Validate translator."""
        if "structured_query_translator" not in values:
            values["structured_query_translator"] = _get_builtin_translator(
                values["vectorstore"]
            )
        return values

    def _prepare_query(
        self, query: str, structured_query: StructuredQuery
    ) -> Tuple[str, Dict[str, Any]]:
        new_query, new_kwargs = self.structured_query_translator.visit_structured_query(
            structured_query
        )
        if structured_query.limit is not None:
            new_kwargs["k"] = structured_query.limit
        search_kwargs = {**self.search_kwargs, **new_kwargs}
        return new_query, search_kwargs

    def _get_docs_with_query(
        self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:
        docs = self.vectorstore.search(query, self.search_type, **search_kwargs)
        return docs

    async def _aget_docs_with_query(
        self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:
        docs = await self.vectorstore.asearch(query, self.search_type, **search_kwargs)
        return docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        current_date: date = None,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        client = sineps.Client(
            api_key=self.sineps_api_key or os.getenv("SINEPS_API_KEY")
        )
        filters = []
        for field in self.sineps_metadata_field_info:
            try:
                response = client.exec_filter_extractor(
                    query=query, field=field.to_dict()
                )
            except sineps.APIStatusError as e:
                response_text = f"{e.status_code}: {e.message}"
                run_manager.on_text(
                    response_text, color="red", end="\n", verbose=self.verbose
                )
                raise ValueError(f"Response is Invalid: {response_text}")
            filter = _translate_sineps_filter_node(
                response.result, field.name, current_date=current_date
            )
            if filter is not None:
                filters.append(filter)
        if len(filters) >= 2:
            structured_query = StructuredQuery(
                query=query,
                filter=Operation(operator=Operator.AND, arguments=filters),
            )
        elif len(filters) == 1:
            structured_query = StructuredQuery(
                query=query,
                filter=filters[0],
            )
        else:
            structured_query = StructuredQuery(query=query)

        if self.verbose:
            logger.info(f"Generated Query: {structured_query}")
        new_query, search_kwargs = self._prepare_query(query, structured_query)
        docs = self._get_docs_with_query(new_query, search_kwargs)
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        current_date: datetime.date = None,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        client = sineps.AsyncClient(
            api_key=self.sineps_api_key or os.getenv("SINEPS_API_KEY")
        )
        filters = []
        for field in self.sineps_metadata_field_info:
            try:
                response = await client.exec_filter_extractor(
                    query=query, field=field.to_dict()
                )
            except sineps.APIStatusError as e:
                response_text = f"{e.status_code}: {e.message}"
                run_manager.on_text(
                    response_text, color="red", end="\n", verbose=self.verbose
                )
                raise ValueError(f"Response is Invalid: {response_text}")
            filter = _translate_sineps_filter_node(
                response.result, field.name, current_date=current_date
            )
            if filter is not None:
                filters.append(filter)
        if len(filters) >= 2:
            structured_query = StructuredQuery(
                query=query,
                filter=Operation(operator=Operator.AND, arguments=filters),
            )
        elif len(filters) == 1:
            structured_query = StructuredQuery(
                query=query,
                filter=filters[0],
            )
        else:
            structured_query = StructuredQuery(query=query)

        if self.verbose:
            logger.info(f"Generated Query: {structured_query}")
        new_query, search_kwargs = self._prepare_query(query, structured_query)
        docs = self._get_docs_with_query(new_query, search_kwargs)
        return docs

from __future__ import annotations

import os
from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
)

if TYPE_CHECKING:
    import SPARQLWrapper


class OntotextGraphDBGraph:
    """Ontotext GraphDB https://graphdb.ontotext.com/ wrapper for graph operations."""

    def __init__(
        self,
        gdb_repository: str,
        custom_http_headers: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Set up Ontotext GraphDB wrapper

        :param gdb_repository: GraphDB repository URL, read access
        If GraphDB is secured,
        either set the environment variables 'GRAPHDB_USERNAME' and 'GRAPHDB_PASSWORD'
        or set the appropriate custom_http_headers for authentication.
        :param custom_http_headers: Custom HTTP headers to pass to GraphDB.
        """

        try:
            from SPARQLWrapper import SPARQLWrapper2
        except ImportError:
            raise ImportError(
                "Could not import sparqlwrapper python package. "
                "Please install it with `pip install sparqlwrapper`."
            )

        self.__sparql_wrapper = SPARQLWrapper2(gdb_repository)
        self.__config_sparql_wrapper(custom_http_headers)
        self.__check_connectivity()

    def __config_sparql_wrapper(
        self,
        custom_http_headers: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Configure authentication and add custom HTTP headers
        """
        from SPARQLWrapper import Wrapper

        gdb_username, gdb_password = self.__get_auth()
        if gdb_username:
            self.__sparql_wrapper.setHTTPAuth(Wrapper.BASIC)
            self.__sparql_wrapper.setCredentials(gdb_username, gdb_password)

        if custom_http_headers:
            for httpHeaderName, httpHeaderValue in custom_http_headers.items():
                self.__sparql_wrapper.addCustomHttpHeader(
                    httpHeaderName, httpHeaderValue
                )

    @staticmethod
    def __get_auth() -> tuple:
        """
        Returns the basic authentication configuration
        """
        username = os.environ.get("GRAPHDB_USERNAME", None)
        password = os.environ.get("GRAPHDB_PASSWORD", None)

        if username and not password:
            raise ValueError(
                "Environment variable 'GRAPHDB_USERNAME' is set, "
                "but 'GRAPHDB_PASSWORD' is not set."
            )
        return username, password

    def __check_connectivity(self) -> None:
        """
        Executes a simple `ASK` query to check connectivity
        """
        try:
            self.exec_query("ASK { ?s ?p ?o }")
        except Exception:
            raise ValueError(
                "Could not query the provided repository. "
                "Please, check, if the value of the provided "
                "gdb_repository points to the right repository. "
                "If GraphDB is secured, please, "
                "make sure that the environment variables "
                "'GRAPHDB_USERNAME' and 'GRAPHDB_PASSWORD' are set, "
                "or the correct authentication headers are set "
                "in custom_http_headers."
            )

    def exec_query(
        self,
        query: str,
    ) -> Union[
        Union[SPARQLWrapper.SmartWrapper.Bindings, SPARQLWrapper.QueryResult],
        SPARQLWrapper.QueryResult.ConvertResult,
    ]:
        self.__sparql_wrapper.setQuery(query)
        return self.__sparql_wrapper.queryAndConvert()

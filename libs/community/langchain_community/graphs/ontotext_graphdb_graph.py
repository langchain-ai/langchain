from __future__ import annotations

import os
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
)

if TYPE_CHECKING:
    import SPARQLWrapper


class OntotextGraphDBGraph:
    """Ontotext GraphDB https://graphdb.ontotext.com/ wrapper for graph operations.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(
        self,
        gdb_repository: str,
        ontology_query: Optional[str] = None,
        ontology_file_path: Optional[Path] = None,
        ontology_file_format: Optional[str] = None,
    ) -> None:
        """
        Set up the GraphDB wrapper

        :param gdb_repository: GraphDB repository URL, read access

        If GraphDB is secured,
        set the environment variables 'GRAPHDB_USERNAME' and 'GRAPHDB_PASSWORD'.

        :param ontology_query: a `CONSTRUCT` query that is executed
        on the SPARQL endpoint and returns the KG schema statements
        Example:
        'CONSTRUCT {?s ?p ?o} FROM <https://example.com/ontology/> WHERE {?s ?p ?o}'
        Currently, DESCRIBE queries like
        'PREFIX onto: <https://example.com/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        DESCRIBE ?term WHERE {
            ?term rdfs:isDefinedBy onto:
        }'
        are not supported, because DESCRIBE returns
        the Symmetric Concise Bounded Description (SCBD),
        i.e. also the incoming class links.
        In case of large graphs with a million of instances, this is not efficient.
        Check https://github.com/eclipse-rdf4j/rdf4j/issues/4857

        :param ontology_file_path: path to a local RDF ontology file.
        Supported RDF formats:
        Turtle, RDF/XML, JSON-LD, N-Triples, Notation-3, Trig, Trix, N-Quads.
        If the rdf format can't be determined from the file extension,
        pass explicitly the rdf format in `ontology_file_format` param.

        :param ontology_file_format: Used if the rdf format can't be determined
        from the local file extension.
        One of "json-ld", "xml", "n3", "turtle", "nt", "trig", "nquads", "trix"

        Either `ontology_schema_sparql_query` or
        `ontology_schema_file_path` should be passed.
        """

        if ontology_query and ontology_file_path:
            raise ValueError("Both file and query provided. Only one is allowed.")

        if not ontology_query and not ontology_file_path:
            raise ValueError("Neither file nor query provided. One is required.")

        try:
            from SPARQLWrapper import SPARQLWrapper2
        except ImportError:
            raise ImportError(
                "Could not import sparqlwrapper python package. "
                "Please install it with `pip install sparqlwrapper`."
            )

        self.gdb_username, self.gdb_password = self._get_auth()
        self.gdb_repository = gdb_repository
        self.sparql_wrapper = SPARQLWrapper2(gdb_repository)
        self._set_credentials(self.sparql_wrapper)
        self._check_connectivity()

        if ontology_file_path:
            self._validate_file(ontology_file_path)
            self.schema = self._load_ontology_schema_from_file(
                ontology_file_path,
                ontology_file_format,
            )
        else:
            self._validate_user_query(ontology_query)
            self.schema = self._load_ontology_schema_with_query(ontology_query)

    @staticmethod
    def _get_auth() -> tuple:
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

    def _set_credentials(self, sparql_wrapper: SPARQLWrapper) -> None:
        """
        Configure the basic authentication
        """
        from SPARQLWrapper import Wrapper

        if self.gdb_username:
            sparql_wrapper.setHTTPAuth(Wrapper.BASIC)
            sparql_wrapper.setCredentials(self.gdb_username, self.gdb_password)

    def _check_connectivity(self) -> None:
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
                "'GRAPHDB_USERNAME' and 'GRAPHDB_PASSWORD' are set."
            )

    @staticmethod
    def _validate_file(file: Path) -> None:
        """
        Validate that the file exists and the permissions are ok
        """
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} does not exist.")
        if not os.access(file, os.R_OK):
            raise PermissionError(f"Read permission for {file} is restricted")

    @staticmethod
    def _load_ontology_schema_from_file(
        file: Path, file_format: Optional[str] = None
    ) -> str:
        """
        Parse the ontology schema statements from the provided file
        and return the results serialized in turtle
        """
        import rdflib

        graph = rdflib.ConjunctiveGraph()
        try:
            graph.parse(file, format=file_format)
        except Exception as e:
            raise ValueError(f"Invalid file format for {file} : ", e)
        return graph.serialize(format="turtle")

    @staticmethod
    def _validate_user_query(query: Optional[str]) -> None:
        """
        Validate the query is a valid SPARQL CONSTRUCT query
        """
        from pyparsing import ParseException
        from rdflib.plugins.sparql import prepareQuery

        if not isinstance(query, str):
            raise TypeError("Ontology query must be provided as string.")
        try:
            parsed_query = prepareQuery(query)
        except ParseException as e:
            raise ValueError("Ontology query is not a valid SPARQL query.", e)

        if parsed_query.algebra.name != "ConstructQuery":
            raise ValueError(
                "Invalid query type. Only CONSTRUCT queries are supported."
            )

    def _load_ontology_schema_with_query(self, query: Optional[str]) -> str:
        """
        Execute the query for collecting the ontology schema statements
        and return the results serialized in turtle
        """
        from SPARQLWrapper import SPARQLWrapper

        sparql_wrapper = SPARQLWrapper(self.gdb_repository)
        self._set_credentials(sparql_wrapper)
        sparql_wrapper.setQuery(query)
        res = sparql_wrapper.queryAndConvert()
        return res.serialize(format="turtle")

    @property
    def get_schema(self) -> str:
        """
        Returns the schema of the graph database in turtle format
        """
        return self.schema

    def exec_query(
        self,
        query: str,
    ) -> Union[
        Union[SPARQLWrapper.SmartWrapper.Bindings, SPARQLWrapper.QueryResult],
        SPARQLWrapper.QueryResult.ConvertResult,
    ]:
        self.sparql_wrapper.setQuery(query)
        return self.sparql_wrapper.queryAndConvert()

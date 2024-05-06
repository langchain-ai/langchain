from __future__ import annotations

import os
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Union,
)

if TYPE_CHECKING:
    import rdflib


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
        query_endpoint: str,
        query_ontology: Optional[str] = None,
        local_file: Optional[str] = None,
        local_file_format: Optional[str] = None,
    ) -> None:
        """
        Set up the GraphDB wrapper

        :param query_endpoint: SPARQL endpoint for queries, read access

        If GraphDB is secured,
        set the environment variables 'GRAPHDB_USERNAME' and 'GRAPHDB_PASSWORD'.

        :param query_ontology: a `CONSTRUCT` query that is executed
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

        :param local_file: a local RDF ontology file.
        Supported RDF formats:
        Turtle, RDF/XML, JSON-LD, N-Triples, Notation-3, Trig, Trix, N-Quads.
        If the rdf format can't be determined from the file extension,
        pass explicitly the rdf format in `local_file_format` param.

        :param local_file_format: Used if the rdf format can't be determined
        from the local file extension.
        One of "json-ld", "xml", "n3", "turtle", "nt", "trig", "nquads", "trix"

        Either `query_ontology` or `local_file` should be passed.
        """

        if query_ontology and local_file:
            raise ValueError("Both file and query provided. Only one is allowed.")

        if not query_ontology and not local_file:
            raise ValueError("Neither file nor query provided. One is required.")

        try:
            import rdflib
            from rdflib.plugins.stores import sparqlstore
        except ImportError:
            raise ImportError(
                "Could not import rdflib python package. "
                "Please install it with `pip install rdflib`."
            )

        auth = self._get_auth()
        store = sparqlstore.SPARQLStore(auth=auth)
        store.open(query_endpoint)

        self.graph = rdflib.Graph(store, identifier=None, bind_namespaces="none")
        self._check_connectivity()

        if local_file:
            ontology_schema_graph = self._load_ontology_schema_from_file(
                local_file,
                local_file_format,  # type: ignore[arg-type]
            )
        else:
            self._validate_user_query(query_ontology)  # type: ignore[arg-type]
            ontology_schema_graph = self._load_ontology_schema_with_query(
                query_ontology  # type: ignore[arg-type]
            )
        self.schema = ontology_schema_graph.serialize(format="turtle")

    @staticmethod
    def _get_auth() -> Union[tuple, None]:
        """
        Returns the basic authentication configuration
        """
        username = os.environ.get("GRAPHDB_USERNAME", None)
        password = os.environ.get("GRAPHDB_PASSWORD", None)

        if username:
            if not password:
                raise ValueError(
                    "Environment variable 'GRAPHDB_USERNAME' is set, "
                    "but 'GRAPHDB_PASSWORD' is not set."
                )
            else:
                return username, password
        return None

    def _check_connectivity(self) -> None:
        """
        Executes a simple `ASK` query to check connectivity
        """
        try:
            self.graph.query("ASK { ?s ?p ?o }")
        except ValueError:
            raise ValueError(
                "Could not query the provided endpoint. "
                "Please, check, if the value of the provided "
                "query_endpoint points to the right repository. "
                "If GraphDB is secured, please, "
                "make sure that the environment variables "
                "'GRAPHDB_USERNAME' and 'GRAPHDB_PASSWORD' are set."
            )

    @staticmethod
    def _load_ontology_schema_from_file(local_file: str, local_file_format: str = None):  # type: ignore[no-untyped-def, assignment]
        """
        Parse the ontology schema statements from the provided file
        """
        import rdflib

        if not os.path.exists(local_file):
            raise FileNotFoundError(f"File {local_file} does not exist.")
        if not os.access(local_file, os.R_OK):
            raise PermissionError(f"Read permission for {local_file} is restricted")
        graph = rdflib.ConjunctiveGraph()
        try:
            graph.parse(local_file, format=local_file_format)
        except Exception as e:
            raise ValueError(f"Invalid file format for {local_file} : ", e)
        return graph

    @staticmethod
    def _validate_user_query(query_ontology: str) -> None:
        """
        Validate the query is a valid SPARQL CONSTRUCT query
        """
        from pyparsing import ParseException
        from rdflib.plugins.sparql import prepareQuery

        if not isinstance(query_ontology, str):
            raise TypeError("Ontology query must be provided as string.")
        try:
            parsed_query = prepareQuery(query_ontology)
        except ParseException as e:
            raise ValueError("Ontology query is not a valid SPARQL query.", e)

        if parsed_query.algebra.name != "ConstructQuery":
            raise ValueError(
                "Invalid query type. Only CONSTRUCT queries are supported."
            )

    def _load_ontology_schema_with_query(self, query: str):  # type: ignore[no-untyped-def]
        """
        Execute the query for collecting the ontology schema statements
        """
        from rdflib.exceptions import ParserError

        try:
            results = self.graph.query(query)
        except ParserError as e:
            raise ValueError(f"Generated SPARQL statement is invalid\n{e}")

        return results.graph

    @property
    def get_schema(self) -> str:
        """
        Returns the schema of the graph database in turtle format
        """
        return self.schema

    def query(
        self,
        query: str,
    ) -> List[rdflib.query.ResultRow]:
        """
        Query the graph.
        """
        from rdflib.query import ResultRow

        res = self.graph.query(query)
        return [r for r in res if isinstance(r, ResultRow)]

from __future__ import annotations

import os
import tempfile
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
)

import rdflib
from SPARQLWrapper import TURTLE, SPARQLWrapper

if TYPE_CHECKING:
    pass

prefixes = {
    "owl": """PREFIX owl: <http://www.w3.org/2002/07/owl#>\n""",
    "rdf": """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n""",
    "rdfs": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n""",
    "xsd": """PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n""",
}

cls_query_rdf = prefixes["rdfs"] + (
    """SELECT DISTINCT ?cls ?com\n"""
    """WHERE { \n"""
    """    ?instance a ?cls . \n"""
    """    OPTIONAL { ?cls rdfs:comment ?com } \n"""
    """}"""
)

cls_query_rdfs = prefixes["rdfs"] + (
    """SELECT DISTINCT ?cls ?com\n"""
    """WHERE { \n"""
    """    ?instance a/rdfs:subClassOf* ?cls . \n"""
    """    OPTIONAL { ?cls rdfs:comment ?com } \n"""
    """}"""
)

cls_query_owl = prefixes["rdfs"] + (
    """SELECT DISTINCT ?cls ?com\n"""
    """WHERE { \n"""
    """    ?instance a/rdfs:subClassOf* ?cls . \n"""
    """    FILTER (isIRI(?cls)) . \n"""
    """    OPTIONAL { ?cls rdfs:comment ?com } \n"""
    """}"""
)

rel_query_rdf = prefixes["rdfs"] + (
    """SELECT DISTINCT ?rel ?com\n"""
    """WHERE { \n"""
    """    ?subj ?rel ?obj . \n"""
    """    OPTIONAL { ?rel rdfs:comment ?com } \n"""
    """}"""
)

rel_query_rdfs = (
    prefixes["rdf"]
    + prefixes["rdfs"]
    + (
        """SELECT DISTINCT ?rel ?com\n"""
        """WHERE { \n"""
        """    ?rel a/rdfs:subPropertyOf* rdf:Property . \n"""
        """    OPTIONAL { ?rel rdfs:comment ?com } \n"""
        """}"""
    )
)

op_query_owl = (
    prefixes["rdfs"]
    + prefixes["owl"]
    + (
        """SELECT DISTINCT ?op ?com\n"""
        """WHERE { \n"""
        """    ?op a/rdfs:subPropertyOf* owl:ObjectProperty . \n"""
        """    OPTIONAL { ?op rdfs:comment ?com } \n"""
        """}"""
    )
)

dp_query_owl = (
    prefixes["rdfs"]
    + prefixes["owl"]
    + (
        """SELECT DISTINCT ?dp ?com\n"""
        """WHERE { \n"""
        """    ?dp a/rdfs:subPropertyOf* owl:DatatypeProperty . \n"""
        """    OPTIONAL { ?dp rdfs:comment ?com } \n"""
        """}"""
    )
)


class AnzoGraphDBGraph:
    def __init__(
        self,
        source_file: Optional[str] = None,
        query_endpoint: Optional[str] = None,
        update_endpoint: Optional[str] = None,
        query_ontology: Optional[str] = None,
        serialization: Optional[str] = "turtle",
        standard: Optional[str] = "rdf",
        local_copy: Optional[str] = None,
        graph_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Set up the AnzoGraph SPARQL graph

        :param source_file: either a path for a local file or a URL
        :param serialization: serialization of the input
        :param query_endpoint: SPARQL endpoint for queries, read access
        :param update_endpoint: SPARQL endpoint for UPDATE queries, write access
        :param standard: RDF, RDFS, or OWL
        :param local_copy: new local copy for storing changes
        :param query_ontology: SPARQL CONSTRUCT query against the AnzoGraph DB endpoint
        :param graph_kwargs: Additional AnzoGraph SPARQL graph specific kwargs
        that will be used to initialize it,
        if query_endpoint is provided.
        """
        self.source_file = source_file
        self.query_endpoint = query_endpoint
        self.update_endpoint = update_endpoint
        self.query_ontology = query_ontology
        self.serialization = serialization
        self.standard = standard
        self.local_copy = local_copy
        self.graph = rdflib.Graph(**(graph_kwargs or {}))
        self.schema = None

        if self.source_file:
            self.load_local_file()
        elif self.query_endpoint and self.query_ontology:
            self.perform_sparql_query()
        else:
            raise ValueError("Insufficient parameters to initialize graph.")

    def perform_sparql_query(self):
        """
        Execute SPARQL query and get serialized result.
        """
        sparql = SPARQLWrapper(self.query_endpoint)
        sparql.setQuery(self.query_ontology)
        sparql.setReturnFormat(TURTLE)
        query_result = sparql.query().convert()
        self.load_and_serialize_graph(query_result)

    def load_local_file(self):
        """
        Load RDF data from a local file.
        """
        self.graph.parse(self.source_file, format=self.serialization)

    def load_and_serialize_graph(self, query_result):
        """
        Load the SPARQL result into a graph, serialize it to a temporary file,
        and then reload it into the main graph.
        """
        temp_graph = rdflib.Graph()
        temp_graph.parse(data=query_result, format="turtle")
        g = temp_graph.serialize(format="turtle")

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp:
            temp.write(g)
            temp_file_path = temp.name

        try:
            self.graph = rdflib.Graph()
            self.graph.parse(temp_file_path, format=self.serialization)
        finally:
            os.unlink(temp_file_path)

    @property
    def get_schema(self) -> str:
        """
        Returns the schema of the graph database.
        """
        return self.schema

    def query(
        self,
        query: str,
    ) -> List[rdflib.query.ResultRow]:
        """
        Query the graph.
        """
        from rdflib.exceptions import ParserError
        from rdflib.query import ResultRow

        try:
            res = self.graph.query(query)
        except ParserError as e:
            raise ValueError("Generated SPARQL statement is invalid\n" f"{e}")
        return [r for r in res if isinstance(r, ResultRow)]

    def update(
        self,
        query: str,
    ) -> None:
        """
        Update the graph.
        """
        from rdflib.exceptions import ParserError

        try:
            self.graph.update(query)
        except ParserError as e:
            raise ValueError("Generated SPARQL statement is invalid\n" f"{e}")
        if self.local_copy:
            self.graph.serialize(
                destination=self.local_copy, format=self.local_copy.split(".")[-1]
            )
        else:
            raise ValueError("No target file specified for saving the updated file.")

    @staticmethod
    def _get_local_name(iri: str) -> str:
        if "#" in iri:
            local_name = iri.split("#")[-1]
        elif "/" in iri:
            local_name = iri.split("/")[-1]
        else:
            raise ValueError(f"Unexpected IRI '{iri}', contains neither '#' nor '/'.")
        return local_name

    def _res_to_str(self, res: rdflib.query.ResultRow, var: str) -> str:
        return (
            "<"
            + str(res[var])
            + "> ("
            + self._get_local_name(res[var])
            + ", "
            + str(res["com"])
            + ")"
        )

    def _rdf_s_schema(self, classes, relationships):
        """
        Constructs a schema description from given classes and relationships.
        """
        return (
            f"In the following, each IRI is followed by the local name and "
            f"optionally its description in parentheses. \n"
            f"The RDF graph supports the following node types:\n"
            f'{", ".join([self._res_to_str(cls, "cls") for cls in classes])}\n'
            f"The RDF graph supports the following relationships:\n"
            f'{", ".join([self._res_to_str(rel, "rel") for rel in relationships])}\n'
        )

    def load_schema(self):
        if self.query_endpoint:
            if self.standard == "rdf":
                clss = self.query(cls_query_rdf)
                rels = self.query(rel_query_rdf)
                self.schema = self._rdf_s_schema(clss, rels)
            elif self.standard == "rdfs":
                clss = self.query(cls_query_rdfs)
                rels = self.query(rel_query_rdfs)
                self.schema = self._rdf_s_schema(clss, rels)
            elif self.standard == "owl":
                clss = self.query(cls_query_owl)
                ops = self.query(op_query_owl)
                dps = self.query(dp_query_owl)
                self.schema = self._rdf_s_schema(clss, ops + dps)
            else:
                raise ValueError(f"Mode '{self.standard}' currently not supported.")
        else:
            self.schema = None

from typing import Any, Dict, List, Optional
import os
import logging

logger = logging.getLogger(__name__)

schema_query = """
CONSTRUCT {
  ?type a rdfs:Class .
  ?property a rdf:Property .
  ?property rdfs:domain ?type .
  ?property rdfs:range ?otype . 
  ?property rdfs:range ?dtype . 
}
WHERE {
  {
    ?s a ?type  .
    ?s ?property ?o .
    ?o a ?otype.
  } UNION {
    ?s a ?type  .
    ?s ?property ?o .
    BIND(datatype(?o) AS ?otype)
    FILTER(isLiteral(?o))
  }
}
"""


class RDFGraph:
    """RDF wrapper for graph operations.
    Compatible with RDF stores that support SPARQL 1.1.
    """

    def __init__(
        self, source: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, http_auth: Optional[str] = None, http_method: Optional[str] = 'GET'
    ) -> None:
        """Connect to an external RDF store or create an in-memory store."""
        self.source = source or "in-memory-store"
        try:
            from rdflib import Graph
            from SPARQLWrapper import SPARQLWrapper, JSON, RDFXML, BASIC, DIGEST, POST
        except ImportError:
            raise ValueError(
                "Please install rdflib and sparqlwrapper first: "
                "`pip install rdflib sparqlwrapper`"
            )

        if source is None:
            logger.info("Creating in-memory RDF store")
            self.graph = Graph()
        elif os.path.exists(source):
            self.graph = Graph()
            try:
                self.graph.parse(source)
            except Exception as e:
                raise ValueError(f"Could not parse RDF file {source}: {e}")
        elif source.startswith("http"):
            class ExternalGraph:
                """Wrapper for SPARQLWrapper to have the same APIs as RDFLib"""

                def __init__(self, url: str, username: Optional[str] = None, password: Optional[str] = None, http_auth: Optional[str] = None, http_method: Optional[str] = 'GET') -> None:
                    self.wrapper = SPARQLWrapper(url)
                    if username is not None and password is not None:
                        if http_auth.lower() == 'digest':
                            self.wrapper.setHTTPAuth(DIGEST)
                        else:
                            self.wrapper.setHTTPAuth(BASIC)
                        self.wrapper.setCredentials(username, password)

                def query(self, query: str) -> Dict[str, Any]:
                    query_type = Graph().query(query, use_store_provided=False).type
                    self.wrapper.setQuery(query)
                    self.wrapper.setMethod(http_method)
                    if query_type == 'CONSTRUCT':
                        self.wrapper.setReturnFormat(RDFXML)
                        return Graph().parse(data=self.wrapper.query().response.read(), format='xml')
                    self.wrapper.setReturnFormat(JSON)
                    return self.wrapper.queryAndConvert()['results']['bindings']
                
                def update(self, update: str) -> Dict[str, Any]:
                    self.wrapper.setQuery(update)
                    self.wrapper.setReturnFormat(RDFXML)
                    self.wrapper.setMethod(POST)
                    self.wrapper.query()
            
            self.graph = ExternalGraph(source, username, password, http_auth)
            try:
                self.graph.query("SELECT * WHERE {?s ?p ?o} LIMIT 1")
            except Exception as e:
                raise ValueError(f"Could not connect to external RDF store {source}: {e}")
        else:
            raise NotImplementedError(
                f"Connectivity to source {source} is not implemented yet"
            )

        # Set schema
        self.refresh_schema()

    @property
    def get_schema(self) -> str:
        """Returns the schema of the RDF database"""
        return self.schema

    def query(self, query: str) -> List[Dict[str, Any]]:
        """Query RDF database."""
        from rdflib import Graph
        try:
            results = self.graph.query(query)
            return list(results)           
        except Exception as e:
            raise ValueError(f"Could not query RDF store {self.source}: {e}")
            
    def update(self, update: str) -> None:
        """Update RDF database."""
        from rdflib.plugins.sparql import prepareQuery
        try:
            self.graph.update(update)
        except Exception as e:
            raise ValueError(f"Could not update RDF store {self.source}: {e}")

    def refresh_schema(self) -> None:
        """
        Refreshes the RDF graph schema information.
        """
        result = self.graph.query(schema_query).serialize(format='turtle')
        self.schema = result if type(result) is str else result.decode()

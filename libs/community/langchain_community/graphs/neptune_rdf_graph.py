import boto3
import json
import requests
import urllib.parse

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import ReadOnlyCredentials
from types import SimpleNamespace

from typing import (
    TYPE_CHECKING,
    List,
    Optional,
)

CLASS_QUERY = """
SELECT DISTINCT ?elem ?com
WHERE { 
 ?instance a ?elem .
 OPTIONAL { ?instance rdf:type/rdfs:subClassOf* ?elem } .
 #FILTER (isIRI(?elem)) .
 OPTIONAL { ?elem rdfs:comment ?com filter (lang(?com) = "en")}
}
"""

REL_QUERY = """
SELECT DISTINCT ?elem ?com
WHERE { 
 ?subj ?elem ?obj . 
 OPTIONAL { 
     ?elem rdf:type/rdfs:subPropertyOf* ?proptype .
     VALUES  ?proptype  { rdf:Property owl:DatatypeProperty owl:ObjectProperty } .
 } . 
 OPTIONAL { ?elem rdfs:comment ?com filter (lang(?com) = "en")} 
}
"""

DTPROP_QUERY = """
SELECT DISTINCT ?elem ?com
WHERE { 
 ?subj ?elem ?obj . 
 OPTIONAL { 
     ?elem rdf:type/rdfs:subPropertyOf* ?proptype .
     ?proptype  a owl:DatatypeProperty .
 } . 
 OPTIONAL { ?elem rdfs:comment ?com filter (lang(?com) = "en")} 
}
"""

OPROP_QUERY = """
SELECT DISTINCT ?elem ?com
WHERE { 
 ?subj ?elem ?obj . 
 OPTIONAL { 
     ?elem rdf:type/rdfs:subPropertyOf* ?proptype .
     ?proptype  a owl:ObjectProperty .
 } . 
 OPTIONAL { ?elem rdfs:comment ?com filter (lang(?com) = "en")} 
}
"""

ELEM_TYPES = {
	'classes': CLASS_QUERY, 
	'rels': REL_QUERY, 
	'dtprops': DTPROP_QUERY, 
	'oprops': OPROP_QUERY
}

class NeptuneRdfGraph:
    """Neptune wrapper for RDF graph operations.

    Args:
        query_endpoint: SPARQL endpoint for Neptune
        use_iam_auth: boolean indicating IAM auth is enabled in Neptune cluster
        region_name: AWS region required if use_iam_auth is True, e.g., us-west-2
        hide_comments: whether to include ontology comments in schema for prompt

    Example:
        .. code-block:: python

        graph = NeptuneRdfGraph(
            query_endpoint='<SPARQL endpoint>',
            use_iam_auth=False
        )
        schema = graph.get_schema()

        OR
        graph = NeptuneRdfGraph(
            query_endpoint='<SPARQL endpoint>',
            use_iam_auth=False
        )
        schema_elem = graph.get_schema_elements()
        ... change schema_elements ... 
        graph.load_from_schema_elements(schema_elem)
        schema = graph.get_schema()

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
        use_iam_auth: bool = False,
        region_name: Optional[str] = None,
        hide_comments: bool = False # we introspect comments, but they might bloat the prompt
    ) -> None:
        self.use_iam_auth = use_iam_auth
        self.region_name = region_name
        self.query_endpoint = query_endpoint
        self.hide_comments = hide_comments

        # Set schema
        self.schema = ""
        self.schema_elements = {}
        self.load_schema()
        
    @property
    def get_schema(self) -> str:
        """
        Returns the schema of the graph database.
        """
        return self.schema

    @property
    def get_schema_elements(self):
        return self.schema_elements
 
    '''
    Run Neptune query.
    '''
    def query(
        self,
        query: str,
    ):
        session = boto3.Session()
        request_data = {
            "query": query
        }
        data = request_data
        request_hdr = None

        if self.use_iam_auth:
            credentials = session.get_credentials()
            credentials = credentials.get_frozen_credentials()
            access_key = credentials.access_key
            secret_key = credentials.secret_key
            service = 'neptune-db'
            session_token = credentials.token
            params=None
            creds = SimpleNamespace(
                access_key=access_key, secret_key=secret_key, token=session_token, region=self.region_name)
            request = AWSRequest(method='POST', url=self.query_endpoint, data=data, params=params)
            SigV4Auth(creds, service, self.region_name).add_auth(request)
            request.headers['Content-Type']= 'application/x-www-form-urlencoded'
            request_hdr = request.headers
        else:
            request_hdr = {}
            request_hdr['Content-Type']= 'application/x-www-form-urlencoded'

        queryres = requests.request(method='POST', url=self.query_endpoint, headers=request_hdr, data=data)
        json_resp = json.loads(queryres.text)
        return json_resp

    '''
    This is a public method that allows the user to create schema from their own
    schema_elements. The anticipated use is that the user prunes the introspected schema.
    '''
    def load_from_schema_elements(self, schema_elements):

        elemstr={}
        for elem in ELEM_TYPES:
            reslist = []
            for elemrec in self.schema_elements[elem]:
                uri = elemrec['uri']
                local = elemrec['local']
                str = f"<{uri}> ({local})"
                if self.hide_comments is False:
                    str = str + f", {comment}"
                reslist.append(str)
            elemstr[elem] = ", ".join(reslist)

        self.schema = "".join([
            f"In the following, each IRI is followed by the local name and ", 
            f"optionally its description in parentheses. \n",
            f"The graph supports the following node types:\n", elemstr['classes'],
            f"The graph supports the following relationships:\n", elemstr['rels'],
            f"The graph supports the following OWL object properties, ", elemstr['dtprops'],
            "The graph supports the following OWL data properties, ", elemstr['oprops']
        ])

    '''
    Private method split URI into prefix and local
    '''
    @staticmethod
    def _get_local_name(iri: str):
        if "#" in iri:
            toks = iri.split("#")
            return [f"{toks[0]}#", toks[-1]]
        elif "/" in iri:
            toks = iri.split("/")
            return [f"{'/'.join(toks[0:len(toks)-1])}/", toks[-1]]
        else:
            raise ValueError(f"Unexpected IRI '{iri}', contains neither '#' nor '/'.")
        
    '''
    Query Neptune to introspect schema.
    '''      
    def load_schema(self) -> None:
        self.schema_elements['distinct_prefixes'] = {}

        for elem in ELEM_TYPES:
            items = self.query(ELEM_TYPES[elem])
            reslist = []
            for r in items['results']['bindings']:
                uri = r['elem']['value']
                toks = self._get_local_name(uri)
                elem_record = {'uri': uri, 'local': toks[1]}
                if self.hide_comments == False:
                    elem_record['comment'] = r['com']['value'] if 'com' in r else ""
                reslist.append(elem_record)
                if not(toks[0] in self.schema_elements['distinct_prefixes']):
                    self.schema_elements['distinct_prefixes'][toks[0]] = "y"
                       
            self.schema_elements[elem] = reslist

        self.load_from_schema_elements(self.schema_elements)
        
        
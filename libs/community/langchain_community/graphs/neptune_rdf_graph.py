import json
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

import requests

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
    "classes": CLASS_QUERY,
    "rels": REL_QUERY,
    "dtprops": DTPROP_QUERY,
    "oprops": OPROP_QUERY,
}


class NeptuneRdfGraph:
    """Neptune wrapper for RDF graph operations.

    Args:
        host: SPARQL endpoint host for Neptune
        port: SPARQL endpoint port for Neptune. Defaults 8182.
        use_iam_auth: boolean indicating IAM auth is enabled in Neptune cluster
        region_name: AWS region required if use_iam_auth is True, e.g., us-west-2
        hide_comments: whether to include ontology comments in schema for prompt

    Example:
        .. code-block:: python

        graph = NeptuneRdfGraph(
            host='<SPARQL host'>,
            port=<SPARQL port>,
            use_iam_auth=False
        )
        schema = graph.get_schema()

        OR
        graph = NeptuneRdfGraph(
            host='<SPARQL host'>,
            port=<SPARQL port>,
            use_iam_auth=False
        )
        schema_elem = graph.get_schema_elements()
        ... change schema_elements ...
        graph.load_schema(schema_elem)
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
        host: str,
        port: int = 8182,
        use_iam_auth: bool = False,
        region_name: Optional[str] = None,
        hide_comments: bool = False,
    ) -> None:
        self.use_iam_auth = use_iam_auth
        self.region_name = region_name
        self.hide_comments = hide_comments
        self.query_endpoint = f"https://{host}:{port}/sparql"

        if self.use_iam_auth:
            try:
                import boto3

                self.session = boto3.Session()
            except ImportError:
                raise ImportError(
                    "Could not import boto3 python package. "
                    "Please install it with `pip install boto3`."
                )
        else:
            self.session = None

        # Set schema
        self.schema = ""
        self.schema_elements: Dict[str, Any] = {}
        self._refresh_schema()

    @property
    def get_schema(self) -> str:
        """
        Returns the schema of the graph database.
        """
        return self.schema

    @property
    def get_schema_elements(self) -> Dict[str, Any]:
        return self.schema_elements

    def query(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Run Neptune query.
        """
        request_data = {"query": query}
        data = request_data
        request_hdr = None

        if self.use_iam_auth:
            credentials = self.session.get_credentials()
            credentials = credentials.get_frozen_credentials()
            access_key = credentials.access_key
            secret_key = credentials.secret_key
            service = "neptune-db"
            session_token = credentials.token
            params = None
            creds = SimpleNamespace(
                access_key=access_key,
                secret_key=secret_key,
                token=session_token,
                region=self.region_name,
            )
            from botocore.awsrequest import AWSRequest

            request = AWSRequest(
                method="POST", url=self.query_endpoint, data=data, params=params
            )
            from botocore.auth import SigV4Auth

            SigV4Auth(creds, service, self.region_name).add_auth(request)
            request.headers["Content-Type"] = "application/x-www-form-urlencoded"
            request_hdr = request.headers
        else:
            request_hdr = {}
            request_hdr["Content-Type"] = "application/x-www-form-urlencoded"

        queryres = requests.request(
            method="POST", url=self.query_endpoint, headers=request_hdr, data=data
        )
        json_resp = json.loads(queryres.text)
        return json_resp

    def load_schema(self, schema_elements: Dict[str, Any]) -> None:
        """
        Generates and sets schema from schema_elements. Helpful in
        cases where introspected schema needs pruning.
        """

        elem_str = {}
        for elem in ELEM_TYPES:
            res_list = []
            for elem_rec in self.schema_elements[elem]:
                uri = elem_rec["uri"]
                local = elem_rec["local"]
                res_str = f"<{uri}> ({local})"
                if self.hide_comments is False:
                    res_str = res_str + f", {elem_rec['comment']}"
                res_list.append(res_str)
            elem_str[elem] = ", ".join(res_list)

        self.schema = (
            "In the following, each IRI is followed by the local name and "
            "optionally its description in parentheses. \n"
            "The graph supports the following node types:\n"
            f"{elem_str['classes']}"
            "The graph supports the following relationships:\n"
            f"{elem_str['rels']}"
            "The graph supports the following OWL object properties, "
            f"{elem_str['dtprops']}"
            "The graph supports the following OWL data properties, "
            f"{elem_str['oprops']}"
        )

    def _get_local_name(self, iri: str) -> Sequence[str]:
        """
        Split IRI into prefix and local
        """
        if "#" in iri:
            tokens = iri.split("#")
            return [f"{tokens[0]}#", tokens[-1]]
        elif "/" in iri:
            tokens = iri.split("/")
            return [f"{'/'.join(tokens[0:len(tokens)-1])}/", tokens[-1]]
        else:
            raise ValueError(f"Unexpected IRI '{iri}', contains neither '#' nor '/'.")

    def _refresh_schema(self) -> None:
        """
        Query Neptune to introspect schema.
        """
        self.schema_elements["distinct_prefixes"] = {}

        for elem in ELEM_TYPES:
            items = self.query(ELEM_TYPES[elem])
            reslist = []
            for r in items["results"]["bindings"]:
                uri = r["elem"]["value"]
                tokens = self._get_local_name(uri)
                elem_record = {"uri": uri, "local": tokens[1]}
                if not self.hide_comments:
                    elem_record["comment"] = r["com"]["value"] if "com" in r else ""
                reslist.append(elem_record)
                if tokens[0] not in self.schema_elements["distinct_prefixes"]:
                    self.schema_elements["distinct_prefixes"][tokens[0]] = "y"

            self.schema_elements[elem] = reslist

        self.load_schema(self.schema_elements)

import json
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

import requests

# Query to find OWL datatype properties
DTPROP_QUERY = """
SELECT DISTINCT ?elem 
WHERE { 
 ?elem a owl:DatatypeProperty . 
}
"""

# Query to find OWL object properties
OPROP_QUERY = """
SELECT DISTINCT ?elem 
WHERE { 
 ?elem a owl:ObjectProperty . 
}
"""

ELEM_TYPES = {
    "classes": None,
    "rels": None,
    "dtprops": DTPROP_QUERY,
    "oprops": OPROP_QUERY,
}


class NeptuneRdfGraph:
    """Neptune wrapper for RDF graph operations.

    Args:
        host: endpoint for the database instance
        port: port number for the database instance, default is 8182
        use_iam_auth: boolean indicating IAM auth is enabled in Neptune cluster
        use_https: whether to use secure connection, default is True
        client: optional boto3 Neptune client
        credentials_profile_name: optional AWS profile name
        region_name: optional AWS region, e.g., us-west-2
        service: optional service name, default is neptunedata
        sign: optional, whether to sign the request payload, default is True

    Example:
        .. code-block:: python

        graph = NeptuneRdfGraph(
            host='<SPARQL host'>,
            port=<SPARQL port>
        )
        schema = graph.get_schema()

        OR
        graph = NeptuneRdfGraph(
            host='<SPARQL host'>,
            port=<SPARQL port>
        )
        schema_elem = graph.get_schema_elements()
        #... change schema_elements ...
        graph.load_schema(schema_elem)

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
        use_https: bool = True,
        use_iam_auth: bool = False,
        client: Any = None,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        service: str = "neptunedata",
        sign: bool = True,
    ) -> None:
        self.use_iam_auth = use_iam_auth
        self.region_name = region_name
        self.query_endpoint = f"https://{host}:{port}/sparql"

        try:
            if client is not None:
                self.client = client
            else:
                import boto3

                if credentials_profile_name is not None:
                    self.session = boto3.Session(profile_name=credentials_profile_name)
                else:
                    # use default credentials
                    self.session = boto3.Session()

                client_params = {}
                if region_name:
                    client_params["region_name"] = region_name

                protocol = "https" if use_https else "http"

                client_params["endpoint_url"] = f"{protocol}://{host}:{port}"

                if sign:
                    self.client = self.session.client(service, **client_params)
                else:
                    from botocore import UNSIGNED
                    from botocore.config import Config

                    self.client = self.session.client(
                        service,
                        **client_params,
                        config=Config(signature_version=UNSIGNED),
                    )

        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            if type(e).__name__ == "UnknownServiceError":
                raise ImportError(
                    "NeptuneGraph requires a boto3 version 1.28.38 or greater."
                    "Please install it with `pip install -U boto3`."
                ) from e
            else:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    "profile name are valid."
                ) from e

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

    def get_summary(self) -> Dict[str, Any]:
        """
        Obtain Neptune statistical summary of classes and predicates in the graph.
        """
        return self.client.get_rdf_graph_summary(mode="detailed")

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
            for elem_rec in schema_elements[elem]:
                uri = elem_rec["uri"]
                local = elem_rec["local"]
                res_str = f"<{uri}> ({local})"
                res_list.append(res_str)
            elem_str[elem] = ", ".join(res_list)

        self.schema = (
            "In the following, each IRI is followed by the local name and "
            "optionally its description in parentheses. \n"
            "The graph supports the following node types:\n"
            f"{elem_str['classes']}\n"
            "The graph supports the following relationships:\n"
            f"{elem_str['rels']}\n"
            "The graph supports the following OWL object properties:\n"
            f"{elem_str['dtprops']}\n"
            "The graph supports the following OWL data properties:\n"
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

        # get summary and build list of classes and rels
        summary = self.get_summary()
        reslist = []
        for c in summary["payload"]["graphSummary"]["classes"]:
            uri = c
            tokens = self._get_local_name(uri)
            elem_record = {"uri": uri, "local": tokens[1]}
            reslist.append(elem_record)
            if tokens[0] not in self.schema_elements["distinct_prefixes"]:
                self.schema_elements["distinct_prefixes"][tokens[0]] = "y"
        self.schema_elements["classes"] = reslist

        reslist = []
        for r in summary["payload"]["graphSummary"]["predicates"]:
            for p in r:
                uri = p
                tokens = self._get_local_name(uri)
                elem_record = {"uri": uri, "local": tokens[1]}
                reslist.append(elem_record)
                if tokens[0] not in self.schema_elements["distinct_prefixes"]:
                    self.schema_elements["distinct_prefixes"][tokens[0]] = "y"
        self.schema_elements["rels"] = reslist

        # get dtprops and oprops too
        for elem in ELEM_TYPES:
            q = ELEM_TYPES.get(elem)
            if not q:
                continue
            items = self.query(q)
            reslist = []
            for r in items["results"]["bindings"]:
                uri = r["elem"]["value"]
                tokens = self._get_local_name(uri)
                elem_record = {"uri": uri, "local": tokens[1]}
                reslist.append(elem_record)
                if tokens[0] not in self.schema_elements["distinct_prefixes"]:
                    self.schema_elements["distinct_prefixes"][tokens[0]] = "y"

            self.schema_elements[elem] = reslist

        self.load_schema(self.schema_elements)

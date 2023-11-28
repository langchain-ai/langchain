import json
from typing import Any, Dict, List, Optional

try:
    import pydgraph
except ImportError:
    raise ImportError(
        "Please install pydgraph with `pip install pydgraph` to use DGraph."
    )


class DGraph:
    """DGraph wrapper for graph operations.

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
        clientUrl: str,
        apiToken: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        self.clientUrl = clientUrl
        client_stub = None
        if apiToken is not None:
            client_stub = pydgraph.DgraphClientStub.from_cloud(clientUrl, apiToken)
        else:
            client_stub = pydgraph.DgraphClientStub(clientUrl)

        self.client = pydgraph.DgraphClient(client_stub)
        if username is not None:
            self.login_name_space(username, password, namespace)

    def get_schema(self) -> Dict[str, Any]:
        txn = self.client.txn(read_only=True)
        try:
            query = """schema {}"""
            res = txn.query(query)
            rawSchema = json.loads(res.json)
            reformattedSchema: Dict[str, Any] = self._reformat_schema(rawSchema)
            return reformattedSchema
        finally:
            txn.discard()

    def add_schema(self, schema_string: str) -> None:
        op = pydgraph.Operation(schema=schema_string)
        self.client.alter(op)

    def add_node(self, data: Dict[str, Any]) -> None:
        txn = self.client.txn()
        try:
            txn.mutate(set_obj=data)
            txn.commit()
        finally:
            txn.discard()

    def add_node_rdf(self, rdf_string: str) -> None:
        txn = self.client.txn()
        try:
            txn.mutate(set_nquads=rdf_string)
            txn.commit()
        finally:
            txn.discard()

    def drop_all(self) -> None:
        op = pydgraph.Operation(drop_all=True)
        self.client.alter(op)

    """
    Reformats a schema from the DGraph schema response to be of the form:
    {
      "type1": [
        {
          "predicate": "predicate1",
          "type": "uid",
          "list"?: true
          "upsert"?: true
          ...
        },
        {
          "predicate": "predicate2",
          "type": "string"
        }
      ],
      ...
    }
    returns a Dict containing the reformatted schema
    """

    def _reformat_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        predicateDict = {}  # Key is predicate name, value predicate info dict
        for predicate in schema["schema"]:
            predicateDict[predicate["predicate"]] = predicate
        reformattedSchema: Dict[str, Any] = {}
        for type in schema["types"]:
            reformattedSchema[type["name"]] = []
            for predicate in type["fields"]:
                reformattedSchema[type["name"]].append(predicateDict[predicate["name"]])
        return reformattedSchema

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query DGraph database"""
        txn = self.client.txn(read_only=True)
        try:
            res = txn.query(query, variables=params)
            return json.loads(res.json)
        finally:
            txn.discard()

    def validate_login_params(
        self, username: Optional[str], password: Optional[str], namespace: Optional[str]
    ) -> bool:
        if username is None:
            return False
        if password is None:
            return False
        if namespace is None:
            return False
        return True

    def login_name_space(
        self, username: Optional[str], password: Optional[str], namespace: Optional[str]
    ) -> None:
        if not self.validate_login_params(username, password, namespace):
            raise ValueError("Missing login parameters.")
        self.client.login_into_namespace(username, password, namespace)

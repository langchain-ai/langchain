from __future__ import annotations

import json
import re
from hashlib import md5
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Pattern, Tuple, Union

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore

if TYPE_CHECKING:
    import psycopg2.extras


class AGEQueryException(Exception):
    """Exception for the AGE queries."""

    def __init__(self, exception: Union[str, Dict]) -> None:
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


class AGEGraph(GraphStore):
    """
    Apache AGE wrapper for graph operations.

    Args:
        graph_name (str): the name of the graph to connect to or create
        conf (Dict[str, Any]): the pgsql connection config passed directly
            to psycopg2.connect
        create (bool): if True and graph doesn't exist, attempt to create it

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

    # python type mapping for providing readable types to LLM
    types = {
        "str": "STRING",
        "float": "DOUBLE",
        "int": "INTEGER",
        "list": "LIST",
        "dict": "MAP",
        "bool": "BOOLEAN",
    }

    # precompiled regex for checking chars in graph labels
    label_regex: Pattern = re.compile("[^0-9a-zA-Z]+")

    def __init__(
        self, graph_name: str, conf: Dict[str, Any], create: bool = True
    ) -> None:
        """Create a new AGEGraph instance."""

        self.graph_name = graph_name

        # check that psycopg2 is installed
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "Could not import psycopg2 python package. "
                "Please install it with `pip install psycopg2`."
            )

        self.connection = psycopg2.connect(**conf)

        with self._get_cursor() as curs:
            # check if graph with name graph_name exists
            graph_id_query = (
                """SELECT graphid FROM ag_catalog.ag_graph WHERE name = '{}'""".format(
                    graph_name
                )
            )

            curs.execute(graph_id_query)
            data = curs.fetchone()

            # if graph doesn't exist and create is True, create it
            if data is None:
                if create:
                    create_statement = """
                        SELECT ag_catalog.create_graph('{}');
                    """.format(graph_name)

                    try:
                        curs.execute(create_statement)
                        self.connection.commit()
                    except psycopg2.Error as e:
                        raise AGEQueryException(
                            {
                                "message": "Could not create the graph",
                                "detail": str(e),
                            }
                        )

                else:
                    raise Exception(
                        (
                            'Graph "{}" does not exist in the database '
                            + 'and "create" is set to False'
                        ).format(graph_name)
                    )

                curs.execute(graph_id_query)
                data = curs.fetchone()

            # store graph id and refresh the schema
            self.graphid = data.graphid
            self.refresh_schema()

    def _get_cursor(self) -> psycopg2.extras.NamedTupleCursor:
        """
        get cursor, load age extension and set search path
        """

        try:
            import psycopg2.extras
        except ImportError as e:
            raise ImportError(
                "Unable to import psycopg2, please install with "
                "`pip install -U psycopg2`."
            ) from e
        cursor = self.connection.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)
        cursor.execute("""LOAD 'age';""")
        cursor.execute("""SET search_path = ag_catalog, "$user", public;""")
        return cursor

    def _get_labels(self) -> Tuple[List[str], List[str]]:
        """
        Get all labels of a graph (for both edges and vertices)
        by querying the graph metadata table directly

        Returns
            Tuple[List[str]]: 2 lists, the first containing vertex
                labels and the second containing edge labels
        """

        e_labels_records = self.query(
            """MATCH ()-[e]-() RETURN collect(distinct label(e)) as labels"""
        )
        e_labels = e_labels_records[0]["labels"] if e_labels_records else []

        n_labels_records = self.query(
            """MATCH (n) RETURN collect(distinct label(n)) as labels"""
        )
        n_labels = n_labels_records[0]["labels"] if n_labels_records else []

        return n_labels, e_labels

    def _get_triples(self, e_labels: List[str]) -> List[Dict[str, str]]:
        """
        Get a set of distinct relationship types (as a list of dicts) in the graph
        to be used as context by an llm.

        Args:
            e_labels (List[str]): a list of edge labels to filter for

        Returns:
            List[Dict[str, str]]: relationships as a list of dicts in the format
                "{'start':<from_label>, 'type':<edge_label>, 'end':<from_label>}"
        """

        # age query to get distinct relationship types
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "Unable to import psycopg2, please install with "
                "`pip install -U psycopg2`."
            ) from e
        triple_query = """
        SELECT * FROM ag_catalog.cypher('{graph_name}', $$
            MATCH (a)-[e:`{e_label}`]->(b)
            WITH a,e,b LIMIT 3000
            RETURN DISTINCT labels(a) AS from, type(e) AS edge, labels(b) AS to
            LIMIT 10
        $$) AS (f agtype, edge agtype, t agtype);
        """

        triple_schema = []

        # iterate desired edge types and add distinct relationship types to result
        with self._get_cursor() as curs:
            for label in e_labels:
                q = triple_query.format(graph_name=self.graph_name, e_label=label)
                try:
                    curs.execute(q)
                    data = curs.fetchall()
                    for d in data:
                        # use json.loads to convert returned
                        # strings to python primitives
                        triple_schema.append(
                            {
                                "start": json.loads(d.f)[0],
                                "type": json.loads(d.edge),
                                "end": json.loads(d.t)[0],
                            }
                        )
                except psycopg2.Error as e:
                    raise AGEQueryException(
                        {
                            "message": "Error fetching triples",
                            "detail": str(e),
                        }
                    )

        return triple_schema

    def _get_triples_str(self, e_labels: List[str]) -> List[str]:
        """
        Get a set of distinct relationship types (as a list of strings) in the graph
        to be used as context by an llm.

        Args:
            e_labels (List[str]): a list of edge labels to filter for

        Returns:
            List[str]: relationships as a list of strings in the format
                "(:`<from_label>`)-[:`<edge_label>`]->(:`<to_label>`)"
        """

        triples = self._get_triples(e_labels)

        return self._format_triples(triples)

    @staticmethod
    def _format_triples(triples: List[Dict[str, str]]) -> List[str]:
        """
        Convert a list of relationships from dictionaries to formatted strings
        to be better readable by an llm

        Args:
            triples (List[Dict[str,str]]): a list relationships in the form
                {'start':<from_label>, 'type':<edge_label>, 'end':<from_label>}

        Returns:
            List[str]: a list of relationships in the form
                "(:`<from_label>`)-[:`<edge_label>`]->(:`<to_label>`)"
        """
        triple_template = "(:`{start}`)-[:`{type}`]->(:`{end}`)"
        triple_schema = [triple_template.format(**triple) for triple in triples]

        return triple_schema

    def _get_node_properties(self, n_labels: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch a list of available node properties by node label to be used
        as context for an llm

        Args:
            n_labels (List[str]): a list of node labels to filter for

        Returns:
            List[Dict[str, Any]]: a list of node labels and
                their corresponding properties in the form
                "{
                    'labels': <node_label>,
                    'properties': [
                        {
                            'property': <property_name>,
                            'type': <property_type>
                        },...
                        ]
                }"
        """
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "Unable to import psycopg2, please install with "
                "`pip install -U psycopg2`."
            ) from e

        # cypher query to fetch properties of a given label
        node_properties_query = """
        SELECT * FROM ag_catalog.cypher('{graph_name}', $$
            MATCH (a:`{n_label}`)
            RETURN properties(a) AS props
            LIMIT 100
        $$) AS (props agtype);
        """

        node_properties = []
        with self._get_cursor() as curs:
            for label in n_labels:
                q = node_properties_query.format(
                    graph_name=self.graph_name, n_label=label
                )

                try:
                    curs.execute(q)
                except psycopg2.Error as e:
                    raise AGEQueryException(
                        {
                            "message": "Error fetching node properties",
                            "detail": str(e),
                        }
                    )
                data = curs.fetchall()

                # build a set of distinct properties
                s = set({})
                for d in data:
                    # use json.loads to convert to python
                    # primitive and get readable type
                    for k, v in json.loads(d.props).items():
                        s.add((k, self.types[type(v).__name__]))

                np = {
                    "properties": [{"property": k, "type": v} for k, v in s],
                    "labels": label,
                }
                node_properties.append(np)

        return node_properties

    def _get_edge_properties(self, e_labels: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch a list of available edge properties by edge label to be used
        as context for an llm

        Args:
            e_labels (List[str]): a list of edge labels to filter for

        Returns:
            List[Dict[str, Any]]: a list of edge labels
                and their corresponding properties in the form
                "{
                    'labels': <edge_label>,
                    'properties': [
                        {
                            'property': <property_name>,
                            'type': <property_type>
                        },...
                        ]
                }"
        """

        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "Unable to import psycopg2, please install with "
                "`pip install -U psycopg2`."
            ) from e
        # cypher query to fetch properties of a given label
        edge_properties_query = """
        SELECT * FROM ag_catalog.cypher('{graph_name}', $$
            MATCH ()-[e:`{e_label}`]->()
            RETURN properties(e) AS props
            LIMIT 100
        $$) AS (props agtype);
        """
        edge_properties = []
        with self._get_cursor() as curs:
            for label in e_labels:
                q = edge_properties_query.format(
                    graph_name=self.graph_name, e_label=label
                )

                try:
                    curs.execute(q)
                except psycopg2.Error as e:
                    raise AGEQueryException(
                        {
                            "message": "Error fetching edge properties",
                            "detail": str(e),
                        }
                    )
                data = curs.fetchall()

                # build a set of distinct properties
                s = set({})
                for d in data:
                    # use json.loads to convert to python
                    # primitive and get readable type
                    for k, v in json.loads(d.props).items():
                        s.add((k, self.types[type(v).__name__]))

                np = {
                    "properties": [{"property": k, "type": v} for k, v in s],
                    "type": label,
                }
                edge_properties.append(np)

        return edge_properties

    def refresh_schema(self) -> None:
        """
        Refresh the graph schema information by updating the available
        labels, relationships, and properties
        """

        # fetch graph schema information
        n_labels, e_labels = self._get_labels()
        triple_schema = self._get_triples(e_labels)

        node_properties = self._get_node_properties(n_labels)
        edge_properties = self._get_edge_properties(e_labels)

        # update the formatted string representation
        self.schema = f"""
        Node properties are the following:
        {node_properties}
        Relationship properties are the following:
        {edge_properties}
        The relationships are the following:
        {self._format_triples(triple_schema)}
        """

        # update the dictionary representation
        self.structured_schema = {
            "node_props": {el["labels"]: el["properties"] for el in node_properties},
            "rel_props": {el["type"]: el["properties"] for el in edge_properties},
            "relationships": triple_schema,
            "metadata": {},
        }

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph"""
        return self.schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the structured schema of the Graph"""
        return self.structured_schema

    @staticmethod
    def _get_col_name(field: str, idx: int) -> str:
        """
        Convert a cypher return field to a pgsql select field
        If possible keep the cypher column name, but create a generic name if necessary

        Args:
            field (str): a return field from a cypher query to be formatted for pgsql
            idx (int): the position of the field in the return statement

        Returns:
            str: the field to be used in the pgsql select statement
        """
        # remove white space
        field = field.strip()
        # if an alias is provided for the field, use it
        if " as " in field:
            return field.split(" as ")[-1].strip()
        # if the return value is an unnamed primitive, give it a generic name
        elif field.isnumeric() or field in ("true", "false", "null"):
            return f"column_{idx}"
        # otherwise return the value stripping out some common special chars
        else:
            return field.replace("(", "_").replace(")", "")

    @staticmethod
    def _wrap_query(query: str, graph_name: str) -> str:
        """
        Convert a Cyper query to an Apache Age compatible Sql Query.
        Handles combined queries with UNION/EXCEPT operators

        Args:
            query (str) : A valid cypher query, can include UNION/EXCEPT operators
            graph_name (str) : The name of the graph to query

        Returns :
            str : An equivalent pgSql query wrapped with ag_catalog.cypher

        Raises:
            ValueError : If query is empty, contain RETURN *, or has invalid field names
        """

        if not query.strip():
            raise ValueError("Empty query provided")

        # pgsql template
        template = """SELECT {projection} FROM ag_catalog.cypher('{graph_name}', $$
            {query}
        $$) AS ({fields});"""

        # split the query into parts based on UNION and EXCEPT
        parts = re.split(r"\b(UNION\b|\bEXCEPT)\b", query, flags=re.IGNORECASE)

        all_fields = []

        for part in parts:
            if part.strip().upper() in ("UNION", "EXCEPT"):
                continue

            # if there are any returned fields they must be added to the pgsql query
            return_match = re.search(r'\breturn\b(?![^"]*")', part, re.IGNORECASE)
            if return_match:
                # Extract the part of the query after the RETURN keyword
                return_clause = part[return_match.end() :]

                # parse return statement to identify returned fields
                fields = (
                    return_clause.lower()
                    .split("distinct")[-1]
                    .split("order by")[0]
                    .split("skip")[0]
                    .split("limit")[0]
                    .split(",")
                )

                # raise exception if RETURN * is found as we can't resolve the fields
                clean_fileds = [f.strip() for f in fields if f.strip()]
                if "*" in clean_fileds:
                    raise ValueError(
                        "Apache Age does not support RETURN * in Cypher queries"
                    )

                # Format fields and maintain order of appearance
                for idx, field in enumerate(clean_fileds):
                    field_name = AGEGraph._get_col_name(field, idx)
                    if field_name not in all_fields:
                        all_fields.append(field_name)

        # if no return statements found in any part
        if not all_fields:
            fields_str = "a agtype"

        else:
            fields_str = ", ".join(f"{field} agtype" for field in all_fields)

        return template.format(
            graph_name=graph_name,
            query=query,
            fields=fields_str,
            projection="*",
        )

    @staticmethod
    def _record_to_dict(record: NamedTuple) -> Dict[str, Any]:
        """
        Convert a record returned from an age query to a dictionary

        Args:
            record (): a record from an age query result

        Returns:
            Dict[str, Any]: a dictionary representation of the record where
                the dictionary key is the field name and the value is the
                value converted to a python type
        """
        # result holder
        d = {}

        # prebuild a mapping of vertex_id to vertex mappings to be used
        # later to build edges
        vertices = {}
        for k in record._fields:
            v = getattr(record, k)
            # agtype comes back '{key: value}::type' which must be parsed
            if isinstance(v, str) and "::" in v:
                dtype = v.split("::")[-1]
                v = v.split("::")[0]
                if dtype == "vertex":
                    vertex = json.loads(v)
                    vertices[vertex["id"]] = vertex.get("properties")

        # iterate returned fields and parse appropriately
        for k in record._fields:
            v = getattr(record, k)
            if isinstance(v, str) and "::" in v:
                dtype = v.split("::")[-1]
                v = v.split("::")[0]
            else:
                dtype = ""

            if dtype == "vertex":
                d[k] = json.loads(v).get("properties")
            # convert edge from id-label->id by replacing id with node information
            # we only do this if the vertex was also returned in the query
            # this is an attempt to be consistent with neo4j implementation
            elif dtype == "edge":
                edge = json.loads(v)
                d[k] = (
                    vertices.get(edge["start_id"], {}),
                    edge["label"],
                    vertices.get(edge["end_id"], {}),
                )
            else:
                d[k] = json.loads(v) if isinstance(v, str) else v

        return d

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """
        Query the graph by taking a cypher query, converting it to an
        age compatible query, executing it and converting the result

        Args:
            query (str): a cypher query to be executed
            params (dict): parameters for the query (not used in this implementation)

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set
        """
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "Unable to import psycopg2, please install with "
                "`pip install -U psycopg2`."
            ) from e

        # convert cypher query to pgsql/age query
        wrapped_query = self._wrap_query(query, self.graph_name)

        # execute the query, rolling back on an error
        with self._get_cursor() as curs:
            try:
                curs.execute(wrapped_query)
                self.connection.commit()
            except psycopg2.Error as e:
                self.connection.rollback()
                raise AGEQueryException(
                    {
                        "message": "Error executing graph query: {}".format(query),
                        "detail": str(e),
                    }
                )

            data = curs.fetchall()
            if data is None:
                result = []
            # convert to dictionaries
            else:
                result = [self._record_to_dict(d) for d in data]

            return result

    @staticmethod
    def _format_properties(
        properties: Dict[str, Any], id: Union[str, None] = None
    ) -> str:
        """
        Convert a dictionary of properties to a string representation that
        can be used in a cypher query insert/merge statement.

        Args:
            properties (Dict[str,str]): a dictionary containing node/edge properties
            id (Union[str, None]): the id of the node or None if none exists

        Returns:
            str: the properties dictionary as a properly formatted string
        """
        props = []
        # wrap property key in backticks to escape
        for k, v in properties.items():
            prop = f"`{k}`: {json.dumps(v)}"
            props.append(prop)
        if id is not None and "id" not in properties:
            props.append(
                f"id: {json.dumps(id)}" if isinstance(id, str) else f"id: {id}"
            )
        return "{" + ", ".join(props) + "}"

    @staticmethod
    def clean_graph_labels(label: str) -> str:
        """
        remove any disallowed characters from a label and replace with '_'

        Args:
            label (str): the original label

        Returns:
            str: the sanitized version of the label
        """
        return re.sub(AGEGraph.label_regex, "_", label)

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """
        insert a list of graph documents into the graph

        Args:
            graph_documents (List[GraphDocument]): the list of documents to be inserted
            include_source (bool): if True add nodes for the sources
                with MENTIONS edges to the entities they mention

        Returns:
            None
        """
        # query for inserting nodes
        node_insert_query = (
            """
            MERGE (n:`{label}` {{`id`: "{id}"}})
            SET n = {properties}
            """
            if not include_source
            else """
            MERGE (n:`{label}` {properties})
            MERGE (d:Document {d_properties})
            MERGE (d)-[:MENTIONS]->(n)
        """
        )

        # query for inserting edges
        edge_insert_query = """
            MERGE (from:`{f_label}` {f_properties})
            MERGE (to:`{t_label}` {t_properties})
            MERGE (from)-[:`{r_label}` {r_properties}]->(to)
        """
        # iterate docs and insert them
        for doc in graph_documents:
            # if we are adding sources, create an id for the source
            if include_source:
                if not doc.source.metadata.get("id"):
                    doc.source.metadata["id"] = md5(
                        doc.source.page_content.encode("utf-8")
                    ).hexdigest()

            # insert entity nodes
            for node in doc.nodes:
                node.properties["id"] = node.id
                if include_source:
                    query = node_insert_query.format(
                        label=node.type,
                        properties=self._format_properties(node.properties),
                        d_properties=self._format_properties(doc.source.metadata),
                    )
                else:
                    query = node_insert_query.format(
                        label=AGEGraph.clean_graph_labels(node.type),
                        properties=self._format_properties(node.properties),
                        id=node.id,
                    )

                self.query(query)

            # insert relationships
            for edge in doc.relationships:
                edge.source.properties["id"] = edge.source.id
                edge.target.properties["id"] = edge.target.id
                inputs = {
                    "f_label": AGEGraph.clean_graph_labels(edge.source.type),
                    "f_properties": self._format_properties(edge.source.properties),
                    "t_label": AGEGraph.clean_graph_labels(edge.target.type),
                    "t_properties": self._format_properties(edge.target.properties),
                    "r_label": AGEGraph.clean_graph_labels(edge.type).upper(),
                    "r_properties": self._format_properties(edge.properties),
                }

                query = edge_insert_query.format(**inputs)
                self.query(query)

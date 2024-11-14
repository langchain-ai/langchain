import logging
import os
from hashlib import md5
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd
from oracledb import Connection, DatabaseError

if TYPE_CHECKING:
    from oracledb import Connection

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore

logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

reserved_words = {
    "ACCESS",
    "ADD",
    "ALL",
    "ALTER",
    "AND",
    "ANY",
    "AS",
    "ASC",
    "AUDIT",
    "BETWEEN",
    "BY",
    "CHAR",
    "CHECK",
    "CLUSTER",
    "COLUMN",
    "COLUMN_VALUE",
    "COMMENT",
    "COMPRESS",
    "CONNECT",
    "CREATE",
    "CURRENT",
    "DATE",
    "DECIMAL",
    "DEFAULT",
    "DELETE",
    "DESC",
    "DISTINCT",
    "DROP",
    "ELSE",
    "EXCLUSIVE",
    "EXISTS",
    "FILE",
    "FLOAT",
    "FOR",
    "FROM",
    "GRANT",
    "GROUP",
    "HAVING",
    "IDENTIFIED",
    "IMMEDIATE",
    "IN",
    "INCREMENT",
    "INDEX",
    "INITIAL",
    "INSERT",
    "INTEGER",
    "INTERSECT",
    "INTO",
    "IS",
    "LEVEL",
    "LIKE",
    "LOCK",
    "LONG",
    "MAXEXTENTS",
    "MINUS",
    "MLSLABEL",
    "MODE",
    "MODIFY",
    "NESTED_TABLE_ID",
    "NOAUDIT",
    "NOCOMPRESS",
    "NOT",
    "NOWAIT",
    "NULL",
    "NUMBER",
    "OF",
    "OFFLINE",
    "ON",
    "ONLINE",
    "OPTION",
    "OR",
    "ORDER",
    "PCTFREE",
    "PRIOR",
    "PUBLIC",
    "RAW",
    "RENAME",
    "RESOURCE",
    "REVOKE",
    "ROW",
    "ROWID",
    "ROWNUM",
    "ROWS",
    "SELECT",
    "SESSION",
    "SET",
    "SHARE",
    "SIZE",
    "SMALLINT",
    "START",
    "SUCCESSFUL",
    "SYNONYM",
    "SYSDATE",
    "TABLE",
    "THEN",
    "TO",
    "TRIGGER",
    "UID",
    "UNION",
    "UNIQUE",
    "UPDATE",
    "USER",
    "VALIDATE",
    "VALUES",
    "VARCHAR",
    "VARCHAR2",
    "VIEW",
    "WHENEVER",
    "WHERE",
    "WITH",
}


def _table_exists(client: Connection, table_name: str) -> bool:
    try:
        with client.cursor() as cursor:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                return True
            except Exception as e:
                logger.debug(e)
                return False
    except DatabaseError as ex:
        err_obj = ex.args
        if err_obj[0].code == 942:
            return False
        raise


def _drop_table_if_exists(client: Connection, table_name: str) -> None:
    table_name = table_name.upper()
    try:
        if _table_exists(client=client, table_name=table_name):
            try:
                _ddl_sql_query(client=client, ddl_query=f"DROP TABLE {table_name}")
                logger.debug(f"Success to Drop table {table_name}")
            except Exception as e:
                logger.error(f"Failed to Drop table {table_name}: {e}")
        else:
            logger.debug(f"{table_name} does not exist")
    except Exception as e:
        logger.error(f"Failed to check and drop table {table_name}: {e}")


def _drop_tables(
    client: Connection, relation_table_list: pd.DataFrame, node_table_list: pd.DataFrame
) -> None:
    relation_table_list["id"].apply(
        lambda table_name: _drop_table_if_exists(client, table_name)
    )
    node_table_list["id"].apply(
        lambda table_name: _drop_table_if_exists(client, table_name)
    )


def _graph_exists(client: Connection, graph_name: str) -> bool:
    graph_exists_query = f"""
        SELECT *
            FROM GRAPH_TABLE({graph_name}
            MATCH (n)
            COLUMNS (n.*))
        """
    try:
        with client.cursor() as cursor:
            try:
                cursor.execute(graph_exists_query)
                logger.debug("Graph exists")
                return True
            except Exception as e:
                logger.debug(f"Graph does not exist: {e}, {graph_exists_query}")
                return False
    except DatabaseError as ex:
        err_obj = ex.args
        if err_obj[0].code == 942:
            return False
        logger.error(f"Graph does not exist: {err_obj}")
        raise


def _remove_backticks(text: str | int) -> str:
    return str(text).replace("`", "")


def _ddl_sql_query(client: Connection, ddl_query: str) -> None:
    """DDL SQL query"""
    with client.cursor() as cursor:
        try:
            cursor.execute(ddl_query)
            logger.debug(f"Success DDL query: {ddl_query}")
        except Exception as e:
            logger.error(f"Failed to execute DDL query: {e}, {ddl_query}")


def _create_node_list(document: GraphDocument, doc_id: int | None) -> pd.DataFrame:
    node_list = pd.DataFrame(
        [
            (_clean_value(node.type), _clean_value(node.id), doc_id)
            for node in document.nodes
        ]
        + [
            (_clean_value(rel.source.type), _clean_value(rel.source.id), doc_id)
            for rel in document.relationships
        ]
        + [
            (_clean_value(rel.target.type), _clean_value(rel.target.id), doc_id)
            for rel in document.relationships
        ],
        columns=["id", "node_value", "doc_id"],
    )
    logger.debug(node_list)
    return node_list


def _create_relation_list(
    document: GraphDocument, node_list: pd.DataFrame
) -> pd.DataFrame:
    relation_list = pd.DataFrame(
        [
            {
                "id": _clean_value(rel.type)
                + ("_rel" if rel.type in node_list["id"].values else ""),
                "source": _clean_value(rel.source.type),
                "source_value": _clean_value(rel.source.id),
                "target": _clean_value(rel.target.type),
                "target_value": _clean_value(rel.target.id),
            }
            for rel in document.relationships
        ]
    )
    logger.debug(relation_list)

    tmp_relation_list = relation_list[["id", "source", "target"]].drop_duplicates(
        ignore_index=True
    )
    tmp_relation_list["id"] = tmp_relation_list["id"] + tmp_relation_list.groupby(
        ["id"]
    ).cumcount().astype(str).replace("0", "")

    for index, row in tmp_relation_list.iterrows():
        relation_list.loc[
            (relation_list["id"] == row["id"].rstrip("123456789"))
            & (relation_list["source"] == row["source"])
            & (relation_list["target"] == row["target"]),
            "id",
        ] = row["id"]
    return relation_list


def _create_document_tables(client: Connection) -> None:
    create_document_table_query = """
        CREATE TABLE document_table (
        doc_id NUMBER,
        document CLOB,
        CONSTRAINT doc_id_pk PRIMARY KEY (doc_id))
    """
    try:
        _ddl_sql_query(client=client, ddl_query=create_document_table_query)
        logger.debug("Created node table: document_table")
    except Exception as e:
        logger.error(f"Failed to create node table: {e}")


def _insert_document_data(
    client: Connection, doc_id: int | None, document_data: str
) -> None:
    tmp_insert_document_data_query = f"""
        INSERT INTO document_table (doc_id, document)
        VALUES ({doc_id}, '{document_data}')
        """
    try:
        with client.cursor() as cursor:
            cursor.execute(tmp_insert_document_data_query)
            logger.debug(f"Insert Document data: {tmp_insert_document_data_query}")
        client.commit()
    except Exception as e:
        logger.error(f"Failed to insert Document data: {e}")


def _create_node_tables(client: Connection, node_table_list: pd.DataFrame) -> None:
    for index, node in node_table_list.iterrows():
        node_table_name = node["id"].upper()
        pk = node_table_name.lower() + "_node_pk"

        create_node_table_query = f"""
            CREATE TABLE {node_table_name} (
            id VARCHAR2(100),
            doc_id NUMBER(10),
            CONSTRAINT {pk} PRIMARY KEY (id))
        """
        try:
            _ddl_sql_query(client=client, ddl_query=create_node_table_query)
            logger.debug(f"Created node table: {node_table_name}")
        except Exception as e:
            logger.error(f"Failed to create node table: {e}")


def _insert_node_data(client: Connection, node_list: pd.DataFrame) -> None:
    node_data_list = node_list.drop_duplicates(ignore_index=True)
    logger.debug(node_data_list)

    for index, node_data in node_data_list.iterrows():
        logger.debug(node_data)
        tmp_insert_node_data_query = f"""
            INSERT INTO {node_data["id"]} (id, doc_id)
            VALUES ('{node_data["node_value"]}', '{node_data["doc_id"]}')
            """
        try:
            with client.cursor() as cursor:
                cursor.execute(tmp_insert_node_data_query)
                logger.debug(tmp_insert_node_data_query)
                logger.debug(f"Insert Node data: {node_data['node_value']}")
            client.commit()
        except Exception as e:
            logger.error(f"Failed to insert node data: {e}")


def _create_relation_tables(
    client: Connection, relation_table_list: pd.DataFrame
) -> None:
    for index, relation in relation_table_list.iterrows():
        logger.debug(relation)
        relation_table_name = relation["id"].upper()
        pk = relation_table_name.lower() + "_relation_pk"
        source_fk = relation_table_name.lower() + "_source_fk"
        target_fk = relation_table_name.lower() + "_target_fk"

        create_relation_table_query = f"""
            CREATE TABLE {relation_table_name} (
            id NUMBER GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1),
            source VARCHAR2(100),
            target VARCHAR2(100),
            name VARCHAR2(100),
            CONSTRAINT {source_fk} FOREIGN KEY (source) 
            REFERENCES {relation["source"]}(id),
            CONSTRAINT {target_fk} FOREIGN KEY (target) 
            REFERENCES {relation["target"]}(id),
            CONSTRAINT {pk} PRIMARY KEY (id)
            )
        """

        try:
            _ddl_sql_query(client=client, ddl_query=create_relation_table_query)
            logger.debug(f"Created relation table: {relation_table_name}")
        except Exception as e:
            logger.error(f"Failed to create relation table: {e}")


def _insert_relation_data(client: Connection, relation_list: pd.DataFrame) -> None:
    relation_data_list = relation_list[
        ["id", "source_value", "target_value"]
    ].drop_duplicates(ignore_index=True)
    relation_data_list["name"] = relation_data_list["id"]
    logger.debug(relation_data_list)

    for index, relation_data in relation_data_list.iterrows():
        tmp_insert_relation_data_query = f"""
            INSERT INTO {relation_data["id"]} (source, target, name)
            VALUES (
                '{relation_data["source_value"]}',
                '{relation_data["target_value"]}',
                '{relation_data["name"]}'
                )
            """
        try:
            with client.cursor() as cursor:
                cursor.execute(tmp_insert_relation_data_query)
                logger.debug(tmp_insert_relation_data_query)
                logger.debug(
                    f"""Insert Relation data: 
                    {relation_data['source_value']},
                    {relation_data['target_value']}
                    """
                )
            client.commit()
        except Exception as e:
            logger.error(f"Failed to insert relation data: {e}")


def _create_property_graph_query(
    node_table_list: pd.DataFrame, relation_table_list: pd.DataFrame, graph_name: str
) -> str:
    create_graph_query = f"""
        CREATE PROPERTY GRAPH {graph_name}
            VERTEX TABLES (
        """

    for index, node in node_table_list.iterrows():
        vertex_query = f"""
            {node['id']} KEY (id)
            LABEL {node['id']}
        """
        create_graph_query = create_graph_query + vertex_query
        if index < len(node_table_list) - 1:
            create_graph_query += ", "
        else:
            create_graph_query += """)
            EDGE TABLES (
            """

    for index, relation in relation_table_list.iterrows():
        edge_query = f"""
            {relation['id']}
                KEY (id)
                SOURCE KEY (source) REFERENCES {relation['source']}(id)
                DESTINATION KEY (target) REFERENCES {relation['target']}(id)
                LABEL {relation['id']}
                PROPERTIES (name)
        """
        create_graph_query = create_graph_query + edge_query
        if index < len(relation_table_list) - 1:
            create_graph_query += ", "
        else:
            create_graph_query += """)
            OPTIONS (ALLOW MIXED PROPERTY TYPES)"""
    logger.debug(create_graph_query)
    return create_graph_query


def _create_graph(
    node_table_list: pd.DataFrame,
    relation_table_list: pd.DataFrame,
    graph_name: str,
    client: Connection,
) -> None:
    create_graph_query = _create_property_graph_query(
        node_table_list=node_table_list,
        relation_table_list=relation_table_list,
        graph_name=graph_name,
    )
    _ddl_sql_query(client=client, ddl_query=create_graph_query)


def _clean_value(value: str | int) -> str:
    val = str(value)
    result = _remove_backticks(val) + ("_t" if val.upper() in reserved_words else "")
    return result


class OracleGraph(GraphStore):
    """Oracle database wrapper for graph operations."""

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        refresh_schema: bool = True,
    ) -> None:
        """Create a new Oracle graph wrapper instance."""
        try:
            from opg4py import pgql
        except ImportError as e:
            raise ImportError(
                "Failed to import opg4py. Please install with "
                "`pip install -U oracle-graph-client-xxx`."
            ) from e
        self.schema: str = ""
        self.insert_mode = "array"
        self.structured_schema: Dict[str, Any] = {}

        try:
            self.connection = pgql.get_connection(
                usr=username,
                pwd=password,
                jdbc_url=url,
            )
        except Exception as e:
            logger.error(f"Failed to connect to Oracle database PGQL: {e}")

        # Set schema
        if refresh_schema:
            try:
                self.schema = str(self.connection.get_schema())
            except Exception as e:
                logger.error(f"Failed to set schema: {e}")

    @property
    def get_schema(
        self,
    ) -> str:
        """
        Get the Oracle graph schema information.
        """
        return self.schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the structured schema of the Graph"""
        return self.structured_schema

    def set_connection(self, client: Connection) -> None:
        """_summary_

        Args:
            client (Connection): The connection of Oracle Database
        """
        client.autocommit = True
        self.client = client

    def refresh_schema(self) -> None:
        """
        Refresh the Oracle graph schema information.
        """

        try:
            self.schema = self.connection.get_schema()
            logger.debug(f"Schema is {self.schema}")
        except Exception as e:
            logger.error(f"Failed to refresh schema: {e}")

    def get_graph(
        self,
    ) -> None:
        """Get the exist graph"""
        try:
            self.exist_graph = self.connection.get_graph()
            logger.debug(f"Exist Graph is {self.exist_graph}")
        except Exception as e:
            logger.error(f"Failed to get graph: {e}")

    def pgql_execute(self, query: str, param: dict = {}) -> None:
        """execute PGQL query

        Args:
            query (str): The PGQL query to execute.
            param (dict, optional): The parameters. Defaults to {}.
        """
        statement = self.connection.create_statement()
        try:
            statement.execute(pgql=query)
        except Exception as e:
            logger.error(f"Failed to execute: {e}")

    def pgql_query(self, query: str, param: dict = {}) -> List[Dict[str, Any]]:
        """PGQL Query Oracle Graph.

        Args:
            query (str): The PGQL query to execute.
            param (dict, optional): The parameters.

        Returns:
            List[Dict[str, Any]]: The list of dictionaries containing the query results.
        """

        statement = self.connection.create_statement()
        try:
            statement = statement.execute_query(pgql=query)
            results = statement.fetchall()
            if not results:
                return []
            else:
                json_data = [r for r in results]
                return json_data
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return []

    def query(self, query: str, param: dict = {}) -> List[Dict[str, Any]]:
        """SQL Execute Oracle Graph

        Args:
            client (Connection): Oracle Database connection.
            query (str): The SQL query to execute.
            param (dict, optional): The parameters to pass to the query. Defaults to {}.

        Returns:
            List[Dict[str,Any]: The list of dictionaries containing the query results.
        """

        try:
            with self.client.cursor() as cursor:
                cursor.execute(query, param)
                results = cursor.fetchall()
                if not results:
                    return []
                else:
                    data = [r for r in results]
                    return data
        except DatabaseError as e:
            logger.error(f"Failed to execute SQL query: {e}, {query}")
            return []
        except Exception as e:
            logger.error(f"Failed to execute query: {e}, {query}")
            return []

    def drop_graph(self, graph_name: str) -> None:
        """Drop the graph with the given name

        Args:
            graph_name (str): The name of the graph to drop.
        """
        try:
            if _graph_exists(client=self.client, graph_name=graph_name):
                self.pgql_execute(query=f"DROP PROPERTY GRAPH {graph_name}")
                logger.debug(f"Success to Drop Graph: {graph_name}")
            else:
                logger.debug(f"{graph_name} does not exist")
        except Exception as e:
            logger.error(f"Failed to drop graph: {e}")

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        graph_name: str = "",
    ) -> None:
        """This method constructs nodes and relationships in the graph based on the
        provided GraphDocument objects.

        Args:
            graph_documents (List[GraphDocument]): A list of GraphDocument objects
            that contain the nodes and relationships to be added to the graph. Each
            GraphDocument should encapsulate the structure of part of the graph,
            including nodes, relationships, and the source document information.
            include_source (bool, optional): _description_. Defaults to False.

        """
        self.graph_documents = graph_documents
        self.graph_name = graph_name
        doc_id = None
        if not include_source:
            doc_id = 1

        for document in self.graph_documents:
            if not document.source.metadata.get("id"):
                document.source.metadata["id"] = md5(
                    document.source.page_content.encode("utf-8")
                ).hexdigest()

        node_list = _create_node_list(document=document, doc_id=doc_id)
        relation_list = _create_relation_list(document=document, node_list=node_list)

        node_table_list = node_list[["id"]].drop_duplicates(ignore_index=True)
        logger.debug(node_table_list)

        relation_table_list = relation_list[["id", "source", "target"]].drop_duplicates(
            ignore_index=True
        )
        logger.debug(relation_table_list)

        self.get_graph()
        self.drop_graph(graph_name=graph_name)

        if not include_source:
            _drop_table_if_exists(client=self.client, table_name="document_table")
            _create_document_tables(self.client)
            _insert_document_data(
                client=self.client,
                doc_id=doc_id,
                document_data=document.source.page_content,
            )

        _drop_tables(
            client=self.client,
            relation_table_list=relation_table_list,
            node_table_list=node_table_list,
        )

        _create_node_tables(client=self.client, node_table_list=node_table_list)

        _insert_node_data(client=self.client, node_list=node_list)

        _create_relation_tables(
            client=self.client, relation_table_list=relation_table_list
        )
        _insert_relation_data(client=self.client, relation_list=relation_list)

        _create_graph(
            node_table_list=node_table_list,
            relation_table_list=relation_table_list,
            graph_name=self.graph_name,
            client=self.client,
        )

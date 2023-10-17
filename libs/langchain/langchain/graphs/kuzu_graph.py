from typing import Any, Dict, List


class KuzuGraph:
    """Kùzu wrapper for graph operations.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.
    """

    def __init__(self, db: Any, database: str = "kuzu") -> None:
        try:
            import kuzu
        except ImportError:
            raise ImportError(
                "Could not import Kùzu python package."
                "Please install Kùzu with `pip install kuzu`."
            )
        self.db = db
        self.conn = kuzu.Connection(self.db)
        self.database = database
        self.refresh_schema()

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Kùzu database"""
        return self.schema

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query Kùzu database"""
        params_list = []
        for param_name in params:
            params_list.append([param_name, params[param_name]])
        result = self.conn.execute(query, params_list)
        column_names = result.get_column_names()
        return_list = []
        while result.has_next():
            row = result.get_next()
            return_list.append(dict(zip(column_names, row)))
        return return_list

    def refresh_schema(self) -> None:
        """Refreshes the Kùzu graph schema information"""
        node_properties = []
        node_table_names = self.conn._get_node_table_names()
        for table_name in node_table_names:
            current_table_schema = {"properties": [], "label": table_name}
            properties = self.conn._get_node_property_names(table_name)
            for property_name in properties:
                property_type = properties[property_name]["type"]
                list_type_flag = ""
                if properties[property_name]["dimension"] > 0:
                    if "shape" in properties[property_name]:
                        for s in properties[property_name]["shape"]:
                            list_type_flag += "[%s]" % s
                    else:
                        for i in range(properties[property_name]["dimension"]):
                            list_type_flag += "[]"
                property_type += list_type_flag
                current_table_schema["properties"].append(
                    (property_name, property_type)
                )
            node_properties.append(current_table_schema)

        relationships = []
        rel_tables = self.conn._get_rel_table_names()
        for table in rel_tables:
            relationships.append(
                "(:%s)-[:%s]->(:%s)" % (table["src"], table["name"], table["dst"])
            )

        rel_properties = []
        for table in rel_tables:
            current_table_schema = {"properties": [], "label": table["name"]}
            properties_text = self.conn._connection.get_rel_property_names(
                table["name"]
            ).split("\n")
            for i, line in enumerate(properties_text):
                # The first 3 lines defines src, dst and name, so we skip them
                if i < 3:
                    continue
                if not line:
                    continue
                property_name, property_type = line.strip().split(" ")
                current_table_schema["properties"].append(
                    (property_name, property_type)
                )
            rel_properties.append(current_table_schema)

        self.schema = (
            f"Node properties: {node_properties}\n"
            f"Relationships properties: {rel_properties}\n"
            f"Relationships: {relationships}\n"
        )

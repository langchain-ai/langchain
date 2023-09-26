import re
from collections import namedtuple
from typing import Any, Dict, List, Match, Optional

Schema = namedtuple("Schema", ["left_node", "relation", "right_node"])


class CypherQueryCorrector:
    """
    Used to correct relationship direction in generated Cypher statements.
    This code is copied from the winner's submission to the Cypher competition:
    https://github.com/sakusaku-rich/cypher-direction-competition
    """

    property_pattern = re.compile(r"\{.+?\}")
    node_pattern = re.compile(r"\(.+?\)")
    path_pattern = re.compile(r"\(.*\).*-.*-.*\(.*\)")
    node_relation_node_pattern = re.compile(
        r"(\()+(?P<left_node>[^()]*?)\)(?P<relation>.*?)\((?P<right_node>[^()]*?)(\))+"
    )

    def __init__(self, schemas: List[Schema]):
        """
        Args:
            schemas: list of schemas
        """
        self.schemas = schemas

    def clean_node(self, node: str) -> str:
        """
        Args:
            node: node in string format

        """
        node = re.sub(self.property_pattern, "", node)
        node = node.replace(" ", "")
        node = node.replace("(", "")
        node = node.replace(")", "")
        return node

    def detect_node_variables(self, query: str) -> Dict[str, Any]:
        """
        Args:
            query: cypher query
        """
        nodes = re.findall(self.node_pattern, query)
        nodes = [self.clean_node(node) for node in nodes]
        res: Dict[str, Any] = {}
        for node in nodes:
            parts = node.split(":")
            parts = [p.replace("`", "") for p in parts]
            if parts == "":
                continue
            variable = parts[0]
            if variable not in res:
                res[variable] = []
            res[variable] += parts[1:]
        return res

    def extract_paths(self, query: str) -> "List[str]":
        """
        Args:
            query: cypher query
        """
        return re.findall(self.path_pattern, query)

    def judge_direction(self, relation: str) -> str:
        """
        Args:
            relation: relation in string format
        """
        direction = "BIDIRECTIONAL"
        if relation[0] == "<":
            direction = "INCOMING"
        if relation[-1] == ">":
            direction = "OUTGOING"
        return direction

    def extract_node_variable(self, part: str) -> Optional[str]:
        """
        Args:
            part: node in string format
        """
        part = part.lstrip("(").rstrip(")")
        idx = part.find(":")
        if idx != -1:
            part = part[:idx]
        return None if part == "" else part

    def detect_labels(
        self, str_node: str, node_variable_dict: Dict[str, Any]
    ) -> List[str]:
        """
        Args:
            str_node: node in string format
            node_variable_dict: dictionary of node variables
        """
        splitted_str = str_node.split(":")
        variable = splitted_str[0]
        labels = []
        if variable in node_variable_dict:
            labels = node_variable_dict[variable]
        elif variable == "" and len(splitted_str) > 1:
            labels = splitted_str[1:]
        return labels

    def verify_schema(
        self,
        from_node_labels: List[str],
        relation_type: Optional[Match[str]],
        to_node_labels: List[str],
    ) -> bool:
        """
        Args:
            from_node_labels: labels of the from node
            relation_type: type of the relation
            to_node_labels: labels of the to node
        """
        valid_schemas = self.schemas
        if from_node_labels != []:
            valid_schemas = [
                schema for schema in valid_schemas if schema[0] in from_node_labels
            ]
        if to_node_labels != []:
            valid_schemas = [
                schema for schema in valid_schemas if schema[2] in to_node_labels
            ]
        if relation_type is not None:
            valid_schemas = [
                schema for schema in valid_schemas if schema[1] == relation_type
            ]
        return valid_schemas != []

    def correct_query(self, query: str) -> str:
        """
        Args:
            query: cypher query
        """
        node_variable_dict = self.detect_node_variables(query)
        paths = self.extract_paths(query)
        for path in paths:
            original_path = path
            start_idx = 0
            while start_idx < len(path):
                match_res = re.match(self.node_relation_node_pattern, path[start_idx:])
                if match_res is None:
                    break
                start_idx += match_res.start()
                match_dict = match_res.groupdict()
                left_node_labels = self.detect_labels(
                    match_dict["left_node"], node_variable_dict
                )
                right_node_labels = self.detect_labels(
                    match_dict["right_node"], node_variable_dict
                )
                relation_direction = self.judge_direction(match_dict["relation"])
                matched_relation_type = re.search(
                    r":\w+\*?", match_dict["relation"].replace("`", "")
                )
                relation_type = (
                    matched_relation_type.group()[1:]
                    if matched_relation_type is not None
                    else None
                )
                end_idx = (
                    start_idx
                    + 4
                    + len(match_dict["left_node"])
                    + len(match_dict["relation"])
                    + len(match_dict["right_node"])
                )
                original_partial_path = original_path[start_idx : end_idx + 1]

                if relation_type is not None and relation_type[-1] == "*":
                    start_idx += (
                        len(match_dict["left_node"]) + len(match_dict["relation"]) + 2
                    )
                    continue

                if relation_direction == "OUTGOING":
                    is_legal = self.verify_schema(
                        left_node_labels, relation_type, right_node_labels
                    )
                    if not is_legal:
                        is_legal = self.verify_schema(
                            right_node_labels, relation_type, left_node_labels
                        )
                        if is_legal:
                            corrected_relation = "<" + match_dict["relation"][:-1]
                            corrected_partial_path = original_partial_path.replace(
                                match_dict["relation"], corrected_relation
                            )
                            query = query.replace(
                                original_partial_path, corrected_partial_path
                            )
                        else:
                            return ""
                elif relation_direction == "INCOMING":
                    is_legal = self.verify_schema(
                        right_node_labels, relation_type, left_node_labels
                    )
                    if not is_legal:
                        is_legal = self.verify_schema(
                            left_node_labels, relation_type, right_node_labels
                        )
                        if is_legal:
                            corrected_relation = match_dict["relation"][1:] + ">"
                            corrected_partial_path = original_partial_path.replace(
                                match_dict["relation"], corrected_relation
                            )
                            query = query.replace(
                                original_partial_path, corrected_partial_path
                            )
                        else:
                            return ""
                else:
                    is_legal = self.verify_schema(
                        left_node_labels, relation_type, right_node_labels
                    )
                    is_legal |= self.verify_schema(
                        right_node_labels, relation_type, left_node_labels
                    )
                    if not is_legal:
                        return ""

                start_idx += (
                    len(match_dict["left_node"]) + len(match_dict["relation"]) + 2
                )
        return query

    def __call__(self, query: str) -> str:
        """Correct the query to make it valid. If
        Args:
            query: cypher query
        """
        return self.correct_query(query)

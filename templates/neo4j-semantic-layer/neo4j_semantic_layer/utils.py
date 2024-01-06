from typing import Dict, List

from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph()


def get_user_id() -> int:
    return 1


def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = input.split()
    for word in words[:-1]:
        full_text_query += f" {word}~0.8 AND"
    full_text_query += f" {words[-1]}~0.8"
    return full_text_query.strip()


candidate_query = """
CALL db.index.fulltext.queryNodes($index, $fulltextQuery, {limit: $limit})
YIELD node
RETURN coalesce(node.name, node.title) AS candidate, [el in labels(node) WHERE el IN ['Person', 'Movie'] | el][0] AS label
"""


def get_candidates(input: str, type: str, limit: int = 3) -> List[Dict[str, str]]:
    ft_query = generate_full_text_query(input)
    candidates = graph.query(
        candidate_query, {"fulltextQuery": ft_query, "index": type, "limit": limit}
    )
    return candidates

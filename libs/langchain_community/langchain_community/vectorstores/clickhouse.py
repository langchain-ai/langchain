import json
from typing import List
from langchain_core.documents import Document


class Clickhouse:
    def __init__(self, connection, table_name: str):
        self.connection = connection
        self.table_name = table_name

    def query(self, query_embedding: List[float], k: int = 4) -> List[Document]:
        # Example query: select text, metadata, distance from table
        sql = (
            f"SELECT text, metadata, distance "
            f"FROM {self.table_name} "
            f"ORDER BY distance ASC "
            f"LIMIT {k}"
        )
        results = self.connection.execute(sql)
        documents = []
        for row in results:
            text = row[0]
            metadata_json = row[1]
            metadata = json.loads(metadata_json) if metadata_json else {}
            distance = row[2]
            metadata["distance"] = distance
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        return documents

    # ...other methods...

import json
from langchain_core.documents import Document
from clickhouse import Clickhouse


class DummyConnection:
    def execute(self, sql):
        # Simulate two rows with text, metadata (as JSON), and distance
        return [
            ("doc1 text", json.dumps({"source": "file1", "id": 1}), 0.12),
            ("doc2 text", json.dumps({"source": "file2", "id": 2}), 0.34),
        ]


def test_query_returns_metadata():
    conn = DummyConnection()
    store = Clickhouse(conn, "dummy_table")
    results = store.query([0.1, 0.2, 0.3], k=2)
    assert len(results) == 2
    assert isinstance(results[0], Document)
    assert results[0].metadata["source"] == "file1"
    assert results[1].metadata["id"] == 2
    print("Test passed: Metadata is returned correctly.")


if __name__ == "__main__":
    test_query_returns_metadata()

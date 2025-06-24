import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))
from examples.rag_webapp import ingest


def test_ingest_and_query(tmp_path: Path, monkeypatch):
    source = tmp_path / "docs"
    source.mkdir()
    (source / "test.md").write_text("Hello world")

    db_dir = tmp_path / "db"
    ingest.ingest(str(source), str(db_dir), store_type="chroma", embedding_model="fake")

    monkeypatch.setenv("PERSIST_DIR", str(db_dir))
    monkeypatch.setenv("VECTOR_STORE_TYPE", "chroma")
    monkeypatch.setenv("EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "dummy")
    monkeypatch.setenv("USE_FAKE_LLM", "true")
    monkeypatch.setenv("API_TOKEN", "secret")

    from examples.rag_webapp import app as web_app

    client = TestClient(web_app.app)
    resp = client.post(
        "/query",
        json={"question": "Hello?"},
        headers={"X-API-Token": "secret"},
    )
    assert resp.status_code == 200
    assert "answer" in resp.json()


def qdrant_is_not_running() -> bool:
    """Check if Qdrant is not running."""
    import requests

    try:
        response = requests.get("http://localhost:6333", timeout=10.0)
        response_json = response.json()
        return response_json.get("title") != "qdrant - vector search engine"
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return True

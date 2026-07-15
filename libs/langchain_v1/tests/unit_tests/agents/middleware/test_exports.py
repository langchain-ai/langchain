from langchain.agents.middleware import PIIMatch

def test_pii_match_export() -> None:
    """Verify that PIIMatch is exported and behaves like a TypedDict."""
    # Check that PIIMatch is a class type (TypedDict)
    assert isinstance(PIIMatch, type)
    
    # Verify we can construct a PIIMatch dict and it adheres to the type keys
    match: PIIMatch = {
        "type": "email",
        "value": "test@example.com",
        "start": 0,
        "end": 16,
    }
    
    assert match["type"] == "email"
    assert match["value"] == "test@example.com"
    assert match["start"] == 0
    assert match["end"] == 16

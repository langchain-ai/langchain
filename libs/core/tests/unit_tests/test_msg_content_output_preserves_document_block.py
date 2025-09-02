def test_msg_content_output_preserves_document_block():
    """Document blocks should pass through unchanged when recognized."""
    from langgraph.schema.message import msg_content_output
    doc_block = [{"type": "document", "content": [{"text": "Hello"}]}]
    result = msg_content_output(doc_block)
    assert result == doc_block


def test_msg_content_output_falls_back_to_json_for_unknown_block():
    """Unknown block types should be JSON serialized for backward compatibility."""
    from langgraph.schema.message import msg_content_output
    unknown_block = [{"type": "foo_block", "content": [{"text": "bar"}]}]
    result = msg_content_output(unknown_block)
    # Should be serialized to a JSON string
    import json
    assert result == json.dumps(unknown_block, ensure_ascii=False)

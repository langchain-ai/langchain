"""Test ChatPromptTemplate save method."""

import json
import tempfile
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate


def test_chat_prompt_template_save() -> None:
    """Test that ChatPromptTemplate can be saved to a file."""
    # Create a simple chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}")
    ])
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)
    
    try:
        # This should not raise NotImplementedError anymore
        prompt.save(temp_path)
        
        # Verify the file was created and contains valid JSON
        assert temp_path.exists()
        
        with open(temp_path, "r") as f:
            data = json.load(f)
            
        # Verify basic structure
        assert "_type" in data
        assert data["_type"] == "chat"
        
    finally:
        # Clean up
        if temp_path.exists():
            temp_path.unlink()

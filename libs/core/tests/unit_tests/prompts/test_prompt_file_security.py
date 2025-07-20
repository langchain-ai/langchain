import tempfile
import os
import pytest
from pathlib import Path

from langchain_core.prompts.prompt import PromptTemplate

def test_from_file_path_traversal_blocked(tmp_path):
    # Create a prompt file outside the base directory
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        f.write('{"template": "Hi {name}", "input_variables": ["name"]}')
        file_path = Path(f.name)

    try:
        # Try loading the template from outside allowed base directory
        with pytest.raises(ValueError, match="Resolved path is outside of the allowed directory"):
            PromptTemplate.from_file(template_file=file_path)
    finally:
        os.remove(file_path)  # Clean up

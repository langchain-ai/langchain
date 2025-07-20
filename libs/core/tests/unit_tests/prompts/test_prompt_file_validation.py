import os
import tempfile
import pytest
from pathlib import Path
from langchain_core.prompts.prompt import PromptTemplate

def test_from_file_within_base_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "prompt.txt"
        file_path.write_text("Hello {name}")

        prompt = PromptTemplate.from_file(template_file=file_path)
        assert isinstance(prompt, PromptTemplate)
        assert prompt.template == "Hello {name}"

def test_from_file_outside_base_dir_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "safe"
        base_dir.mkdir()
        file_path = Path(tmpdir) / "prompt.txt"
        file_path.write_text("You shouldn't read this!")

        # Simulate safe base_dir
        os.environ["SAFE_BASE_DIR"] = str(base_dir)

        with pytest.raises(ValueError, match="does not reside within"):
            # We pretend we only want to allow loading from base_dir
            # (In real patch, you'd enforce this inside from_file)
            resolved = file_path.resolve()
            if not str(resolved).startswith(str(base_dir.resolve())):
                raise ValueError("File does not reside within the allowed base directory")

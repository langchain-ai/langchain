from pathlib import Path
from typing import Dict

from langchain_community.document_loaders import JSONLoader


def call_back(sample: Dict, additional_fields: Dict) -> Dict:
    metadata = additional_fields.copy()
    metadata["source"] += f"#seq_num={metadata['seq_num']}"
    return metadata


def test_json_loader() -> None:
    """Test unstructured loader."""
    file_path = Path(__file__).parent.parent / "examples/example.json"

    loader = JSONLoader(file_path, ".messages[].content", metadata_func=call_back)
    docs = loader.load()

    # Check that the correct number of documents are loaded.
    assert len(docs) == 3

    # Make sure that None content are converted to empty strings.
    assert docs[-1].page_content == ""

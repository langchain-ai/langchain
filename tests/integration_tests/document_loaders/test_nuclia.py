from typing import Any
from unittest import mock

from langchain.document_loaders.nuclia import NucliaLoader
from langchain.tools.nuclia.tool import NucliaUnderstandingAPI


def fakerun(**args: Any) -> Any:
    def run(self: Any, **args: Any) -> str:
        return '{"extracted_text": [{"body": {"text": "Hello World"}}], "file_extracted_data": [{"language": "en"}], "field_metadata": [{"metadata": {"metadata": {"paragraphs": [{ "end": 66, "sentences": [{ "start": 1, "end": 67 }] }]}}}]}'

    return run


def test_nuclia_loader() -> None:
    with mock.patch(
        "langchain.tools.nuclia.tool.NucliaUnderstandingAPI._run", new_callable=fakerun
    ):
        nua = NucliaUnderstandingAPI()
        loader = NucliaLoader("/whatever/file.mp3", nua)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].page_content == "Hello World"
        assert docs[0].metadata["file"]["language"] == "en"
        assert (
            len(docs[0].metadata["metadata"]["metadata"]["metadata"]["paragraphs"]) == 1
        )

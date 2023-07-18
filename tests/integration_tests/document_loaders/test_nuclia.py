import json
import os
from typing import Any
from unittest import mock

from langchain.document_loaders.nuclia import NucliaLoader
from langchain.tools.nuclia.tool import NucliaUnderstandingAPI


def fakerun(**args: Any) -> Any:
    def run(self: Any, **args: Any) -> str:
        data = {
            "extracted_text": [{"body": {"text": "Hello World"}}],
            "file_extracted_data": [{"language": "en"}],
            "field_metadata": [
                {
                    "metadata": {
                        "metadata": {
                            "paragraphs": [
                                {"end": 66, "sentences": [{"start": 1, "end": 67}]}
                            ]
                        }
                    }
                }
            ],
        }
        return json.dumps(data)

    return run


@mock.patch.dict(os.environ, {"NUCLIA_NUA_KEY": "_a_key_"})
def test_nuclia_loader() -> None:
    with mock.patch(
        "langchain.tools.nuclia.tool.NucliaUnderstandingAPI._run", new_callable=fakerun
    ):
        nua = NucliaUnderstandingAPI(enable_ml=False)
        loader = NucliaLoader("/whatever/file.mp3", nua)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].page_content == "Hello World"
        assert docs[0].metadata["file"]["language"] == "en"
        assert (
            len(docs[0].metadata["metadata"]["metadata"]["metadata"]["paragraphs"]) == 1
        )

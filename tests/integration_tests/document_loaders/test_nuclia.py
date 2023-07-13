from langchain.document_loaders.nuclia import NucliaLoader

class MockNucliaUnderstandingAPI:
    def run(self, data):
        return '{"extracted_text": [{"body": {"text": "Hello World"}}], "file_extracted_data": [{"language": "en"}], "field_metadata": [{"metadata": {"metadata": {"paragraphs": [{ "end": 66, "sentences": [{ "start": 1, "end": 67 }] }]}}}]}'


def test_nuclia_loader() -> None:
    nua = MockNucliaUnderstandingAPI()
    loader = NucliaLoader("/whatever/file.mp3", nua)
    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].page_content == "Hello World"
    assert docs[0].metadata["file"]["language"] == "en"
    assert len(docs[0].metadata["metadata"]["metadata"]["metadata"]["paragraphs"]) == 1
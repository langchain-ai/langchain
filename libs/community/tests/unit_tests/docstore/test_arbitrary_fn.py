from langchain_core.documents import Document

from langchain_community.docstore.arbitrary_fn import DocstoreFn


def test_document_found() -> None:
    # we use a dict here for simiplicity, but this could be any function
    # including a remote lookup
    dummy_dict = {"foo": Document(page_content="bar")}
    docstore = DocstoreFn(lambda x: dummy_dict[x])
    output = docstore.search("foo")
    assert isinstance(output, Document)
    assert output.page_content == "bar"

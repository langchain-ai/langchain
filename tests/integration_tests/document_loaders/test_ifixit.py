from langchain.document_loaders.ifixit import IFixitLoader


def test_ifixit_loader() -> None:
    """Test iFixit loader."""
    web_path = "https://www.ifixit.com/Guide/iPad+9+Battery+Replacement/151279"
    loader = IFixitLoader(web_path)
    assert loader.page_type == "Guide"
    assert loader.id == "151279"
    assert loader.web_path == web_path


def test_ifixit_loader_teardown() -> None:
    web_path = "https://www.ifixit.com/Teardown/Banana+Teardown/811"
    loader = IFixitLoader(web_path)
    """ Teardowns are just guides by a different name """
    assert loader.page_type == "Guide"
    assert loader.id == "811"


def test_ifixit_loader_device() -> None:
    web_path = "https://www.ifixit.com/Device/Standard_iPad"
    loader = IFixitLoader(web_path)
    """ Teardowns are just guides by a different name """
    assert loader.page_type == "Device"
    assert loader.id == "Standard_iPad"


def test_ifixit_loader_answers() -> None:
    web_path = (
        "https://www.ifixit.com/Answers/View/318583/My+iPhone+6+is+typing+and+"
        "opening+apps+by+itself"
    )
    loader = IFixitLoader(web_path)

    assert loader.page_type == "Answers"
    assert loader.id == "318583"

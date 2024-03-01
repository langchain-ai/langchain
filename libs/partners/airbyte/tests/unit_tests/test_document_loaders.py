from langchain_core.prompts import PromptTemplate

from langchain_airbyte import AirbyteLoader


def test_initialization() -> None:
    """Test integration loader initialization."""
    AirbyteLoader(
        source="source-faker",
        stream="users",
        config={"count": 3},
    )


def test_load() -> None:
    """Test loading from source."""
    airbyte_loader = AirbyteLoader(
        source="source-faker",
        stream="users",
        config={"count": 5},
    )
    documents = airbyte_loader.load()
    assert len(documents) == 5


def test_lazy_load() -> None:
    """Test lazy loading from source."""
    airbyte_loader = AirbyteLoader(
        source="source-faker",
        stream="users",
        config={"count": 3},
    )
    documents = airbyte_loader.lazy_load()
    assert len(list(documents)) == 3


async def test_alazy_load() -> None:
    """Test async lazy loading from source."""
    airbyte_loader = AirbyteLoader(
        source="source-faker",
        stream="users",
        config={"count": 3},
    )
    documents = airbyte_loader.alazy_load()
    lendocs = 0
    async for _ in documents:
        lendocs += 1
    assert lendocs == 3


def test_load_with_template() -> None:
    """Test loading from source with template."""
    airbyte_loader = AirbyteLoader(
        source="source-faker",
        stream="users",
        config={"count": 3},
        template=PromptTemplate.from_template("My name is {name}"),
    )
    documents = airbyte_loader.load()
    assert len(documents) == 3
    for doc in documents:
        assert doc.page_content.startswith("My name is ")
        assert doc.metadata["name"]  # should have a name


def test_load_no_metadata() -> None:
    """Test loading from source with no metadata."""
    airbyte_loader = AirbyteLoader(
        source="source-faker",
        stream="users",
        config={"count": 3},
        include_metadata=False,
    )
    documents = airbyte_loader.load()
    assert len(documents) == 3
    for doc in documents:
        assert doc.metadata == {}

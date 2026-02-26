import datetime
import uuid
from unittest.mock import MagicMock, patch

from langsmith.schemas import Example

from langchain_core.document_loaders import LangSmithLoader
from langchain_core.documents import Document


def test_init() -> None:
    LangSmithLoader(api_key="secret")


EXAMPLES = [
    Example(
        inputs={"first": {"second": "foo"}},
        outputs={"res": "a"},
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    ),
    Example(
        inputs={"first": {"second": "bar"}},
        outputs={"res": "b"},
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    ),
    Example(
        inputs={"first": {"second": "baz"}},
        outputs={"res": "c"},
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    ),
]


@patch("langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES)))
def test_lazy_load() -> None:
    loader = LangSmithLoader(
        api_key="dummy",
        dataset_id="mock",
        content_key="first.second",
        format_content=(lambda x: x.upper()),
    )
    expected = []
    for example in EXAMPLES:
        metadata = {
            k: v if not v or isinstance(v, dict) else str(v)
            for k, v in example.dict().items()
        }
        expected.append(
            Document(example.inputs["first"]["second"].upper(), metadata=metadata)
            if example.inputs
            else None
        )
    actual = list(loader.lazy_load())
    assert expected == actual

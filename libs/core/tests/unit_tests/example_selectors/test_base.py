from typing import Dict, List, Optional

from langchain_core.example_selectors import BaseExampleSelector


class DummyExampleSelector(BaseExampleSelector):
    def __init__(self) -> None:
        self.example: Optional[Dict[str, str]] = None

    def add_example(self, example: Dict[str, str]) -> None:
        self.example = example

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        return [input_variables]


async def test_aadd_example() -> None:
    selector = DummyExampleSelector()
    await selector.aadd_example({"foo": "bar"})
    assert selector.example == {"foo": "bar"}


async def test_aselect_examples() -> None:
    selector = DummyExampleSelector()
    examples = await selector.aselect_examples({"foo": "bar"})
    assert examples == [{"foo": "bar"}]

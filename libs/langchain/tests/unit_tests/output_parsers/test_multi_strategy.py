from pathlib import Path
from typing import Any, List, Tuple

import pytest

from langchain.output_parsers.multi_strategy import strategies
from langchain.output_parsers.multi_strategy.agent import ConvMultiStrategyParser
from langchain.output_parsers.multi_strategy.base import MultiStrategyParser

# How the test works:
# it loads all llm output files from the ../data/llm_outputs directory
# For each file it tries a MultiStrategyParser with the strategies to test.


def prepare_outputs() -> List[Tuple[str, str]]:
    outputs = []
    for path in (Path(__file__).parent.parent / "data/llm_outputs/").glob("*"):
        with open(str(path), "r") as f:
            outputs.append((f.read(), path.name))
    return outputs


llm_outputs = prepare_outputs()


@pytest.mark.parametrize("output, name", llm_outputs, ids=[x[1] for x in llm_outputs])
def test_json_react_strategies(
    output: str, name: str, parser: MultiStrategyParser[Any, Any]
) -> None:
    # the ignored test is for the fallback strategy
    if name != "ignored_format_instructions":
        _test_json_react_strategy(output, name, parser)


def _test_json_react_strategy(
    output: str, name: str, parser: MultiStrategyParser[Any, Any]
) -> None:
    try:
        parser.parse(output)
    except Exception:
        pytest.fail(f"Error parsing output entry: {name}.")


def test_fix_json_with_embedded_code_block() -> None:
    path = Path(__file__).parent.parent / "data/llm_outputs/bare_json_embed_code_block"
    with open(str(path), "r") as f:
        output = f.read()
    res = strategies.fix_json_with_embedded_code_block(output)
    assert type(res) == dict
    with pytest.raises(Exception):
        res = strategies.fix_json_with_embedded_code_block(output, max_loop=1)


@pytest.fixture(name="parser")
def conv_multi_strategy_parser() -> Any:
    return ConvMultiStrategyParser(strategies.json_react_strategies)

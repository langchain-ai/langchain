"""Test the run collector."""

import uuid

from langchain.callbacks import collect_runs
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_collect_runs() -> None:
    llm = FakeLLM(queries={"hi": "hello"}, sequential_responses=True)
    with collect_runs() as cb:
        llm.predict("hi")
        assert cb.traced_runs
        assert len(cb.traced_runs) == 1
        assert isinstance(cb.traced_runs[0].id, uuid.UUID)
        assert cb.traced_runs[0].inputs == {"prompts": ["hi"]}

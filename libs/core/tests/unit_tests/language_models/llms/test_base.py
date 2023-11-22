from tests.unit_tests.fake.llm import FakeListLLM


def test_batch() -> None:
    llm = FakeListLLM(responses=["foo"] * 3)
    output = llm.batch(["foo", "bar", "foo"])
    assert output == ["foo"] * 3

    output = llm.batch(["foo", "bar", "foo"], config={"max_concurrency": 2})
    assert output == ["foo"] * 3


async def test_abatch() -> None:
    llm = FakeListLLM(responses=["foo"] * 3)
    output = await llm.abatch(["foo", "bar", "foo"])
    assert output == ["foo"] * 3

    output = await llm.abatch(["foo", "bar", "foo"], config={"max_concurrency": 2})
    assert output == ["foo"] * 3

import inspect
from langchain.concise.rulex import RulEx, Rule
from langchain.llms.fake import FakeListLLM


def test_rulex():

    rules = [
        Rule(
            name="add comments",
            pattern="a block of code without comments",
            replacement="that same block of code with comments added",
        ),
        Rule(
            name="fix typos",
            pattern="a misspelled word",
            replacement="the correct spelling of that word",
        ),
        Rule(
            name="add doctests",
            pattern="a function without doctests",
            replacement="that same function with doctests added",
        ),
        Rule(
            name="decompose large functions",
            pattern="a function that is too long",
            replacement="that same function decomposed into smaller functions (make sure to implement the smaller functions too!)",
        ),
        Rule(
            name="add type hints",
            pattern="a function without type hints",
            replacement="that same function with type hints added",
        ),
    ]
    llm = FakeListLLM(64 * ["[mocked]"])
    ru: RulEx = RulEx.create(rules, llm=llm)

    input = inspect.getsource(test_rulex)
    output = ru(input)
    assert isinstance(output, str)

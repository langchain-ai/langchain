from langchain_core.runnables.fact_checker import RunnableFactChecker


class DummyLLM:
    def __init__(self, responses):
        self.responses = responses
        self.index = 0

    def invoke(self, prompt):
        class Response:
            def __init__(self, content):
                self.content = content

        result = self.responses[self.index]
        self.index += 1
        return Response(result)


def test_fact_checker_all_true():
    llm = DummyLLM(["TRUE", "TRUE"])
    checker = RunnableFactChecker(llm=llm)

    result = checker.invoke({
        "context": [],
        "answer": "Sky is blue. Grass is green."
    })

    assert result.response_metadata["confidence_score"] == 1.0


def test_fact_checker_mixed():
    llm = DummyLLM(["TRUE", "FALSE"])
    checker = RunnableFactChecker(llm=llm)

    result = checker.invoke({
        "context": [],
        "answer": "Sky is blue. Sky is red."
    })

    assert result.response_metadata["confidence_score"] == 0.5


def test_fact_checker_strict():
    llm = DummyLLM(["TRUE", "FALSE"])
    checker = RunnableFactChecker(llm=llm, strict_mode=True)

    result = checker.invoke({
        "context": [],
        "answer": "Sky is blue. Sky is red."
    })

    assert "Sky is red." not in result.content

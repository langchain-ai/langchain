from langchain.concise.pattern import pattern
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.schema import BaseLanguageModel


def decide(
    input: str,
    query: str = None,
    true_examples: list[str] = [],
    false_examples: list[str] = [],
    examples: list[tuple[str, bool]] = [],
    llm: BaseLanguageModel = None,
) -> bool:
    """Decide whether a statement is true or false based on the input and query.

    Args:
        input (str): The statement to evaluate.
        query (str): The query that determines what to evaluate.
        true_examples (list[str], optional): A list of example statements that are true. Defaults to [].
        false_examples (list[str], optional): A list of example statements that are false. Defaults to [].
        examples (list[tuple[str, bool]], optional): A list of example statements and their corresponding boolean outputs. If examples are provided in true_examples or false_examples, they will be automatically included in the examples list. Defaults to [].
        llm (BaseLanguageModel, optional): Language model to override the default LLM. Defaults to None.

    Returns:
        bool: The boolean value of the evaluated statement.
    """
    parser = BooleanOutputParser()
    examples.extend([(ex, True) for ex in true_examples])
    examples.extend([(ex, False) for ex in false_examples])
    # Rephrase the outputs in the vernacular of the boolean parser.
    examples = [
        (input, parser.true_val if output else parser.false_val)
        for input, output in examples
    ]

    return pattern(
        input=input,
        query=query,
        pattern_name="decide",
        parser=parser,
        examples=examples,
        llm=llm,
    )

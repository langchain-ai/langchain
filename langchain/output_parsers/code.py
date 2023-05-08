import re

from langchain.schema import BaseOutputParser, OutputParserException


class CodeOutputParser(BaseOutputParser[str]):
    """Parses the first triple-backticked enclosed code block in the resposne.

    This allows the LM to think out loud before beginning its answer.

    Note that code blocks often exceed the response length limit, so you may need to chain
    this with a `StitchedOutputParser` to intelligently piece together the output blocks. Like so:
        code_parser = CodeOutputParser()
        is_complete_prompt = generate(f"Generate a string prompt that asks whether a piece of code is a complete and correct implementation of the following instructions: {instructions}")
        def is_complete(code: str) -> bool:
            return decide(
                prompt=dedent(
                    f'''
                    Instructions:
                    {is_complete_prompt}. Do not disqualify an output for having ellipses, as these merely indicate intentional truncation. We're asking you to disqualify outputs that are incomplete or incorrect.

                    Input:
                    {code}
                    '''
                ).strip(),
            )
        stitched_output_parser = StitchedOutputParser.from_llm(
            completion_validator=is_complete,
            llm=llm,
        )
        multi_retry_with_error_parser = MultiAttemptRetryWithErrorOutputParser.from_llm(
            llm=llm, parser=ChainedOutputParser(
                parsers=[stitched_output_parser, code_parser]
            ),
        )
        return generate(
            messages, parser=multi_retry_with_error_parser
        )

    """

    def parse(self, text: str) -> str:
        # Use regular expressions to find the first code block
        code_match = re.search(r"```(?:\w+\n)?(.*?)```", text, re.DOTALL)
        # If no code block is found, raise an exception
        if not code_match:
            raise OutputParserException(
                "No code block found. Please write your code in a code block encloded with triple backticks (```)."
            )
        # Extract the code and return it
        code = code_match.group(1).strip()
        return code

    def get_format_instructions(self) -> str:
        return "Write the code. Do not include any other text in your answer."

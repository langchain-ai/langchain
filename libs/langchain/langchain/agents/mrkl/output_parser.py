from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS


class MRKLOutputParser(ReActSingleInputOutputParser):
    """MRKL Output parser for the chat agent."""

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    @property
    def _type(self) -> str:
        return "mrkl"

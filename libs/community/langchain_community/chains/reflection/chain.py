"""
ReflectionOutputTagExtractor Chain for extracting content within <output> tags from LLM responses.
"""

from typing import Any, Dict, List, Optional
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.schema import BaseOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
import re

class ReflectionOutputTagExtractor(BaseOutputParser):
    """Extracts content within <output> tags."""

    def parse(self, text: str) -> str:
        pattern = r'<output>(.*?)</output>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "No output found."

class ReflectionOutputTagExtractorChain(Chain):
    """Chain for extracting content within <output> tags from LLM responses."""

    llm_chain: LLMChain = Field(exclude=True)
    """LLMChain to use for generating responses."""
    output_parser: ReflectionOutputTagExtractor = Field(default_factory=ReflectionOutputTagExtractor)
    """Parser to extract content from <output> tags."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Output keys."""
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        question = inputs[self.input_key]
        response = self.llm_chain.run(question)
        extracted_output = self.output_parser.parse(response)
        return {self.output_key: extracted_output}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate,
        **kwargs: Any,
    ) -> "ReflectionOutputTagExtractorChain":
        """Create an ReflectionOutputTagExtractorChain from an LLM and prompt."""
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)

    @property
    def _chain_type(self) -> str:
        return "output_tag_extractor_chain"
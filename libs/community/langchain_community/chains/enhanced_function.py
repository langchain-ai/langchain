from typing import Any, Dict, List, Optional
from langchain.chains import LLMMathChain
from langchain.llms.base import BaseLLM
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from pydantic import Field

class EnhancedFunctionChain(Chain):
    """Chain that enhances function calls with math capabilities."""

    llm: BaseLLM = Field(description="The base language model to use.")
    math_chain: LLMMathChain = Field(default_factory=lambda: None)
    contains_math: bool = Field(default=False, description="Flag to indicate if math processing is required.")

    math_detection_prompt: PromptTemplate = PromptTemplate(
        input_variables=["query"],
        template="Determine if the following query requires mathematical calculations. Respond with 'Yes' or 'No'.\nQuery: {query}\nRequires math:"
    )

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return ["result"]

    def __init__(self, llm: BaseLLM, **kwargs: Any):
        """Initialize the chain."""
        super().__init__(**kwargs)
        self.llm = llm
        self.math_chain = LLMMathChain.from_llm(llm=self.llm)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Process the input query."""
        query = inputs["query"]

        if not self.contains_math:
            self.contains_math = self._detect_math(query)

        if self.contains_math:
            try:
                result = self.math_chain.run(query)
            except Exception as e:
                result = f"Error in math calculation: {str(e)}"
        else:
            result = self.llm(query)

        return {"result": result}

    def _detect_math(self, query: str) -> bool:
        """Detect if the query requires mathematical calculations."""
        response = self.llm(self.math_detection_prompt.format(query=query))
        return response.strip().lower() == "yes"

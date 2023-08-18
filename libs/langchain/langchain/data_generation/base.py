import asyncio
from typing import List, Optional

from pydantic.class_validators import root_validator
from pydantic.error_wrappers import ValidationError
from pydantic.main import BaseModel

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.schema.language_model import BaseLanguageModel


class SyntheticDataGenerator(BaseModel):
    """Generates synthetic data using the given LLM and few-shot template.

    Utilizes the provided LLM to produce synthetic data based on the
    few-shot prompt template. Optionally, it evaluates the fitness of the
    generated results using an evaluator function.

    Attributes:
        template (FewShotPromptTemplate): Template for few-shot prompting.
        runs (int): Number of runs for synthetic data generation.
        llm (LLM): Large Language Model to use for generation.
        llm_chain (LLMChain): LLM chain initialized with the LLM and few-shot template.
    """

    template: FewShotPromptTemplate
    llm: Optional[BaseLanguageModel] = None
    results: list = []
    llm_chain: Optional[Chain] = None

    class Config:
        validate_assignment = True

    @root_validator(pre=False, skip_on_failure=True)
    def set_llm_chain(cls, values):
        llm_chain = values.get("llm_chain")
        llm = values.get("llm")
        few_shot_template = values.get("template")

        if not llm_chain:  # If llm_chain is None or not present
            if llm is None or few_shot_template is None:
                raise ValidationError("Both llm and few_shot_template must be provided if llm_chain is not given.")
            values["llm_chain"] = LLMChain(llm=llm, prompt=few_shot_template)

        return values

    def generate(self, subject: str, runs: int, **kwargs) -> List[str]:
        """Generate synthetic data using the given subject string.

        Args:
            subject (str): The subject the synthetic data will be about.
            runs (int): Number of times to generate the data using the given subject.

        Returns:
            List[str]: List of generated synthetic data.
        """
        for _ in range(runs):
            result = self.llm_chain.run(subject=subject, **kwargs)
            self.results.append(result)
        return self.results

    async def agenerate(self, subject: str, runs: int, **kwargs) -> List[str]:
        """Generate synthetic data using the given subject async.

        Args:
            subject (str): The subject the synthetic data will be about.
            runs (int): Number of times to generate the data using the given subject async.

        Returns:
            List[str]: List of generated synthetic data for the given subject.
        """

        async def run_chain(subject: str, **kwargs):
            result = await self.llm_chain.arun(subject=subject, **kwargs)
            self.results.append(result)

        await asyncio.gather(*(run_chain(subject=subject, **kwargs) for _ in range(runs)))
        return self.results

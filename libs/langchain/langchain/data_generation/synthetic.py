from typing import Dict, List

from pydantic.class_validators import root_validator
import asyncio
from pydantic.main import BaseModel

from langchain.chains.llm import LLMChain
from langchain.data_generation.prompts import (
    EXAMPLE_PROMPT,
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain.llms import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate


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
    llm: BaseLLM = OpenAI(temperature=1)
    _llm_chain: LLMChain = None  # Will be populated post-init
    results: list = []

    class Config:
        validate_assignment = True

    @root_validator(pre=False, skip_on_failure=True)
    def set_llm_chain(cls, values):
        llm = values.get("llm")
        few_shot_template = values.get("template")

        values["_llm_chain"] = LLMChain(llm=llm, prompt=few_shot_template)

        return values

    def generate(self, subject: str, runs: int) -> List[str]:
        """Generate synthetic data using the given subject matter.

        Args:
            subject (str): The subject the synthetic data will be about.
            runs (int): Number of times to generate the data using the given subject.

        Returns:
            List[str]: List of generated synthetic data.
        """
        for _ in range(runs):
            result = self._llm_chain.run(subject)
            self.results.append(result)
        return self.results

    async def agenerate(self, subject: str, runs: int) -> List[str]:
        """Generate synthetic data using the given subject async.

        Args:
            subject (str): The subject the synthetic data will be about.
            runs (int): Number of times to generate the data using the given subject async.

        Returns:
            List[str]: List of generated synthetic data for the given subject.
        """

        async def run_chain(subject):
            result = await self._llm_chain.arun(subject)
            self.results.append(result)

        await asyncio.gather(*(run_chain(subject) for _ in range(runs)))
        return self.results


def generate_synthetic(
        examples: List[Dict[str, str]],
        subject: str,
        llm=OpenAI(temperature=1),
        prompt_template: PromptTemplate = EXAMPLE_PROMPT,
        runs: int = 10,  # default value
) -> List[str]:
    """Generate synthetic examples based on the provided examples and subject matter.

    This function uses the LLM to produce synthetic examples based on the
    provided examples and the given subject matter. The prompt used for the
    synthetic generation is constructed based on the examples and the
    predefined few-shot prefix and suffix.

    Args:
        examples (List[Dict[str, str]]): List of examples to be used in the prompt.
        subject (str): The subject the synthetic data will be about.
        llm (LLM, optional): Large Language Model to use for generation. Defaults to OpenAI with temperature 1.
        prompt_template (PromptTemplate, optional): Prompt template to use. Defaults to EXAMPLE_PROMPT.
        runs (int, optional): Number of synthetic examples to generate. Defaults to 10.

    Returns:
        List[str]: List of generated synthetic examples.
    """

    prompt = FewShotPromptTemplate(
        prefix=SYNTHETIC_FEW_SHOT_PREFIX,
        examples=examples,
        suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
        input_variables=["subject"],
        example_prompt=prompt_template,
    )

    generator = SyntheticDataGenerator(template=prompt, llm=llm)
    return generator.generate(subject, runs)


async def agenerate_synthetic(
        examples: List[Dict[str, str]],
        subject: str,
        llm=OpenAI(temperature=1),
        prompt_template: PromptTemplate = EXAMPLE_PROMPT,
        runs: int = 10,  # default value
) -> List[str]:
    """Generate synthetic examples based on the provided examples and the subject matter.

    This function uses the LLM to produce synthetic examples based on the
    provided examples and the given subject matter. The prompt used for the
    synthetic generation is constructed based on the examples and the
    predefined few-shot prefix and suffix.

    Args:
        examples (List[Dict[str, str]]): List of examples to be used in the prompt.
        subject (str): The subject the synthetic data will be about.
        llm (LLM, optional): Large Language Model to use for generation. Defaults to OpenAI with temperature 1.
        prompt_template (PromptTemplate, optional): Prompt template to use. Defaults to EXAMPLE_PROMPT.
        runs (int, optional): Number of synthetic examples to generate. Defaults to 10.

    Returns:
        List[str]: List of generated synthetic examples.
    """

    prompt = FewShotPromptTemplate(
        prefix=SYNTHETIC_FEW_SHOT_PREFIX,
        examples=examples,
        suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
        input_variables=["subject"],
        example_prompt=prompt_template,
    )
    generator = SyntheticDataGenerator(template=prompt, llm=llm)
    return await generator.agenerate(subject, runs)

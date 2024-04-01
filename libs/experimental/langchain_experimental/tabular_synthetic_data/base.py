import asyncio
from typing import Any, Dict, List, Optional, Union

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain_core.language_models import BaseLanguageModel


class SyntheticDataGenerator(BaseModel):
    """Generate synthetic data using the given LLM and few-shot template.

    Utilizes the provided LLM to produce synthetic data based on the
    few-shot prompt template.

    Attributes:
        template (FewShotPromptTemplate): Template for few-shot prompting.
        llm (Optional[BaseLanguageModel]): Large Language Model to use for generation.
        llm_chain (Optional[Chain]): LLM chain with the LLM and few-shot template.
        example_input_key (str): Key to use for storing example inputs.

    Usage Example:
        >>> template = FewShotPromptTemplate(...)
        >>> llm = BaseLanguageModel(...)
        >>> generator = SyntheticDataGenerator(template=template, llm=llm)
        >>> results = generator.generate(subject="climate change", runs=5)
    """

    template: FewShotPromptTemplate
    llm: Optional[BaseLanguageModel] = None
    results: list = []
    llm_chain: Optional[Chain] = None
    example_input_key: str = "example"

    class Config:
        validate_assignment = True

    @root_validator(pre=False, skip_on_failure=True)
    def set_llm_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        llm_chain = values.get("llm_chain")
        llm = values.get("llm")
        few_shot_template = values.get("template")

        if not llm_chain:  # If llm_chain is None or not present
            if llm is None or few_shot_template is None:
                raise ValueError(
                    "Both llm and few_shot_template must be provided if llm_chain is "
                    "not given."
                )
            values["llm_chain"] = LLMChain(llm=llm, prompt=few_shot_template)

        return values

    @staticmethod
    def _format_dict_to_string(input_dict: Dict) -> str:
        formatted_str = ", ".join(
            [f"{key}: {value}" for key, value in input_dict.items()]
        )
        return formatted_str

    def _update_examples(self, example: Union[BaseModel, Dict[str, Any], str]) -> None:
        """Prevents duplicates by adding previously generated examples to the few shot
        list."""
        if self.template and self.template.examples:
            if isinstance(example, BaseModel):
                formatted_example = self._format_dict_to_string(example.dict())
            elif isinstance(example, dict):
                formatted_example = self._format_dict_to_string(example)
            else:
                formatted_example = str(example)
            self.template.examples.pop(0)
            self.template.examples.append({self.example_input_key: formatted_example})

    def generate(self, subject: str, runs: int, *args: Any, **kwargs: Any) -> List[str]:
        """Generate synthetic data using the given subject string.

        Args:
            subject (str): The subject the synthetic data will be about.
            runs (int): Number of times to generate the data.
            extra (str): Extra instructions for steerability in data generation.

        Returns:
            List[str]: List of generated synthetic data.

        Usage Example:
            >>> results = generator.generate(subject="climate change", runs=5,
            extra="Focus on environmental impacts.")
        """
        if self.llm_chain is None:
            raise ValueError(
                "llm_chain is none, either set either llm_chain or llm at generator "
                "construction"
            )
        for _ in range(runs):
            result = self.llm_chain.run(subject=subject, *args, **kwargs)
            self.results.append(result)
            self._update_examples(result)
        return self.results

    async def agenerate(
        self, subject: str, runs: int, extra: str = "", *args: Any, **kwargs: Any
    ) -> List[str]:
        """Generate synthetic data using the given subject asynchronously.

        Note: Since the LLM calls run concurrently,
        you may have fewer duplicates by adding specific instructions to
        the "extra" keyword argument.

        Args:
            subject (str): The subject the synthetic data will be about.
            runs (int): Number of times to generate the data asynchronously.
            extra (str): Extra instructions for steerability in data generation.

        Returns:
            List[str]: List of generated synthetic data for the given subject.

        Usage Example:
            >>> results = await generator.agenerate(subject="climate change", runs=5,
            extra="Focus on env impacts.")
        """

        async def run_chain(
            subject: str, extra: str = "", *args: Any, **kwargs: Any
        ) -> None:
            if self.llm_chain is not None:
                result = await self.llm_chain.arun(
                    subject=subject, extra=extra, *args, **kwargs
                )
                self.results.append(result)

        await asyncio.gather(
            *(run_chain(subject=subject, extra=extra) for _ in range(runs))
        )
        return self.results

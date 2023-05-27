from typing import Any, List, Tuple, Callable

from langchain.prompts.base import BasePromptTemplate
from langchain.schema import PromptValue


class PipelinePromptTemplate(BasePromptTemplate):
    final_prompt: BasePromptTemplate
    pipeline_prompts: List[Tuple[str, BasePromptTemplate, bool]]

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        for k, prompt, as_str in self.pipeline_prompts:
            prompt_val = prompt.format_prompt(**kwargs)
            if as_str:
                kwargs[k] = prompt_val.to_string()
            else:
                kwargs[k] = prompt_val.to_messages()
        return self.final_prompt.format_prompt(**kwargs)

    def format(self, **kwargs: Any) -> str:
        return self.format_prompt(**kwargs).to_string()

    @property
    def _prompt_type(self) -> str:
        raise ValueError

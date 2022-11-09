from langchain.llms.base import LLM
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import Prompt
from langchain.input import print_text
from typing import List, Optional

class ModelComparer:

    def __init__(self, llms: List[LLM], prompt: Optional[Prompt] = None):
        self.llms = llms
        if prompt is None:
            self.prompt = Prompt(input_variables=["_input"], template="{_input}")
        else:
            self.prompt = prompt

    def compare(self, text: str):
        print_text(f"Input:\n{text}", end="\n")
        for llm in self.llms:
            print_text(llm, end="\n")
            chain = LLMChain(llm=llm, prompt=self.prompt)
            output = chain.predict(_input=text)
            print_text(output, end="\n")

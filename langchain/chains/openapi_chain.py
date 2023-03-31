from typing import Dict, List

from langchain.chains.base import Chain
from langchain.requests import RequestsWrapper
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.schema import BaseOutputParser

prompt = """Here is an API:

{api_spec}

Your job is to answer questions by returning valid JSON to send to this API in order to answer the user's question. 
Response with valid markdown, eg in the format:

[JSON BEGIN]
```json
...
```
[JSON END]

Here is the question you are trying to answer:

{question}"""
PROMPT = PromptTemplate.from_template(prompt)

class OpenAPIChain(Chain):

    endpoint_spec: str
    llm_chain: LLMChain
    output_parser: BaseOutputParser
    url: str
    requests_method: str
    requests_wrapper: RequestsWrapper

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, endpoint_spec: str, output_parser: BaseOutputParser, **kwargs):
        chain = LLMChain(llm=llm, prompt=PROMPT)
        return cls(llm_chain=chain, endpoint_spec=endpoint_spec, output_parser=output_parser, **kwargs)


    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        return ["response"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        question = inputs["input"]
        response = self.llm_chain.run(question=question, api_spec=self.endpoint_spec)
        parsed = self.output_parser.parse(response)
        requst_fn = getattr(self.requests_wrapper, self.requests_method)
        result = requst_fn(
            self.url,
            params=parsed,
        )
        return {"response": result}

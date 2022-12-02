from langchain import LLMChain
from typing import Optional, List, Any, Dict
from pydantic import BaseModel
from langchain.chains.base import Chain
from langchain.chains.api.prompt import API_URL_PROMPT, API_RESPONSE_PROMPT
from langchain.llms.base import LLM
import requests

class RequestsChain(BaseModel):
    headers : Optional[dict] = None

    def run(self, url: str) -> str:
      return requests.get(url, headers=self.headers).text

class APIChain(Chain, BaseModel):
  api_request_chain: LLMChain
  api_answer_chain: LLMChain
  requests_chain: RequestsChain
  api_docs: str
  question_key: str = "question"
  output_key : str = "output"
  
  @property
  def input_keys(self) -> List[str]:
    """Input keys this chain expects."""
    return [self.question_key]

  @property
  def output_keys(self) -> List[str]:
    """Output keys this chain expects."""
    return  [self.output_key]

  def _call(self, inputs : Dict[str, str]) -> Dict[str, str]:
      question = inputs[self.question_key]
      api_url = self.api_request_chain.predict(question=question,api_docs=self.api_docs)
      api_response = self.requests_chain.run(api_url)
      answer = self.api_answer_chain.predict(question=question,api_docs=self.api_docs, api_url=api_url, api_response=api_response)
      return {self.output_key: answer}

  @classmethod
  def from_llm_and_api_docs(cls, llm: LLM, api_docs: str, **kwargs : Any) -> "APIChain":
    get_request_chain = LLMChain(llm=llm, prompt=API_URL_PROMPT)
    requests_wrapper = RequestsChain()
    get_answer_chain = LLMChain(llm=llm, prompt=API_RESPONSE_PROMPT)
    return cls(request_chain=get_request_chain, answer_chain=get_answer_chain, requests_wrapper=requests_wrapper, api_docs=api_docs, **kwargs)
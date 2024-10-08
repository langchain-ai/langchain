"""
This class can allow interactions with LMStudio hosted LLMs.
To make it work just go to Local Inference Server tab on LMStudio application and click start.
Then, by utilizing this class you can use LangChain to interact with your locally hosted LMStudio LLM.

Example usage:
Example prompt template (it expects those)
context_prompt = ChatPromptTemplate.from_messages([
    ('system',
        "Answer the question using only the context"
        "\n\nQuestion: {question}\n\nContext: {context}" ## Double reinforcement
    ), ('user', "{question}"),
])

Example loading and calling the LLM (hardcoded details so it'll be easily understandable):
llm = LMStudioLLM(base_url="http://localhost:1234/v1", api_key="lm-studio", model="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF")

Then either llm.invoke("your prompt")
or use it as part of a chain and call the invoke function.
"""
from langchain.llms import BaseLLM
from typing import List, Optional, Dict, Any
import requests


class LMStudioLLM(BaseLLM):
    base_url: str
    api_key: str
    model: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None, temperature: float = 0.5, max_tokens: int = 1000) -> str:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        json_data = {
            'model': self.model,
            'messages': [
                # This following thing assumes you're using ChatTemplate and having system prompt there
                {"role": "system", "content": prompt.messages[0].content},
                {"role": "user", "content": prompt.messages[1].content}
            ],
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        response = requests.post(f'{self.base_url}/chat/completions', headers=headers, json=json_data)
        response_json = response.json()
        return response_json['choices'][0]['message']['content']

    def _identifying_params(self) -> Dict[str, Any]:
        return {
            'base_url': self.base_url,
            'api_key': self.api_key,
            'model': self.model,
        }

    @property
    def _llm_type(self) -> str:
        return "lmstudio"

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, temperature: float = 0.5, max_tokens: int = 1000) -> List[str]:
        results = []
        for prompt in prompts:
            results.append(self._call(prompt, stop, temperature, max_tokens))
        return results

    def invoke(self, prompt: str, stop: Optional[List[str]] = None, temperature: float = 0.5, max_tokens: int = 1000) -> str:
        return self._call(prompt, stop, temperature, max_tokens)

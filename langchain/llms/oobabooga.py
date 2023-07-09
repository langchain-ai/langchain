import langchain
from langchain.llms.base import LLM, Optional, List, Mapping, Any
import requests
from pydantic import Field


class Oobabooga(LLM):
    endpoint: str = Field(...)

    @property
    def _llm_type(self) -> str:
        return "custom"
    

    def _call(self, prompt: str, stop: Optional[List[str]]=None) -> str:
        data = {
            'prompt': prompt,
            'max_new_tokens': 1800,
            'preset': 'None',
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.1,
            'typical_p': 1,
            'epsilon_cutoff': 0,
            'eta_cutoff': 0,
            'tfs': 1,
            'top_a': 0,
            'repetition_penalty': 1.21,
            'top_k': 40,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mirostat_eta': 0.1,
            'seed': -1,
            'add_bos_token': True,
            'truncation_length': 2048,
            'ban_eos_token': False,
            'skip_special_tokens': True
        }

        if stop is not None:
            data["stop_sequence"] = stop

        response = requests.post(f'{self.endpoint}/api/v1/generate', json=data)
        response.raise_for_status()

        json_response = response.json()
        if 'results' in json_response and len(json_response['results']) > 0 and 'text' in json_response['results'][0]:
            text = json_response['results'][0]['text'].strip()
            if stop is not None:
                for sequence in stop:
                    if text.endswith(sequence):
                        text = text[: -len(sequence)].rstrip()

            print(text)
            return text
        else:
            raise ValueError('Unexpected response format from Ooba API')

    def __call__(self, prompt: str, stop: Optional[List[str]]=None) -> str:
        return self._call(prompt, stop)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'endpoint': self.endpoint} #return the kobold_ai_api as an identifying parameter
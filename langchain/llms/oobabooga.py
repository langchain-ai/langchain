import langchain
from langchain.llms.base import LLM, Optional, List, Mapping, Any
import requests
from pydantic import Field

def clean_url(url):
    if url.endswith('/api'):
        return url[:-4]
    elif url.endswith('/'):
        return url[:-1]
    else:
        return url

class OobaboogaLLM(LLM):
    endpoint: str = Field(...)
    temperature: float = Field(0.7)
    max_new_tokens: int = Field(1800)
    preset: str = Field('None')
    do_sample: bool = Field(True)
    top_p: float = Field(0.1)
    typical_p: float = Field(1)
    epsilon_cutoff: int = Field(0)
    eta_cutoff: int = Field(0)
    tfs: int = Field(1)
    top_a: int = Field(0)
    repetition_penalty: float = Field(1.21)
    top_k: int = Field(40)
    min_length: int = Field(0)
    no_repeat_ngram_size: int = Field(0)
    num_beams: int = Field(1)
    penalty_alpha: int = Field(0)
    length_penalty: int = Field(1)
    early_stopping: bool = Field(False)
    mirostat_mode: int = Field(0)
    mirostat_tau: int = Field(5)
    mirostat_eta: float = Field(0.1)
    seed: int = Field(-1)
    add_bos_token: bool = Field(True)
    truncation_length: int = Field(2048)
    ban_eos_token: bool = Field(False)
    skip_special_tokens: bool = Field(True)

    @property
    def _llm_type(self) -> str:
        return "custom"
    

    def _call(self, prompt: str, stop: Optional[List[str]]=None) -> str:
        data = {
            'prompt': prompt,
            'max_new_tokens': self.max_new_tokens,
            'preset': self.preset,
            'do_sample': self.do_sample,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'typical_p': self.typical_p,
            'epsilon_cutoff': self.epsilon_cutoff,
            'eta_cutoff': self.eta_cutoff,
            'tfs': self.tfs,
            'top_a': self.top_a,
            'repetition_penalty': self.repetition_penalty,
            'top_k': self.top_k,
            'min_length': self.min_length,
            'no_repeat_ngram_size': self.no_repeat_ngram_size,
            'num_beams': self.num_beams,
            'penalty_alpha': self.penalty_alpha,
            'length_penalty': self.length_penalty,
            'early_stopping': self.early_stopping,
            'mirostat_mode': self.mirostat_mode,
            'mirostat_tau': self.mirostat_tau,
            'mirostat_eta': self.mirostat_eta,
            'seed': self.seed,
            'add_bos_token': self.add_bos_token,
            'truncation_length': self.truncation_length,
            'ban_eos_token': self.ban_eos_token,
            'skip_special_tokens': self.skip_special_tokens
        }

        if stop is not None:
            data["stop_sequence"] = stop

        response = requests.post(f'{clean_url(self.endpoint)}/api/v1/generate', json=data)
        response.raise_for_status()

        json_response = response.json()
        if "results" in json_response and len(json_response["results"]) > 0:
            text = json_response["results"][0]["text"].strip()
            if not text:
                raise ValueError("No text was generated. Check the server.")

    def __call__(self, prompt: str, stop: Optional[List[str]]=None) -> str:
        return self._call(prompt, stop)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'endpoint': self.endpoint} #return the kobold_ai_api as an identifying parameter
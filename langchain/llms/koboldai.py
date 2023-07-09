import langchain
from langchain.llms.base import LLM, Optional, List, Mapping, Any
import requests
from pydantic import Field


class KoboldApiLLM(LLM):
    endpoint: str = Field(...)
    use_story: bool = Field(False)
    use_authors_note: bool = Field(False)
    use_world_info: bool = Field(False)
    use_memory: bool = Field(False)
    max_context_length: int = Field(1600)
    max_length: int = Field(80)
    rep_pen: float = Field(1.12)
    rep_pen_range: int = Field(1024)
    rep_pen_slope: float = Field(0.9)
    temperature: float = Field(0.6)
    tfs: float = Field(0.9)
    top_p: float = Field(0.95)
    top_k: float = Field(0.6)
    typical: int = Field(1)
    frmttriminc: bool = Field(True)

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]]=None) -> str:
        # Prepare the JSON data
        data = {
            "prompt": prompt,
            "use_story": self.use_story,
            "use_authors_note": self.use_authors_note,
            "use_world_info": self.use_world_info,
            "use_memory": self.use_memory,
            "max_context_length": self.max_context_length,
            "max_length": self.max_length,
            "rep_pen": self.rep_pen,
            "rep_pen_range": self.rep_pen_range,
            "rep_pen_slope": self.rep_pen_slope,
            "temperature": self.temperature,
            "tfs": self.tfs,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "typical": self.typical,
            "frmttriminc": self.frmttriminc,
        }

        # Add the stop sequences to the data if they are provided
        if stop is not None:
            data["stop_sequence"] = stop
            
        # Send a POST request to the Kobold API with the data
        response = requests.post(f"{self.endpoint}/api/v1/generate", json=data)
        response.raise_for_status()

        # Check for the expected keys in the response JSON
        json_response = response.json()
        if "results" in json_response and len(json_response["results"]) > 0 and "text" in json_response["results"][0]:
            # Return the generated text
            text = json_response["results"][0]["text"].strip().replace("'''", "```")

            # Remove the stop sequence from the end of the text, if it's there
            if stop is not None:
                for sequence in stop:
                    if text.endswith(sequence):
                        text = text[: -len(sequence)].rstrip()


            print(text)
            return text
        else:
            raise ValueError("Unexpected response format from Kobold API")


    
    def __call__(self, prompt: str, stop: Optional[List[str]]=None) -> str:
        return self._call(prompt, stop)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'endpoint': self.endpoint} #return the kobold_ai_api as an identifying parameter


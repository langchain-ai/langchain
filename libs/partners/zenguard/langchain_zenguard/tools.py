import os
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError, validator
from langchain_core.tools import BaseTool


class Detector(str, Enum):
    ALLOWED_TOPICS = "allowed_subjects"
    BANNED_TOPICS = "banned_subjects"
    PROMPT_INJECTION = "prompt_injection"
    KEYWORDS = "keywords"
    PII = "pii"
    SECRETS = "secrets"
    TOXICITY = "toxicity"

class DetectorAPI(str, Enum):
    ALLOWED_TOPICS = "v1/detect/topics/allowed"
    BANNED_TOPICS = "v1/detect/topics/banned"
    PROMPT_INJECTION = "v1/detect/prompt_injection"
    KEYWORDS = "v1/detect/keywords"
    PII = "v1/detect/pii"
    SECRETS = "v1/detect/secrets"
    TOXICITY = "v1/detect/toxicity"

class ZenGuardInput(BaseModel):
    prompts: List[str] = Field(..., min_items=1, min_length=1)
    detectors: List[Detector] = Field(..., min_items=1)
    in_parallel: bool = Field(default=True)

class ZenGuardTool(BaseTool):
    name = "ZenGuard"
    description = "Fastest LLM Security Guardrails"
    args_schema = ZenGuardInput
    return_direct = True

    zenguard_api_key: Optional[str] = Field(default=None)

    _ZENGUARD_API_URL_ROOT = "https://api.zenguard.ai/"
    _ZENGUARD_API_KEY_ENV_NAME = "ZENGUARD_API_KEY"

    @validator("api_key", pre=True, always=True)
    def set_api_key(cls, v):
        if v is None:
            v = os.getenv(cls._ZENGUARD_API_KEY_ENV_NAME)
        if v is None:
            raise ValidationError(
                "The api_key tool option must be set either "
                "by passing api_key to the tool or by setting "
                f"the f{cls._ZENGUARD_API_KEY_ENV_NAME} environment variable"
            )
        return v
        
    def _run(
            self,
            prompts: List[str],
            detectors: List[Detector],
            in_parallel: bool = True,
    ) -> Dict[str, Any]:
        try:
            postfix = None
            json = None
            if len(detectors) == 1:
                postfix = self._convert_detector_to_api(detectors[0])
                json = {"messages": prompts}
            else:
                postfix = "v1/detect"
                json = {
                    "messages": prompts,
                    "in_parallel": True,
                    "detectors": detectors,
                }
            response = requests.post(
                self._ZENGUARD_API_URL_ROOT + postfix,
                json=json,
                headers={"x-api-key": self.zenguard_api_key},
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except (requests.HTTPError, requests.Timeout) as e:
            return {"error": str(e)}
        
    def _convert_detector_to_api(detector: Detector):
        return DetectorAPI[f"{Detector.title}"].value

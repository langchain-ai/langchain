from langchain_core.pydantic_v1 import BaseModel
from typing import Optional


class Model(BaseModel):
    id: str
    model_type: Optional[str] = None
    client: Optional[str] = None
    path: str


MODEL_SPECS = {
    'playground_smaug_72b': {"model_type": "chat"},
    'playground_kosmos_2': {"model_type": "chat"},
    'playground_llama2_70b': {"model_type": "chat"},
    'playground_nvolveqa_40k': {"model_type": "embedding"},
    'playground_nemotron_qa_8b': {"model_type": "qa"},
    'playground_gemma_7b': {"model_type": "chat"},
    'playground_mistral_7b': {"model_type": "chat"},
    'playground_mamba_chat': {"model_type": "chat"},
    'playground_phi2': {"model_type": "chat"},
    'playground_sdxl': {"model_type": "image_out"},
    'playground_nv_llama2_rlhf_70b': {"model_type": "chat"},
    'playground_neva_22b': {"model_type": "image_in"},
    'playground_yi_34b': {"model_type": "chat"},
    'playground_nemotron_steerlm_8b': {"model_type": "chat"},
    'playground_cuopt': {"model_type": "cuopt"},
    'playground_llama_guard': {"model_type": "classifier"},
    'playground_starcoder2_15b': {"model_type": "completion"},
    'playground_deplot': {"model_type": "image_in"},
    'playground_llama2_code_70b': {"model_type": "chat"},
    'playground_gemma_2b': {"model_type": "chat"},
    'playground_seamless': {"model_type": "translation"},
    'playground_mixtral_8x7b': {"model_type": "chat"},
    'playground_fuyu_8b': {"model_type": "image_in"},
    'playground_llama2_code_34b': {"model_type": "chat"},
    'playground_llama2_code_13b': {"model_type": "chat"},
    'playground_steerlm_llama_70b': {"model_type": "chat"},
    'playground_clip': {"model_type": "similarity"},
    'playground_llama2_13b': {"model_type": "chat"},
}

client_map = {
    "chat": "ChatNVIDIA",
    "classifier": None,
    "completion": "NVIDIA",
    "cuopt": None,
    "embedding": "NVIDIAEmbeddings",
    "image_in": "ChatNVIDIA",
    "image_out": "ImageNVIDIA",
    "qa": "ChatNVIDIA",
    "similarity": None,
    "translation": None,
}

MODEL_SPECS = {
    k: {**v, "client": client_map[v["model_type"]]}
    for k, v in MODEL_SPECS.items()
}
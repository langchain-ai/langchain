import os
import json
from loguru import logger
import torch

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM


def download_and_unload_peft(model_id, revision, trust_remote_code):
    torch_dtype = torch.float16

    logger.info("Peft model detected.")
    logger.info("Loading the model it might take a while without feedback")
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
    except Exception:
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
    logger.info(f"Loaded.")
    logger.info(f"Merging the lora weights.")

    base_model_id = model.peft_config["default"].base_model_name_or_path

    model = model.merge_and_unload()

    os.makedirs(model_id, exist_ok=True)
    cache_dir = model_id
    logger.info(f"Saving the newly created merged model to {cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
    model.save_pretrained(cache_dir, safe_serialization=True)
    model.config.save_pretrained(cache_dir)
    tokenizer.save_pretrained(cache_dir)

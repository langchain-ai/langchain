from typing import Dict, List


def load_dataset(uri: str) -> List[Dict]:
    from datasets import load_dataset

    dataset = load_dataset(f"LangChainDatasets/{uri}")
    return [d for d in dataset["train"]]

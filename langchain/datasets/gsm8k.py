import json
import requests

from dataset import Dataset
from dataset import LANGCHAIN_STOP_SEQUENCE
from example import Example


class GSM8KDataset(Dataset):
    """ Dataset for grade school math problems from OpenAI. 
        For details, see https://arxiv.org/abs/2110.14168v2
    """

    def __init__(self, split="test", stop_sequence=LANGCHAIN_STOP_SEQUENCE):
        super().__init__(stop_sequence=stop_sequence)
        if split != "test" or split != "train":
            raise ValueError("split must be either 'test' or 'train'")
        text = self._maybe_load_cached_data(f"gsm8k/{split}.jsonl")
        if text is None:
            url = f"https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/{split}.jsonl"
            text = requests.get(url).text
            self._maybe_cache_data(f"gsm8k/{split}.jsonl", text)
        self.data = [json.loads(line) for line in text.splitlines()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = item["question"]
        y = item["answer"]
        return Example(
            x, y, x_prefix="Q:", y_prefix="A:", stop_sequence=self.stop_sequence
        )


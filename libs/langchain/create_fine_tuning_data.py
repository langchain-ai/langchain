
from typing import Literal
from datasets import load_dataset

from langchain.pydantic_v1 import BaseModel


dataset = load_dataset("griffin/chain_of_density", "unannotated")

# demo script for LLM-as-a-judge
# TODO: either create a notebook or delete this file


class Sample(BaseModel):
    article: str
    starting_summary: str
    final_summary: str


samples: list[Sample] = []

for sample in dataset["train"]:
    samples.append(
        Sample(
            article=sample["article"],
            starting_summary=sample["prediction"][0],
            final_summary=sample["prediction"][-1],
        )
    )

# Reserve 200 samples for testing

print("Total number of samples:", len(samples))

samples = samples[200:]

class _Message(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: str

class OpenAIFineTuningSample(BaseModel):
    messages: list[_Message]


fine_tuning_samples: list[OpenAIFineTuningSample] = []

for sample in samples:
    fine_tuning_samples.append(
        OpenAIFineTuningSample(
            messages=[
                _Message(role="user", content=f"Give a summary of the following article:\n\n{sample.article}"),
                _Message(role="assistant", content=sample.final_summary)
            ]
        )
    )

print("Number of samples:", len(fine_tuning_samples))

total_chars = 0

for sample in fine_tuning_samples:
    total_chars += sum([len(message.content) for message in sample.messages])

print("Total tokens:", total_chars/3.5)

with open("fine_tuning_examples.jsonl", "w") as f:
    for sample in fine_tuning_samples:
        f.write(sample.json() + "\n")


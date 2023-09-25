import asyncio

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

import langchain
from langchain.cache import SQLiteCache
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.comparison.llm_as_a_judge import LLMAsAJudgePairwiseEvalChain
from langchain.pydantic_v1 import BaseModel

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

dataset = load_dataset("griffin/chain_of_density", "unannotated")

load_dotenv()

# demo script for LLM-as-a-judge
# TODO: either create a notebook or delete this file

llm = ChatOpenAI(temperature=0, model="gpt-4", max_retries=1000)

evaluator = LLMAsAJudgePairwiseEvalChain.from_llm(llm=llm)


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


def _reverse_verdict(verdict: str) -> str:
    return "Win" if verdict == "Loss" else "Loss" if verdict == "Win" else "Tie"


async def evaluate(sample: Sample) -> bool:
    reverse = (len(sample.starting_summary) + len(sample.final_summary)) % 2 == 0
    result = await evaluator.aevaluate_string_pairs(
        input=f"Give a summary of the following article:\n\n{sample.article}",
        prediction=sample.final_summary if not reverse else sample.starting_summary,
        prediction_b=sample.starting_summary if not reverse else sample.final_summary,
    )
    print(result)
    if reverse:
        return _reverse_verdict(result["verdict"])
    return result["verdict"]


async def main() -> None:
    pbar = tqdm(total=len(samples[:100]))
    sempahore = asyncio.Semaphore(10)

    async def boxed_evaluate(sample: Sample) -> str:
        with get_openai_callback() as cb:
            async with sempahore:
                results = await evaluate(sample)
                pbar.update(1)
                print("Total cost:", cb.total_cost)
                return results

    results = await asyncio.gather(
        *[boxed_evaluate(sample) for sample in samples[:100]]
    )

    results_excluding_ties = [result for result in results if result != "Tie"]
    print(
        "Win rate:",
        sum([result == "Win" for result in results]) / len(results_excluding_ties),
    )


if __name__ == "__main__":
    asyncio.run(main())

# e = evaluator.evaluate_string_pairs(
#     prediction="The chemical formula for water is H2O, which means there are two hydrogen atoms and one oxygen atom",
#     prediction_b="The chemical formula for water is H2O.",
#     input="What is the chemical formula for water?",
# )

# print(e)

# N=100 With first and last summary
# Win rate: 83%

import asyncio
from tqdm import tqdm
from langchain.cache import SQLiteCache
from dotenv import load_dotenv
from datasets import load_dataset
import langchain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chat_models.openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.evaluation.scoring import ScoreStringEvalChain
from langchain.callbacks.manager import get_openai_callback
    
class SummaryParser(SimpleJsonOutputParser):

    def parse(self, text: str) -> str:
        raw_json = super().parse(text)
        return raw_json[-1]["Denser_Summary"]

    @property
    def _type(self) -> str:
        return "summary_parser"

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

dataset = load_dataset("griffin/chain_of_density", "unannotated")

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4-0613", max_retries=1000)

ft_llm = ChatOpenAI(temperature=0, model="ft:gpt-3.5-turbo-0613:personal:cod-summarization:82oPBKod", max_retries=1000)

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

BASE_PROMPT = ChatPromptTemplate.from_template("""Write a VERY short summary of the Article. Do not exceed 70 words.
                                               
Article: {article}""")

FT_PROMPT = ChatPromptTemplate.from_template("""Give a summary of the following article:\n\n{article}""")

base_summarize_chaim = BASE_PROMPT | llm

ft_summarize_chain = FT_PROMPT | ft_llm

evaluator = ScoreStringEvalChain.from_llm(llm=llm)

async def evaluate(sample: Sample) -> float:
    #base_summary = (await base_summarize_chaim.ainvoke({"article": sample.article})).content
    ft_summary = (await ft_summarize_chain.ainvoke({"article": sample.article})).content
    result = await evaluator.aevaluate_strings(
        input=f"Give a summary of the following article:\n\n{sample.article}",
        prediction=ft_summary,
    )
    print("Summary:", ft_summary)
    print("Reasoning:", result["reasoning"])
    return result["score"]

async def main() -> None:
    pbar = tqdm(total=len(samples[:40]))
    sempahore = asyncio.Semaphore(10)

    async def boxed_evaluate(sample: Sample) -> str:
        with get_openai_callback() as cb:
            async with sempahore:
                results = await evaluate(sample)
                pbar.update(1)
                print("Total cost:", cb.total_cost)
                return results

    results = await asyncio.gather(
        *[boxed_evaluate(sample) for sample in samples[:40]]
    )

    print("Average score:", sum(results) / len(results))
    
if __name__ == "__main__":
    asyncio.run(main())

# N=40 With base summary
# Average score: 6.4

# N=40 With ft summary
# Average score: 7.7
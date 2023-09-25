from langsmith import Client

from langchain.cache import SQLiteCache
from dotenv import load_dotenv
from datasets import load_dataset
import langchain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.callbacks.manager import get_openai_callback
from langchain.schema.runnable.config import RunnableConfig

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

dataset = load_dataset("griffin/chain_of_density", "unannotated")

load_dotenv()

client = Client()

llm = ChatOpenAI(temperature=0, model="gpt-4-0613", max_retries=1000)

articles: list[str] = []

for sample in dataset["train"]:
    articles.append(
        sample["article"]
    )

PROMPT = """Article: {article}
You will generate increasingly concise, entity-dense summaries of the above article. 

Repeat the following 2 steps 5 times. 

Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary. 
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities. 

A missing entity is:
- relevant to the main story, 
- specific yet concise (5 words or fewer), 
- novel (not in the previous summary), 
- faithful (present in the article), 
- anywhere (can be located anywhere in the article).

Guidelines:

- The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
- Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
- The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article. 
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities. 

Remember, use the exact same number of words for each summary.
Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary"."""  # noqa: E501

class SummaryParser(SimpleJsonOutputParser):

    def parse(self, text: str) -> str:
        raw_json = super().parse(text)
        return raw_json[-1]["Denser_Summary"]

    @property
    def _type(self) -> str:
        return "summary_parser"

cod_summarization_prompt = ChatPromptTemplate.from_messages(
    ("human", PROMPT)
)

cod_summarize_chain = LLMChain(llm=llm, prompt=cod_summarization_prompt, output_parser=SummaryParser())

# Batches of 10 articles

batches = [
    articles[i : i + 10] for i in range(0, len(articles), 10)
]

dataset_name = "Summarization Dataset using Chain of Density"

# Storing inputs in a dataset lets us
# run chains and LLMs over a shared set of examples.
dataset = client.create_dataset(
    dataset_name=dataset_name, description="Summaries of news articles"
)

with get_openai_callback() as cb:
    for batch in batches[:10]:
        outputs = cod_summarize_chain.batch(inputs=batch)
        print("Total cost:", cb.total_cost)
        for input, output in zip(batch, outputs):
            client.create_example(
                inputs={"article": input},
                outputs={"summary": output["text"]},
                dataset_id=dataset.id,
            )
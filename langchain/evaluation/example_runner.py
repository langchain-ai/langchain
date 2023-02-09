import asyncio

from pydantic import BaseModel, validator, root_validator, Field
from pathlib import Path
from typing import Optional, Dict, Any, List
from langchain.utils import get_from_dict_or_env
from pydantic.networks import AnyHttpUrl
import requests
import datetime
import langchain
from langchain.agents import AgentExecutor
from langchain.chains.base import Chain
from langchain.llms.base import BaseLLM
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.stdout import StdOutCallbackHandler
from urllib.parse import urlparse
import os


class ExampleBase(BaseModel):
    """Base class for Example."""
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] | None = None
    dataset_id: int


class ExampleCreate(ExampleBase):
    """Create class for Example."""


class Example(ExampleBase):
    """Example schema."""
    id: int


class DatasetBase(BaseModel):
    """Base class for Dataset."""
    name: str
    description: str
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    examples: List[Example] = Field(default_factory=list)


class DatasetCreate(DatasetBase):
    """Create class for Dataset."""
    pass


class Dataset(DatasetBase):
    """Dataset schema."""
    id: int


class CsvDataset(BaseModel):
    """Class for a csv file that can be uploaded to a LangChain endpoint."""
    csv_path: Path
    description: str
    input_keys: List[str]
    output_keys: List[str]

    @validator("csv_path")
    def validate_csv_path(cls, v):
        """Validate that the csv path is valid."""
        if not v.exists():
            raise ValueError("CSV file does not exist.")
        return v


def fetch_dataset_from_endpoint(name: str, headers: Dict[str, str], endpoint: str = "https://localhost:8000") -> Dataset:
    """Fetch a dataset from a LangChain endpoint."""
    response = requests.get(f"{endpoint}/datasets?name={name}", headers=headers)
    response.raise_for_status()
    if len(response.json()) == 0:
        raise ValueError(f"Dataset with name {name} does not exist.")
    return Dataset(**(response.json()[0]))


def upload_csv_dataset_to_endpoint(csv_dataset: CsvDataset, headers: Dict[str, str], endpoint: str = "https://localhost:8000") -> Dataset:
    """Upload a csv to a LangChain endpoint."""
    with open(csv_dataset.csv_path, "rb") as f:
        response = requests.post(
            f"{endpoint}/datasets/upload",
            headers=headers,
            files={"file": (csv_dataset.csv_path.name, f)},
            data={
                "input_keys": csv_dataset.input_keys,
                "output_keys": csv_dataset.output_keys,
                "description": csv_dataset.description,
            },
        )
    response.raise_for_status()
    return Dataset(**response.json())


class ExampleRunner(BaseModel):
    """Class that runs an LLM, chain or agent on a set of examples."""

    langchain_endpoint: AnyHttpUrl
    dataset: Dataset
    csv_dataset: Optional[CsvDataset] = None
    langchain_dataset_name: Optional[str] = None
    langchain_api_key: Optional[str] = None

    @root_validator(pre=True)
    def validate_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that either csv_path or langchain_dataset is provided but not both."""
        csv_dataset = values.get("csv_dataset")
        langchain_dataset_name = values.get("langchain_dataset_name")
        values["langchain_endpoint"] = os.environ.get("LANGCHAIN_ENDPOINT", "https://localhost:8000")
        langchain_endpoint = values["langchain_endpoint"]
        if csv_dataset is None and langchain_dataset_name is None:
            raise ValueError("Must provide either csv_path or langchain_dataset.")
        if csv_dataset is not None and langchain_dataset_name is not None:
            raise ValueError("Cannot provide both csv_path and langchain_dataset.")
        if urlparse(langchain_endpoint).hostname not in ["localhost", "127.0.0.1", "0.0.0.0"]:
            values["langchain_api_key"] = get_from_dict_or_env(
                values, "langchain_api_key", "LANGCHAIN_API_KEY"
            )
        # Try fetching the dataset to make sure it exists
        if langchain_dataset_name is not None:
            headers: Dict[str, str] = {}
            if values.get("langchain_api_key"):
                headers["x-api-key"] = values["langchain_api_key"]
            values["dataset"] = fetch_dataset_from_endpoint(langchain_dataset_name, headers, langchain_endpoint)
        if csv_dataset is not None:
            # Upload the csv to the endpoint
            headers: Dict[str, str] = {}
            if values.get("langchain_api_key"):
                headers["x-api-key"] = values["langchain_api_key"]
            values["dataset"] = upload_csv_dataset_to_endpoint(csv_dataset, headers, langchain_endpoint)
        return values

    def examples(self) -> List[Example]:
        """Get the examples from the dataset."""
        return self.dataset.examples

    def run_agent(self, agent: AgentExecutor):
        """Run an agent on the examples."""
        for example in self.examples():
            agent.run(**example.inputs)

    def run_chain(self, chain: Chain):
        """Run a chain on the examples."""
        for example in self.examples():
            langchain.set_tracing_callback_manager(example_id=example.id)
            print(chain.run(**example.inputs))

    def run_llm(self, llm: BaseLLM):
        """Run an LLM on the examples."""
        for example in self.examples():
            llm.generate([val for val in example.inputs.values()])


    # async def arun_agent(self, agent: AgentExecutor, num_workers: int = 1):
    #     """Run an agent on the examples."""
    #     # Copy the agent num_workers times
    #     agents = []
    #     for _ in range(num_workers):
    #         tracer = LangChainTracer()
    #         tracer.load_default_session()
    #         manager = CallbackManager([StdOutCallbackHandler(), tracer])
    #         agent.from_agent_and_tools(agent.agent, agent.tools, manager)
    #         agents.append(agent)
    #
    #     i = 0
    #     while i < len(self.examples()):
    #         for agent in agents:
    #             example = self.examples()[i]
    #             await agent.arun(**example.inputs)
    #             i += 1


if __name__ == "__main__":
    os.environ["LANGCHAIN_ENDPOINT"] = "http://127.0.0.1:8000"
    runner = ExampleRunner(
        csv_dataset=CsvDataset(
            csv_path="test_dataset.csv",
            description="Dummy dataset for testing",
            input_keys=["input1", "input2", "input3"],
            output_keys=["output1"],
        ),
    )

    # runner = ExampleRunner(
    #     langchain_dataset_name="test_dataset.csv",
    # )

    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    llm = OpenAI(temperature=0.9, model_name="text-ada-001")
    prompt = PromptTemplate(
        input_variables=["input1", "input2", "input3"],
        template="Complete the sequence: {input1}, {input2}, {input3}",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    runner.run_chain(chain)




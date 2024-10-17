# GPTResearcher - Autonomous LLM-Powered Research Agent 🚀

Unleash the power of **GPTResearcher** – an LLM-powered autonomous agent that dives deep into **web, local, or hybrid sources** to deliver **comprehensive research on any topic**! Now available as a **LangChain Tool**, seamlessly integrate it into **ANY** agent ecosystem and elevate your AI workflows.

[👉 Explore the core library here](https://github.com/assafelovic/gpt-researcher)

---

## Table of Contents
- [Introduction](#introduction)
  - [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [Usage Examples](#usage-examples)
  - [LocalGPTResearcher Example](#localgptresearcher-example)
  - [WebGPTResearcher Example](#webgptresearcher-example)
  - [HybridGPTResearcher Example](#hybridgptresearcher-example)
- [Chaining with Other Components and Agentic Systems](#chaining-with-other-components-and-agentic-systems)
  - [Using AgentExecutor with WebGPTResearcher](#example-using-agentexecutor-with-webgptresearcher)
  - [Simple Sequential Chaining of WebGPTResearcher](#simple-sequential-chaining-of-webgptresearcher)
- [Building from Base Class](#building-from-base-class)
  - [Extending BaseGPTResearcher](#extending-basegptresearcher)
  - [Building CustomGPTResearcher](#building-customgptresearcher)
  - [Off-the-Shelf Usage](#off-the-shelf-usage)
- [Performance Considerations](#performance-considerations)
- [Links and References](#links-and-references)
- [Contribution Guide](#contribution-guide)

---

## Introduction

The `LocalGPTResearcher`, `WebGPTResearcher` and `HybridGPTResearcher` tools are designed to assist with conducting thorough research on specific topics or queries. These tools leverage the power of GPT models to generate detailed reports, making them ideal for various research-related tasks. The `LocalGPTResearcher` tool accesses local data files, while the `WebGPTResearcher` retrieves information from the web. The `HybridGPTResearcher` does both to answer your research questions!

### Key Features

- 🔬 The `LocalGPTResearcher` can work with various local file formats such as PDF, Word documents, CSVs, and more.
- 🛜 The `WebGPTResearcher` fetches data directly from the internet, making it suitable for up-to-date information gathering.
- 🔬🛜 The `HybridGPTResearcher` does them both!
- 📝 Generate research, outlines, resources and lessons reports with local documents and web sources
- 📜 Can generate long and detailed research reports (over 2K words)
- 🌐 Aggregates over 20 web sources per research to form objective and factual conclusions
- 🖥️ Includes an easy-to-use web interface (HTML/CSS/JS)
- 🔍 Scrapes web sources with javascript support
- 📂 Keeps track and context of visited and used web sources
- 📄 Export research reports to PDF, Word and more...

---

## Installation and Setup

### Prerequisites
Ensure you have Python 3 installed on your system.

### Installation
Install the necessary packages using pip:

```bash
pip install gpt-researcher
```

### Environment Variables
For `LocalGPTResearcher` and `HybridGPTResearcher`, you need to set the following environment variables:

```bash
export DOC_PATH=/path/to/your/documents
export OPENAI_API_KEY=your-openai-api-key
export TAVILY_API_KEY=your-tavily-api-key
```

For `WebGPTResearcher`, only the `OPENAI_API_KEY` and `TAVILY_API_KEY` are required:

```bash
export OPENAI_API_KEY=your-openai-api-key
export TAVILY_API_KEY=your-tavily-api-key
```

---

## Usage Examples

### LocalGPTResearcher Example
This example demonstrates how to use `LocalGPTResearcher` to generate a report based on local documents. 

Remember to export the `DOC_PATH` with "Data/" which contains `sample.txt` file for this example. Feel free to modify the file and the query below!

```python
from tool import WebGPTResearcher, LocalGPTResearcher, HybridGPTResearcher # This will be changed after successful PR

# Initialize the tool
researcher_local = LocalGPTResearcher(report_type="research_report")
# You can also define it as `researcher_local = LocalGPTResearcher()` - default report_type is research_report.

# Run a query
query = "What does Higgs look like?"
report = researcher_local.invoke({"query":query})

print("Generated Report:", report)
```

### WebGPTResearcher Example
This example shows how to use `WebGPTResearcher` to generate a report based on web data.

```python
from tool import WebGPTResearcher, LocalGPTResearcher, HybridGPTResearcher # This will be changed after successful PR

# Initialize the tool
researcher_web = WebGPTResearcher(report_type="research_report") # report_type="research_report" is optional as the default value is `research_report`

# Run a query
query = "What are the latest advancements in AI?"
report = researcher_web.invoke({"query":query})

print("Generated Report:", report)
```

### HybridGPTResearcher Example

This example demonstrates how to use `HybridGPTResearcher` to generate a report based on local documents AND the internet.

```python
from tool import WebGPTResearcher, LocalGPTResearcher, HybridGPTResearcher # This will be changed after successful PR

# Initialize the tool
researcher_hybrid = HybridGPTResearcher(report_type="research_report")
# You can also define it as `researcher_local = LocalGPTResearcher()` - default report_type is research_report.

# Run a query
query = "Tell me about the British theoretical physicist Peter Higgs"
report = researcher_hybrid.invoke({"query":query})

print("Generated Report:", report)
```

---

## Chaining with Other Components and Agentic Systems

### Example: Using `AgentExecutor` with `WebGPTResearcher`

Let us see how to build an AgentExecutor wrapper that uses an LLM and our tool to write an essay and provide a citation/signature at the end of the report.

```python
from tool import WebGPTResearcher, LocalGPTResearcher, HybridGPTResearcher # This will be changed after successful PR

from langchain import hub
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


# Let us see how to use the WebGPTResearcher tool along with AgentExecutor to perform a grand task with decision making.
# 1. Let us build a Reactive Agent who takes decisions based on reasoning.
# 2. Let us give our agent 2 tools - WebGPTResearcher and a dummy tool that provides a signature at the end of the text
# 3. We can then wrap our agent and tools inside an AgentExecutor object and ask our question!
# The expectation is the response must be signed at the end after a long report on a research topic.


# Create a new tool called citation_provider.
@tool
def citation_provider(text: str) -> str:
    """Provides a citation or signature"""
    return "\n- Written by GPT-Makesh\nThanks for reading!\n"


# Create the WebGPTResearcher tool
researcher_web = WebGPTResearcher("research_report")

# Initialize tools and components
tools = [
    researcher_web,
    Tool(
        name = "citation_tool",  
        func = citation_provider,  
        description = "Useful for when you need to add citation or signature at the end of text",
    ),
]

# Create an LLM
llm = ChatOpenAI(model="gpt-4o")
prompt = hub.pull("hwchase17/react")

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# Wrap the components inside an AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Run the agent
question = "What are the recent advancements in AI? Provide a citation for your report too."
response = agent_executor.invoke({"input": question})
print("Agent Response:", response)
```

Follow the same steps to implement the same with `LocalGPTResearcher` and `HybridGPTResearcher`. Just make sure to export `DOC_PATH`.

### Example: Simple Sequential Chaining of `WebGPTResearcher`

Let us build a chain of runnables that have a researcher who writes a report and a grader who then grades and scores the report.

```python
from tool import WebGPTResearcher, LocalGPTResearcher, HybridGPTResearcher # This will be changed after successful PR

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI

# Let us use WebGPTResearcher to grade the essay using LECL langchain Chaining tricks
# 1. Use the researcher to write an essay
# 2. Pass it as a chat_prompt_template (a runnable) to a grader to score the essay
# 3. Parse the output as a string


# Create a ChatOpenAI model
grader = ChatOpenAI(model="gpt-4o")
researcher_tool = WebGPTResearcher()
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a essay grader. Give score out of 10 in brief"),
        ("human", "The essay: {essay}"),
    ]
)

# Define our WebGPTResearcher tool as a RunnableLambda
researcher = RunnableLambda(lambda x: researcher_tool.invoke(x))

# Create the combined chain using LangChain Expression Language (LCEL)
chain = researcher | prompt_template | grader | StrOutputParser() 

# Run the chain
result = chain.invoke({"query": "What are the recent advancements in AI?"})

# Output
print(result)
```

Follow the same steps to implement the same with `LocalGPTResearcher` and `HybridGPTResearcher`. Just make sure to export `DOC_PATH`.

---

## Building from Base Class

### Extending `BaseGPTResearcher`

You can create custom tools by extending the `BaseGPTResearcher` class. Here's an example:

```python
from tool import WebGPTResearcher, LocalGPTResearcher, HybridGPTResearcher # This will be changed after successful PR

class CustomGPTResearcher(BaseGPTResearcher):
    name = ""
    description = ""  
    def __init__(self, report_type: ReportType = ReportType.RESEARCH):
        super().__init__(report_type=report_type, report_source="web")

    # Override or extend methods as needed (You need to implement `_run()` method, `_arun()` is optional)
```
API reference: (GPT Researcher tool)[link]

### Building CustomGPTResearcher

You can define a custom GPTR tool as shown below:

```python
import asyncio
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from gpt_researcher import GPTResearcher


class GPTRInput(BaseModel):
    """Input schema for the GPT-Researcher tool."""
    query: str = Field(description="The search query for the research")

class MyGPTResearcher(BaseTool):
    name: str = "custom_gpt_researcher"
    description: str = "Base tool for researching and producing detailed information about a topic or query using the internet."
    args_schema: Type[BaseModel] = GPTRInput

    async def get_report(self, query: str) -> str:
        try:
            researcher = GPTResearcher(
                query=query,
                report_type="research_report",
                report_source="web",
                verbose=False
            )
            await researcher.conduct_research()
            report = await researcher.write_report()
            return report
        except Exception as e:
            raise ValueError(f"Error generating report: {str(e)}")

    def _run(
            self, 
            query: str, 
            run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
        answer = asyncio.run(self.get_report(query=query))
        answer += "\n\n- By GPT-Makesh.\nThanks for reading!"
        return answer

my_researcher = MyGPTResearcher()
report = my_researcher.invoke({"query": "What are the recent advancements in AI?"})
print(report)
```

### Off-the-Shelf Usage

Alternatively, you can directly use the provided tools without modification off-the-shelf.

```python
from tool import WebGPTResearcher, LocalGPTResearcher, HybridGPTResearcher # This will be changed after successful PR

# Use LocalGPTResearcher
researcher_local = LocalGPTResearcher(report_type="research_report")
report = researcher_local.invoke({'query':"What can you tell about Higgs?"})

# Use WebGPTResearcher
researcher_web = WebGPTResearcher(report_type="research_report")
report = researcher_web.invoke({'query':"What are the latest advancements in AI?"})

# Use HybridGPTResearcher
researcher_hybrid = HybridGPTResearcher(report_type="research_report")
report = researcher_hybrid.invoke({'query':"What are the latest advancements in AI?"})
```

---

## Performance Considerations

- **Time and Cost Estimates:** The tools are optimized for performance and cost, using models like `gpt-4o-mini` and `gpt-4o` (128K context) only when necessary. The average research task takes about 3 minutes and costs approximately $0.005.
- **Usage Limitations:** Be aware of potential limitations such as maximum query length and data size when working with large local datasets or complex web queries.

---

## Links and References

- **GPT-Researcher Documentation:** For a comprehensive guide, visit [GPT-Researcher Documentation](https://docs.gptr.dev/docs/gpt-researcher/introduction).
- **GitHub Repository:** Explore the code and contribute at [GPT-Researcher on GitHub](https://github.com/assafelovic/gpt-researcher).

---

## Contribution Guide

We welcome contributions to improve and extend the GPT-Researcher tools. Visit the [GitHub repository](https://github.com/assafelovic/gpt-researcher) to get started with contributing.

---

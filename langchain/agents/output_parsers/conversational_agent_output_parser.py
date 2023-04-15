import ast
import re
from typing import List, Any

from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseOutputParser, BaseLanguageModel, Document
from langchain.tools import BaseTool
from langchain.vectorstores import Chroma


class ConversationalAgentOutputParser(BaseOutputParser):
    llm: BaseLanguageModel = None
    tools: List[BaseTool] = None
    FORMAT_INSTRUCTIONS = ""
    ai_prefix: str = "AI"

    def __init__(self, llm, tools, FORMAT_INSTRUCTIONS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.tools = tools
        self.FORMAT_INSTRUCTIONS = FORMAT_INSTRUCTIONS

    def get_format_instructions(self) -> str:
        return self.FORMAT_INSTRUCTIONS

    def parse(self, llm_output: str) -> Any:
        try:
            if f"{self.ai_prefix}:" in llm_output:
                return self.ai_prefix, llm_output.split(f"{self.ai_prefix}:")[-1].strip()
            regex = r"Action: (.*?)[\n]*Action Input: (.*)"
            match = re.search(regex, llm_output)
            if match:
                action = match.group(1).strip()
                action_input = match.group(2).strip(" ").strip('"')
            else:
                embeddings = OpenAIEmbeddings()
                docsearch = Chroma.from_documents([Document(page_content=llm_output)], embeddings)
                qa = RetrievalQA.from_chain_type(self.llm, retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
                tools_str = ",".join(["\"" + tool.name + "\"" for tool in self.tools])
                action_rsp = qa.run(
                    F"Extract the Action. Choices: [{tools_str}]. Put your answer in this textbox: []. Do not add additional text. Example Answer: [\"Example Action\"]")
                action_input_rsp = qa.run(
                    "Extract the Action Input. Hint: Should be relevant to the action. Put your answer in this textbox: []. Do not add additional text. Example Answers: [\"Example Action Input\"] or [\"\"] (for empty action inputs)")

                action = ast.literal_eval(action_rsp)[0]
                action_input = ast.literal_eval(action_input_rsp)[0]
        except Exception:
            raise ValueError(f"Could not parse LLM output: {llm_output}")

        return action, action_input


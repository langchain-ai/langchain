import ast
import json
from typing import List, Any

from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseLanguageModel, BaseOutputParser, Document
from langchain.tools import BaseTool
from langchain.vectorstores import Chroma


class ChatAgentOutputParser(BaseOutputParser):
    llm: BaseLanguageModel = None
    tools: List[BaseTool] = None
    FORMAT_INSTRUCTIONS = ""

    def __init__(self, llm, tools, FORMAT_INSTRUCTIONS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.tools = tools
        self.FORMAT_INSTRUCTIONS = FORMAT_INSTRUCTIONS

    def get_format_instructions(self) -> str:
        return self.FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Any:
        try:
            parsed_text = self.semantic_parse(text)
        except Exception:
            raise ValueError(f"Could not parse LLM output: {text}")

        return parsed_text

    def semantic_parse(self, text: str):
        try:
            cleaned_output = text.strip()

            if "```json" in cleaned_output:
                _, cleaned_output = cleaned_output.split("```json")
            if "```" in cleaned_output:
                cleaned_output, _ = cleaned_output.split("```")
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[len("```json"):]
            if cleaned_output.startswith("```"):
                cleaned_output = cleaned_output[len("```"):]
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[: -len("```")]
            cleaned_output = cleaned_output.strip()

            response = json.loads(cleaned_output)

        except Exception:
            embeddings = OpenAIEmbeddings()
            docsearch = Chroma.from_documents([Document(page_content=cleaned_output)], embeddings)
            qa = RetrievalQA.from_chain_type(self.llm, retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
            tools_str = ",".join(["\"" + tool.name + "\"" for tool in self.tools])
            action_rsp = qa.run(
                F"Extract the action. Choices: [{tools_str}]. Put your answer in this textbox: []. Do not add additional text. Example Answer: [\"Example Action\"]")
            action_input_rsp = qa.run(
                "Extract the action_input. Hint: Should be relevant to the action. Put your answer in this textbox: []. Do not add additional text. Example Answers: [\"Example Action Input\"] or [\"\"] (for empty action inputs)")

            action = ast.literal_eval(action_rsp)[0]
            action_input = ast.literal_eval(action_input_rsp)[0]

            response = {"action": action, "action_input": action_input}

        return response["action"], response["action_input"]



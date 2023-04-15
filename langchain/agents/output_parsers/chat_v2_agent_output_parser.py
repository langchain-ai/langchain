import ast
import json
from typing import List, Union

from langchain.agents import AgentOutputParser
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseLanguageModel, AgentAction, AgentFinish, Document
from langchain.tools import BaseTool
from langchain.vectorstores import Chroma


class ChatOutputParser(AgentOutputParser):
    llm: BaseLanguageModel = None
    tools: List[BaseTool] = None
    FORMAT_INSTRUCTIONS = ""

    def __init__(self, llm, tools, FORMAT_INSTRUCTIONS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.tools = tools
        self.FORMAT_INSTRUCTIONS = FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            if "Final Answer:" in text:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": text.split("Final Answer:")[-1].strip()},
                    log=text,
                )
            _, llm_output, _ = text.split("```")

            if self.is_json_parsable(llm_output):
                response = json.loads(llm_output)
                action = response["action"]
                action_input = response["action_input"]
            else:
                embeddings = OpenAIEmbeddings()
                docsearch = Chroma.from_documents([Document(page_content=llm_output)], embeddings)
                qa = RetrievalQA.from_chain_type(self.llm, retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
                tools_str = ",".join(["\"" + tool.name + "\"" for tool in self.tools])
                action_rsp = qa.run(
                    F"Extract the action. Choices: [{tools_str}]. Put your answer in this textbox: []. Do not add additional text. Example Answer: [\"Example Action\"]")
                action_input_rsp = qa.run(
                    "Extract the action_input. Hint: Should be relevant to the action. Put your answer in this textbox: []. Do not add additional text. Example Answers: [\"Example Action Input\"] or [\"\"] (for empty action inputs)")

                action = ast.literal_eval(action_rsp)[0]
                action_input = ast.literal_eval(action_input_rsp)[0]

        except Exception:
            raise ValueError(f"Could not parse LLM output: {text}")

        return AgentAction(tool=action, tool_input=action_input, log=text)


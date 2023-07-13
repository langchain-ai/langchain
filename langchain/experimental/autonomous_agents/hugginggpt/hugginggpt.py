from transformers import load_tool
from langchain.experimental.autonomous_agents.hugginggpt.task_planner import load_chat_planner
from langchain.experimental.autonomous_agents.hugginggpt.task_executor import TaskExecutor
from langchain.experimental.autonomous_agents.hugginggpt.repsonse_generator import load_response_generator

class HuggingGPT:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.chat_planner = load_chat_planner(llm)
        self.response_generator = load_response_generator(llm)
        self.task_executor = None
    
    def run(self, input):
        plan = self.chat_planner.plan(inputs={"input": input, "hf_tools": self.tools})
        self.task_executor = TaskExecutor(plan)
        self.task_executor.run()
        response = self.response_generator.generate({"task_execution": self.task_executor})
        return response

if __name__ == "__main__":
    from langchain.llms import OpenAI
    llm = OpenAI(model_name="gpt-3.5-turbo")
    hf_tools = [load_tool(tool_name) for tool_name in [
        "document-question-answering", 
        "image-captioning", 
        "image-question-answering", 
        "image-segmentation", 
        "speech-to-text", 
        "summarization", 
        "text-classification", 
        "text-question-answering", 
        "translation", 
        "huggingface-tools/text-to-image", 
        "huggingface-tools/text-to-video", 
        "text-to-speech", 
        "huggingface-tools/text-download", 
        "huggingface-tools/image-transformation"
        ]
    ]
    agent = HuggingGPT(llm, hf_tools)
    agent.chat_planner.llm_chain.verbose = True
    # output = agent.run("translate the sentence 'a boy is running' into Chinese one")
    output = agent.run("please show me a video and an image of (based on the text) 'a boy is running' and dub it")
    print(output)
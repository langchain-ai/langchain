from io import StringIO
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents.tools import Tool
from langchain.llms import OpenAI
from langchain.utilities import PythonREPL
      
llm = OpenAI(temperature=0.0)
python_repl_util = Tool(
        "Python REPL",
        PythonREPL().run,
        """A Python shell. Use this to execute python commands. 
        Input should be a valid python command.
        If you expect output it should be printed out.""",
    )
tools = [python_repl_util]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
agent.run("""Understand, write a single neuron neural network in PyTorch.
Take synthetic data for y=2x. Train for 1000 epochs and print every 100 epochs.
Return prediction for x = 5""")

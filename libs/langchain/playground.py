import os

from e2b.templates.data_analysis import Artifact

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import E2BDataAnalysisTool

os.environ["E2B_API_KEY"] = "e2b_d5db0b24b13f426a698503143a06f53e51302059"
os.environ["OPENAI_API_KEY"] = "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"


# Artifacts are charts created by matplotlib when `plt.show()` is called
def save_artifact(artifact: Artifact):
    # Matplotlib charts created by `plt.show()`
    # We return them as `bytes` and leave it up to the user to display them (on frontend, for example)
    file = artifact.download()
    basename = os.path.basename(artifact.name)
    with open(f"./charts/{basename}", "wb") as f:
        f.write(file)


# e2b_data_analysis_tool = E2BDataAnalysisTool(api_key=os.environ["E2B_API_KEY"])
e2b_data_analysis_tool = E2BDataAnalysisTool(
    env_vars={},
    on_stdout=print,
    on_stderr=print,
    on_artifact=save_artifact,
)


with open(
    "/Users/mlejva/Developer/e2b-cookbook/guides/langchain_data_analysis/netflix.csv",
    "rb",
) as f:
    e2b_data_analysis_tool.upload_file(
        file=f,
        description="Data about Netflix tv shows",
    )


tools = [e2b_data_analysis_tool.as_tool()]


llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)


agent.run(
    "What are the 5 longest movies on netflix released between 2000 and 2010? Create a chart with their lengths."
)

# End the sessions once done.
e2b_data_analysis_tool.close()

import json
import urllib.request
import uuid

import pytest

from langchain.agents import AgentType, initialize_agent
from langchain.agents._agent_toolkits.ainetwork.toolkit import AINetworkToolkit
from langchain.chat_models import ChatOpenAI
from langchain.tools.ainetwork.utils import authenticate


@pytest.fixture(scope="module")
def get_faucet(address):
    with urllib.request.urlopen(
        f"http://faucet.ainetwork.ai/api/test/{address}/"
    ) as response:
        status_code = response.getcode()
    return status_code == 200


def test_ainetwork_toolkit() -> None:
    toolkit = AINetworkToolkit(network="testnet")
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    agent = initialize_agent(
        tools=toolkit.get_tools(),
        llm=llm,
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
    )
    ain = toolkit.get_tools()[0].interface
    self_address = ain.wallet.defaultAccount.address
    co_address = "0x6813Eb9362372EEF6200f3b1dbC3f819671cBA69"

    # Test creating an app
    app_name = f"_langchain_test_{str(uuid.uuid1()).replace('-', '_')}"
    agent.run(f"""Create app {app_name}""")
    assert ...

    # Test reading owner config
    agent.run(f"""Read owner config of /apps/{app_name}/DB .""")
    assert ...

    # Test granting owner config
    agent.run(
        f"""Grant owner authority to {co_address} for edit write rule permission of /apps/{app_name}/DB_co ."""
    )
    assert ...

    # Test reading owner config
    agent.run("""Read owner config of /apps/{app_name}/DB_co .""")
    assert ...  # check owner {co_address} exist

    # Test reading owner config
    agent.run("""Read owner config of /apps/{app_name}/DB .""")
    assert ...  # check owner {self_address} exist

    # Test reading a value
    agent.run(f"""Read value in /apps/{app_name}/DB""")
    assert ...  # empty

    # Test writing a value
    agent.run(f"""Write value {{1: 1904, 2: 43}} in /apps/{app_name}/DB""")
    assert ...

    # Test reading a value
    agent.run(f"""Read value in /apps/{app_name}/DB""")
    assert ...  # check value

    # Test reading a rule
    agent.run(f"""Read write rule of app {app_name} .""")
    assert ...  # check rule that self_address exists

    # Test sending AIN
    if get_faucet(self_address):
        agent.run(f"""Send 10 AIN to {co_address}""")
        assert ...  # check send success

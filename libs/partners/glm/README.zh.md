#  <img height="30" width="30" src="docs/img/MetaGLM.png"> 🔗 LangChain-GLM


## 项目介绍
本项目通过langchain的基础组件，实现了完整的支持智能体和相关任务架构。底层采用智谱AI的最新的 `GLM-4 All Tools`, 通过智谱AI的API接口，
能够自主理解用户的意图，规划复杂的指令，并能够调用一个或多个工具（例如网络浏览器、Python解释器和文本到图像模型）以完成复杂的任务。

![all_tools.png](docs/img/all_tools.png)

> 图｜GLM-4 All Tools 和定制 GLMs（智能体）的整体流程。

## 项目结构

| 包路径                                                       | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [agent_toolkits](https://github.com/MetaGLM/langchain-zhipuai/tree/main/langchain_glm/agent_toolkits) | 平台工具AdapterAllTool适配器， 是一个用于为各种工具提供统一接口的平台适配器工具，目的是在不同平台上实现无缝集成和执行。该工具通过适配特定的平台参数，确保兼容性和一致的输出。 |
| [agents](https://github.com/MetaGLM/langchain-zhipuai/tree/main/langchain_glm/agents) | 定义AgentExecutor的输入、输出、智能体会话、工具参数、工具执行策略的封装 |
| [callbacks](https://github.com/MetaGLM/langchain-zhipuai/tree/main/langchain_glm/callbacks) | 抽象AgentExecutor过程中的一些交互事件，通过事件展示信息      |
| [chat_models](https://github.com/MetaGLM/langchain-zhipuai/tree/main/langchain_glm/chat_models) | zhipuai sdk的封装层，提供langchain的BaseChatModel集成，格式化输入输出为消息体 |
| [embeddings](https://github.com/MetaGLM/langchain-zhipuai/tree/main/langchain_glm/embeddings) | zhipuai sdk的封装层，提供langchain的Embeddings集成           |
| [utils](https://github.com/MetaGLM/langchain-zhipuai/tree/main/langchain_glm/utils) | 一些会话工具                                                 |


## 快速使用

- 从 repo 安装
https://github.com/MetaGLM/langchain-glm/releases
- 直接使用pip源码安装
pip install git+https://github.com/MetaGLM/langchain-glm.git -v
- 从pypi安装
pip install langchain-glm

> 使用前请设置环境变量`ZHIPUAI_API_KEY`，值为智谱AI的API Key。
 

#### 工具使用
- Set environment variables
```python
import getpass
import os

os.environ["ZHIPUAI_API_KEY"] = getpass.getpass()

```
```python
from langchain_glm import ChatZhipuAI
llm = ChatZhipuAI(model="glm-4")
```


- 定义一些示例工具：
```python
from langchain_core.tools import tool

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int

@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent
```
- 构建chain
绑定工具到语言模型并调用：
```python
from operator import itemgetter
from typing import Dict, List, Union

from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)

tools = [multiply, exponentiate, add]
llm_with_tools = llm.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}


def call_tools(msg: AIMessage) -> Runnable:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls


chain = llm_with_tools | call_tools
```

- 调用chain
```python
chain.invoke(
    "What's 23 times 7, and what's five times 18 and add a million plus a billion and cube thirty-seven"
)
```

#### 代码解析使用示例


- 创建一个代理执行器
我们的glm-4-alltools的模型提供了平台工具，通过ZhipuAIAllToolsRunnable，你可以非常方便的设置了一个执行器来运行多个工具。
 
code_interpreter:使用`sandbox`指定代码沙盒环境，
    默认 = auto，即自动调用沙盒环境执行代码。 
    设置 sandbox = none，不启用沙盒环境。

web_browser:使用`web_browser`指定浏览器工具。
drawing_tool:使用`drawing_tool`指定绘图工具。

```python

from langchain_glm.agents.zhipuai_all_tools import ZhipuAIAllToolsRunnable
agent_executor = ZhipuAIAllToolsRunnable.create_agent_executor(
    model_name="glm-4-alltools",
    tools=[
        {"type": "code_interpreter", "code_interpreter": {"sandbox": "none"}},
        {"type": "web_browser"},
        {"type": "drawing_tool"},
        multiply, exponentiate, add
    ],
)

```


- 执行agent_executor并打印结果
这部分使用代理来运行一个Shell命令，并在结果出现时打印出来。它检查结果的类型并打印相关信息。
这个invoke返回一个异步迭代器，可以用来处理代理的输出。
你可以多次调用invoke方法，每次调用都会返回一个新的迭代器。
ZhipuAIAllToolsRunnable会自动处理状态保存和恢复，一些状态信息会被保存实例中
你可以通过callback属性获取intermediate_steps的状态信息。
```python
from langchain_glm.agents.zhipuai_all_tools.base import (
    AllToolsAction, 
    AllToolsActionToolEnd,
    AllToolsActionToolStart,
    AllToolsFinish, 
    AllToolsLLMStatus
)
from langchain_glm.callbacks.agent_callback_handler import AgentStatus


chat_iterator = agent_executor.invoke(
    chat_input="看下本地文件有哪些，告诉我你用的是什么文件,查看当前目录"
)
async for item in chat_iterator:
    if isinstance(item, AllToolsAction):
        print("AllToolsAction:" + str(item.to_json()))
    elif isinstance(item, AllToolsFinish):
        print("AllToolsFinish:" + str(item.to_json()))
    elif isinstance(item, AllToolsActionToolStart):
        print("AllToolsActionToolStart:" + str(item.to_json()))
    elif isinstance(item, AllToolsActionToolEnd):
        print("AllToolsActionToolEnd:" + str(item.to_json()))
    elif isinstance(item, AllToolsLLMStatus):
        if item.status == AgentStatus.llm_end:
            print("llm_end:" + item.text)
```

## 集成demo
我们提供了一个集成的demo，可以直接运行，查看效果。
- 安装依赖
```shell
fastapi = "~0.109.2"
sse_starlette = "~1.8.2" 
uvicorn = ">=0.27.0.post1"
# webui
streamlit = "1.34.0"
streamlit-option-menu = "0.3.12"
streamlit-antd-components = "0.3.1"
streamlit-chatbox = "1.1.12.post4"
streamlit-modal = "0.1.0"
streamlit-aggrid = "1.0.5"
streamlit-extras = "0.4.2"
```

- 运行后端服务[server.py](tests/assistant/server/server.py)
```shell
python tests/assistant/server/server.py
```

- 运行前端服务[test_chat.py](tests/assistant/test_chat.py)
```shell
python tests/assistant/start_chat.py
```

> 展示


https://github.com/MetaGLM/langchain-zhipuai/assets/16206043/06863f9c-cd03-4a74-b76a-daa315718104

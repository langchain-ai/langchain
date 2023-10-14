# Debugging

If you're building with LLMs, at some point something will break, and you'll need to debug. A model call will fail, or the model output will be misformatted, or there will be some nested model calls and it won't be clear where along the way an incorrect output was created.

Here are a few different tools and functionalities to aid in debugging.



## Tracing

Platforms with tracing capabilities like [LangSmith](/docs/guides/langsmith/) and [WandB](/docs/integrations/providers/wandb_tracing) are the most comprehensive solutions for debugging. These platforms make it easy to not only log and visualize LLM apps, but also to actively debug, test and refine them.

For anyone building production-grade LLM applications, we highly recommend using a platform like this.

![LangSmith run](/img/run_details.png)

## `set_debug` and `set_verbose`

If you're prototyping in Jupyter Notebooks or running Python scripts, it can be helpful to print out the intermediate steps of a Chain run. 

There are a number of ways to enable printing at varying degrees of verbosity.

Let's suppose we have a simple agent, and want to visualize the actions it takes and tool outputs it receives. Without any debugging, here's what we see:


```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
tools = load_tools(["ddg-search", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```


```python
agent.run("Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?")
```

<CodeOutputBlock lang="python">

```
    'The director of the 2023 film Oppenheimer is Christopher Nolan and he is approximately 19345 days old in 2023.'
```

</CodeOutputBlock>

### `set_debug(True)`

Setting the global `debug` flag will cause all LangChain components with callback support (chains, models, agents, tools, retrievers) to print the inputs they receive and outputs they generate. This is the most verbose setting and will fully log raw inputs and outputs.


```python
from langchain.globals import set_debug

set_debug(True)

agent.run("Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?")
```

<details> <summary>Console output</summary>

<CodeOutputBlock lang="python">

```
    [chain/start] [1:RunTypeEnum.chain:AgentExecutor] Entering Chain run with input:
    {
      "input": "Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?"
    }
    [chain/start] [1:RunTypeEnum.chain:AgentExecutor > 2:RunTypeEnum.chain:LLMChain] Entering Chain run with input:
    {
      "input": "Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?",
      "agent_scratchpad": "",
      "stop": [
        "\nObservation:",
        "\n\tObservation:"
      ]
    }
    [llm/start] [1:RunTypeEnum.chain:AgentExecutor > 2:RunTypeEnum.chain:LLMChain > 3:RunTypeEnum.llm:ChatOpenAI] Entering LLM run with input:
    {
      "prompts": [
        "Human: Answer the following questions as best you can. You have access to the following tools:\n\nduckduckgo_search: A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.\nCalculator: Useful for when you need to answer questions about math.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [duckduckgo_search, Calculator]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?\nThought:"
      ]
    }
    [llm/end] [1:RunTypeEnum.chain:AgentExecutor > 2:RunTypeEnum.chain:LLMChain > 3:RunTypeEnum.llm:ChatOpenAI] [5.53s] Exiting LLM run with output:
    {
      "generations": [
        [
          {
            "text": "I need to find out who directed the 2023 film Oppenheimer and their age. Then, I need to calculate their age in days. I will use DuckDuckGo to find out the director and their age.\nAction: duckduckgo_search\nAction Input: \"Director of the 2023 film Oppenheimer and their age\"",
            "generation_info": {
              "finish_reason": "stop"
            },
            "message": {
              "lc": 1,
              "type": "constructor",
              "id": [
                "langchain",
                "schema",
                "messages",
                "AIMessage"
              ],
              "kwargs": {
                "content": "I need to find out who directed the 2023 film Oppenheimer and their age. Then, I need to calculate their age in days. I will use DuckDuckGo to find out the director and their age.\nAction: duckduckgo_search\nAction Input: \"Director of the 2023 film Oppenheimer and their age\"",
                "additional_kwargs": {}
              }
            }
          }
        ]
      ],
      "llm_output": {
        "token_usage": {
          "prompt_tokens": 206,
          "completion_tokens": 71,
          "total_tokens": 277
        },
        "model_name": "gpt-4"
      },
      "run": null
    }
    [chain/end] [1:RunTypeEnum.chain:AgentExecutor > 2:RunTypeEnum.chain:LLMChain] [5.53s] Exiting Chain run with output:
    {
      "text": "I need to find out who directed the 2023 film Oppenheimer and their age. Then, I need to calculate their age in days. I will use DuckDuckGo to find out the director and their age.\nAction: duckduckgo_search\nAction Input: \"Director of the 2023 film Oppenheimer and their age\""
    }
    [tool/start] [1:RunTypeEnum.chain:AgentExecutor > 4:RunTypeEnum.tool:duckduckgo_search] Entering Tool run with input:
    "Director of the 2023 film Oppenheimer and their age"
    [tool/end] [1:RunTypeEnum.chain:AgentExecutor > 4:RunTypeEnum.tool:duckduckgo_search] [1.51s] Exiting Tool run with output:
    "Capturing the mad scramble to build the first atomic bomb required rapid-fire filming, strict set rules and the construction of an entire 1940s western town. By Jada Yuan. July 19, 2023 at 5:00 a ... In Christopher Nolan's new film, "Oppenheimer," Cillian Murphy stars as J. Robert Oppenheimer, the American physicist who oversaw the Manhattan Project in Los Alamos, N.M. Universal Pictures... Oppenheimer: Directed by Christopher Nolan. With Cillian Murphy, Emily Blunt, Robert Downey Jr., Alden Ehrenreich. The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb. Christopher Nolan goes deep on 'Oppenheimer,' his most 'extreme' film to date. By Kenneth Turan. July 11, 2023 5 AM PT. For Subscribers. Christopher Nolan is photographed in Los Angeles ... Oppenheimer is a 2023 epic biographical thriller film written and directed by Christopher Nolan.It is based on the 2005 biography American Prometheus by Kai Bird and Martin J. Sherwin about J. Robert Oppenheimer, a theoretical physicist who was pivotal in developing the first nuclear weapons as part of the Manhattan Project and thereby ushering in the Atomic Age."
    [chain/start] [1:RunTypeEnum.chain:AgentExecutor > 5:RunTypeEnum.chain:LLMChain] Entering Chain run with input:
    {
      "input": "Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?",
      "agent_scratchpad": "I need to find out who directed the 2023 film Oppenheimer and their age. Then, I need to calculate their age in days. I will use DuckDuckGo to find out the director and their age.\nAction: duckduckgo_search\nAction Input: \"Director of the 2023 film Oppenheimer and their age\"\nObservation: Capturing the mad scramble to build the first atomic bomb required rapid-fire filming, strict set rules and the construction of an entire 1940s western town. By Jada Yuan. July 19, 2023 at 5:00 a ... In Christopher Nolan's new film, \"Oppenheimer,\" Cillian Murphy stars as J. Robert Oppenheimer, the American physicist who oversaw the Manhattan Project in Los Alamos, N.M. Universal Pictures... Oppenheimer: Directed by Christopher Nolan. With Cillian Murphy, Emily Blunt, Robert Downey Jr., Alden Ehrenreich. The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb. Christopher Nolan goes deep on 'Oppenheimer,' his most 'extreme' film to date. By Kenneth Turan. July 11, 2023 5 AM PT. For Subscribers. Christopher Nolan is photographed in Los Angeles ... Oppenheimer is a 2023 epic biographical thriller film written and directed by Christopher Nolan.It is based on the 2005 biography American Prometheus by Kai Bird and Martin J. Sherwin about J. Robert Oppenheimer, a theoretical physicist who was pivotal in developing the first nuclear weapons as part of the Manhattan Project and thereby ushering in the Atomic Age.\nThought:",
      "stop": [
        "\nObservation:",
        "\n\tObservation:"
      ]
    }
    [llm/start] [1:RunTypeEnum.chain:AgentExecutor > 5:RunTypeEnum.chain:LLMChain > 6:RunTypeEnum.llm:ChatOpenAI] Entering LLM run with input:
    {
      "prompts": [
        "Human: Answer the following questions as best you can. You have access to the following tools:\n\nduckduckgo_search: A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.\nCalculator: Useful for when you need to answer questions about math.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [duckduckgo_search, Calculator]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?\nThought:I need to find out who directed the 2023 film Oppenheimer and their age. Then, I need to calculate their age in days. I will use DuckDuckGo to find out the director and their age.\nAction: duckduckgo_search\nAction Input: \"Director of the 2023 film Oppenheimer and their age\"\nObservation: Capturing the mad scramble to build the first atomic bomb required rapid-fire filming, strict set rules and the construction of an entire 1940s western town. By Jada Yuan. July 19, 2023 at 5:00 a ... In Christopher Nolan's new film, \"Oppenheimer,\" Cillian Murphy stars as J. Robert Oppenheimer, the American physicist who oversaw the Manhattan Project in Los Alamos, N.M. Universal Pictures... Oppenheimer: Directed by Christopher Nolan. With Cillian Murphy, Emily Blunt, Robert Downey Jr., Alden Ehrenreich. The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb. Christopher Nolan goes deep on 'Oppenheimer,' his most 'extreme' film to date. By Kenneth Turan. July 11, 2023 5 AM PT. For Subscribers. Christopher Nolan is photographed in Los Angeles ... Oppenheimer is a 2023 epic biographical thriller film written and directed by Christopher Nolan.It is based on the 2005 biography American Prometheus by Kai Bird and Martin J. Sherwin about J. Robert Oppenheimer, a theoretical physicist who was pivotal in developing the first nuclear weapons as part of the Manhattan Project and thereby ushering in the Atomic Age.\nThought:"
      ]
    }
    [llm/end] [1:RunTypeEnum.chain:AgentExecutor > 5:RunTypeEnum.chain:LLMChain > 6:RunTypeEnum.llm:ChatOpenAI] [4.46s] Exiting LLM run with output:
    {
      "generations": [
        [
          {
            "text": "The director of the 2023 film Oppenheimer is Christopher Nolan. Now I need to find out his age.\nAction: duckduckgo_search\nAction Input: \"Christopher Nolan age\"",
            "generation_info": {
              "finish_reason": "stop"
            },
            "message": {
              "lc": 1,
              "type": "constructor",
              "id": [
                "langchain",
                "schema",
                "messages",
                "AIMessage"
              ],
              "kwargs": {
                "content": "The director of the 2023 film Oppenheimer is Christopher Nolan. Now I need to find out his age.\nAction: duckduckgo_search\nAction Input: \"Christopher Nolan age\"",
                "additional_kwargs": {}
              }
            }
          }
        ]
      ],
      "llm_output": {
        "token_usage": {
          "prompt_tokens": 550,
          "completion_tokens": 39,
          "total_tokens": 589
        },
        "model_name": "gpt-4"
      },
      "run": null
    }
    [chain/end] [1:RunTypeEnum.chain:AgentExecutor > 5:RunTypeEnum.chain:LLMChain] [4.46s] Exiting Chain run with output:
    {
      "text": "The director of the 2023 film Oppenheimer is Christopher Nolan. Now I need to find out his age.\nAction: duckduckgo_search\nAction Input: \"Christopher Nolan age\""
    }
    [tool/start] [1:RunTypeEnum.chain:AgentExecutor > 7:RunTypeEnum.tool:duckduckgo_search] Entering Tool run with input:
    "Christopher Nolan age"
    [tool/end] [1:RunTypeEnum.chain:AgentExecutor > 7:RunTypeEnum.tool:duckduckgo_search] [1.33s] Exiting Tool run with output:
    "Christopher Edward Nolan CBE (born 30 July 1970) is a British and American filmmaker. Known for his Hollywood blockbusters with complex storytelling, Nolan is considered a leading filmmaker of the 21st century. His films have grossed $5 billion worldwide. The recipient of many accolades, he has been nominated for five Academy Awards, five BAFTA Awards and six Golden Globe Awards. July 30, 1970 (age 52) London England Notable Works: "Dunkirk" "Tenet" "The Prestige" See all related content → Recent News Jul. 13, 2023, 11:11 AM ET (AP) Cillian Murphy, playing Oppenheimer, finally gets to lead a Christopher Nolan film July 11, 2023 5 AM PT For Subscribers Christopher Nolan is photographed in Los Angeles. (Joe Pugliese / For The Times) This is not the story I was supposed to write. Oppenheimer director Christopher Nolan, Cillian Murphy, Emily Blunt and Matt Damon on the stakes of making a three-hour, CGI-free summer film. Christopher Nolan, the director behind such films as "Dunkirk," "Inception," "Interstellar," and the "Dark Knight" trilogy, has spent the last three years living in Oppenheimer's world, writing ..."
    [chain/start] [1:RunTypeEnum.chain:AgentExecutor > 8:RunTypeEnum.chain:LLMChain] Entering Chain run with input:
    {
      "input": "Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?",
      "agent_scratchpad": "I need to find out who directed the 2023 film Oppenheimer and their age. Then, I need to calculate their age in days. I will use DuckDuckGo to find out the director and their age.\nAction: duckduckgo_search\nAction Input: \"Director of the 2023 film Oppenheimer and their age\"\nObservation: Capturing the mad scramble to build the first atomic bomb required rapid-fire filming, strict set rules and the construction of an entire 1940s western town. By Jada Yuan. July 19, 2023 at 5:00 a ... In Christopher Nolan's new film, \"Oppenheimer,\" Cillian Murphy stars as J. Robert Oppenheimer, the American physicist who oversaw the Manhattan Project in Los Alamos, N.M. Universal Pictures... Oppenheimer: Directed by Christopher Nolan. With Cillian Murphy, Emily Blunt, Robert Downey Jr., Alden Ehrenreich. The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb. Christopher Nolan goes deep on 'Oppenheimer,' his most 'extreme' film to date. By Kenneth Turan. July 11, 2023 5 AM PT. For Subscribers. Christopher Nolan is photographed in Los Angeles ... Oppenheimer is a 2023 epic biographical thriller film written and directed by Christopher Nolan.It is based on the 2005 biography American Prometheus by Kai Bird and Martin J. Sherwin about J. Robert Oppenheimer, a theoretical physicist who was pivotal in developing the first nuclear weapons as part of the Manhattan Project and thereby ushering in the Atomic Age.\nThought:The director of the 2023 film Oppenheimer is Christopher Nolan. Now I need to find out his age.\nAction: duckduckgo_search\nAction Input: \"Christopher Nolan age\"\nObservation: Christopher Edward Nolan CBE (born 30 July 1970) is a British and American filmmaker. Known for his Hollywood blockbusters with complex storytelling, Nolan is considered a leading filmmaker of the 21st century. His films have grossed $5 billion worldwide. The recipient of many accolades, he has been nominated for five Academy Awards, five BAFTA Awards and six Golden Globe Awards. July 30, 1970 (age 52) London England Notable Works: \"Dunkirk\" \"Tenet\" \"The Prestige\" See all related content → Recent News Jul. 13, 2023, 11:11 AM ET (AP) Cillian Murphy, playing Oppenheimer, finally gets to lead a Christopher Nolan film July 11, 2023 5 AM PT For Subscribers Christopher Nolan is photographed in Los Angeles. (Joe Pugliese / For The Times) This is not the story I was supposed to write. Oppenheimer director Christopher Nolan, Cillian Murphy, Emily Blunt and Matt Damon on the stakes of making a three-hour, CGI-free summer film. Christopher Nolan, the director behind such films as \"Dunkirk,\" \"Inception,\" \"Interstellar,\" and the \"Dark Knight\" trilogy, has spent the last three years living in Oppenheimer's world, writing ...\nThought:",
      "stop": [
        "\nObservation:",
        "\n\tObservation:"
      ]
    }
    [llm/start] [1:RunTypeEnum.chain:AgentExecutor > 8:RunTypeEnum.chain:LLMChain > 9:RunTypeEnum.llm:ChatOpenAI] Entering LLM run with input:
    {
      "prompts": [
        "Human: Answer the following questions as best you can. You have access to the following tools:\n\nduckduckgo_search: A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.\nCalculator: Useful for when you need to answer questions about math.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [duckduckgo_search, Calculator]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?\nThought:I need to find out who directed the 2023 film Oppenheimer and their age. Then, I need to calculate their age in days. I will use DuckDuckGo to find out the director and their age.\nAction: duckduckgo_search\nAction Input: \"Director of the 2023 film Oppenheimer and their age\"\nObservation: Capturing the mad scramble to build the first atomic bomb required rapid-fire filming, strict set rules and the construction of an entire 1940s western town. By Jada Yuan. July 19, 2023 at 5:00 a ... In Christopher Nolan's new film, \"Oppenheimer,\" Cillian Murphy stars as J. Robert Oppenheimer, the American physicist who oversaw the Manhattan Project in Los Alamos, N.M. Universal Pictures... Oppenheimer: Directed by Christopher Nolan. With Cillian Murphy, Emily Blunt, Robert Downey Jr., Alden Ehrenreich. The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb. Christopher Nolan goes deep on 'Oppenheimer,' his most 'extreme' film to date. By Kenneth Turan. July 11, 2023 5 AM PT. For Subscribers. Christopher Nolan is photographed in Los Angeles ... Oppenheimer is a 2023 epic biographical thriller film written and directed by Christopher Nolan.It is based on the 2005 biography American Prometheus by Kai Bird and Martin J. Sherwin about J. Robert Oppenheimer, a theoretical physicist who was pivotal in developing the first nuclear weapons as part of the Manhattan Project and thereby ushering in the Atomic Age.\nThought:The director of the 2023 film Oppenheimer is Christopher Nolan. Now I need to find out his age.\nAction: duckduckgo_search\nAction Input: \"Christopher Nolan age\"\nObservation: Christopher Edward Nolan CBE (born 30 July 1970) is a British and American filmmaker. Known for his Hollywood blockbusters with complex storytelling, Nolan is considered a leading filmmaker of the 21st century. His films have grossed $5 billion worldwide. The recipient of many accolades, he has been nominated for five Academy Awards, five BAFTA Awards and six Golden Globe Awards. July 30, 1970 (age 52) London England Notable Works: \"Dunkirk\" \"Tenet\" \"The Prestige\" See all related content → Recent News Jul. 13, 2023, 11:11 AM ET (AP) Cillian Murphy, playing Oppenheimer, finally gets to lead a Christopher Nolan film July 11, 2023 5 AM PT For Subscribers Christopher Nolan is photographed in Los Angeles. (Joe Pugliese / For The Times) This is not the story I was supposed to write. Oppenheimer director Christopher Nolan, Cillian Murphy, Emily Blunt and Matt Damon on the stakes of making a three-hour, CGI-free summer film. Christopher Nolan, the director behind such films as \"Dunkirk,\" \"Inception,\" \"Interstellar,\" and the \"Dark Knight\" trilogy, has spent the last three years living in Oppenheimer's world, writing ...\nThought:"
      ]
    }
    [llm/end] [1:RunTypeEnum.chain:AgentExecutor > 8:RunTypeEnum.chain:LLMChain > 9:RunTypeEnum.llm:ChatOpenAI] [2.69s] Exiting LLM run with output:
    {
      "generations": [
        [
          {
            "text": "Christopher Nolan was born on July 30, 1970, which makes him 52 years old in 2023. Now I need to calculate his age in days.\nAction: Calculator\nAction Input: 52*365",
            "generation_info": {
              "finish_reason": "stop"
            },
            "message": {
              "lc": 1,
              "type": "constructor",
              "id": [
                "langchain",
                "schema",
                "messages",
                "AIMessage"
              ],
              "kwargs": {
                "content": "Christopher Nolan was born on July 30, 1970, which makes him 52 years old in 2023. Now I need to calculate his age in days.\nAction: Calculator\nAction Input: 52*365",
                "additional_kwargs": {}
              }
            }
          }
        ]
      ],
      "llm_output": {
        "token_usage": {
          "prompt_tokens": 868,
          "completion_tokens": 46,
          "total_tokens": 914
        },
        "model_name": "gpt-4"
      },
      "run": null
    }
    [chain/end] [1:RunTypeEnum.chain:AgentExecutor > 8:RunTypeEnum.chain:LLMChain] [2.69s] Exiting Chain run with output:
    {
      "text": "Christopher Nolan was born on July 30, 1970, which makes him 52 years old in 2023. Now I need to calculate his age in days.\nAction: Calculator\nAction Input: 52*365"
    }
    [tool/start] [1:RunTypeEnum.chain:AgentExecutor > 10:RunTypeEnum.tool:Calculator] Entering Tool run with input:
    "52*365"
    [chain/start] [1:RunTypeEnum.chain:AgentExecutor > 10:RunTypeEnum.tool:Calculator > 11:RunTypeEnum.chain:LLMMathChain] Entering Chain run with input:
    {
      "question": "52*365"
    }
    [chain/start] [1:RunTypeEnum.chain:AgentExecutor > 10:RunTypeEnum.tool:Calculator > 11:RunTypeEnum.chain:LLMMathChain > 12:RunTypeEnum.chain:LLMChain] Entering Chain run with input:
    {
      "question": "52*365",
      "stop": [
        "```output"
      ]
    }
    [llm/start] [1:RunTypeEnum.chain:AgentExecutor > 10:RunTypeEnum.tool:Calculator > 11:RunTypeEnum.chain:LLMMathChain > 12:RunTypeEnum.chain:LLMChain > 13:RunTypeEnum.llm:ChatOpenAI] Entering LLM run with input:
    {
      "prompts": [
        "Human: Translate a math problem into a expression that can be executed using Python's numexpr library. Use the output of running this code to answer the question.\n\nQuestion: ${Question with math problem.}\n```text\n${single line mathematical expression that solves the problem}\n```\n...numexpr.evaluate(text)...\n```output\n${Output of running the code}\n```\nAnswer: ${Answer}\n\nBegin.\n\nQuestion: What is 37593 * 67?\n```text\n37593 * 67\n```\n...numexpr.evaluate(\"37593 * 67\")...\n```output\n2518731\n```\nAnswer: 2518731\n\nQuestion: 37593^(1/5)\n```text\n37593**(1/5)\n```\n...numexpr.evaluate(\"37593**(1/5)\")...\n```output\n8.222831614237718\n```\nAnswer: 8.222831614237718\n\nQuestion: 52*365"
      ]
    }
    [llm/end] [1:RunTypeEnum.chain:AgentExecutor > 10:RunTypeEnum.tool:Calculator > 11:RunTypeEnum.chain:LLMMathChain > 12:RunTypeEnum.chain:LLMChain > 13:RunTypeEnum.llm:ChatOpenAI] [2.89s] Exiting LLM run with output:
    {
      "generations": [
        [
          {
            "text": "```text\n52*365\n```\n...numexpr.evaluate(\"52*365\")...\n",
            "generation_info": {
              "finish_reason": "stop"
            },
            "message": {
              "lc": 1,
              "type": "constructor",
              "id": [
                "langchain",
                "schema",
                "messages",
                "AIMessage"
              ],
              "kwargs": {
                "content": "```text\n52*365\n```\n...numexpr.evaluate(\"52*365\")...\n",
                "additional_kwargs": {}
              }
            }
          }
        ]
      ],
      "llm_output": {
        "token_usage": {
          "prompt_tokens": 203,
          "completion_tokens": 19,
          "total_tokens": 222
        },
        "model_name": "gpt-4"
      },
      "run": null
    }
    [chain/end] [1:RunTypeEnum.chain:AgentExecutor > 10:RunTypeEnum.tool:Calculator > 11:RunTypeEnum.chain:LLMMathChain > 12:RunTypeEnum.chain:LLMChain] [2.89s] Exiting Chain run with output:
    {
      "text": "```text\n52*365\n```\n...numexpr.evaluate(\"52*365\")...\n"
    }
    [chain/end] [1:RunTypeEnum.chain:AgentExecutor > 10:RunTypeEnum.tool:Calculator > 11:RunTypeEnum.chain:LLMMathChain] [2.90s] Exiting Chain run with output:
    {
      "answer": "Answer: 18980"
    }
    [tool/end] [1:RunTypeEnum.chain:AgentExecutor > 10:RunTypeEnum.tool:Calculator] [2.90s] Exiting Tool run with output:
    "Answer: 18980"
    [chain/start] [1:RunTypeEnum.chain:AgentExecutor > 14:RunTypeEnum.chain:LLMChain] Entering Chain run with input:
    {
      "input": "Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?",
      "agent_scratchpad": "I need to find out who directed the 2023 film Oppenheimer and their age. Then, I need to calculate their age in days. I will use DuckDuckGo to find out the director and their age.\nAction: duckduckgo_search\nAction Input: \"Director of the 2023 film Oppenheimer and their age\"\nObservation: Capturing the mad scramble to build the first atomic bomb required rapid-fire filming, strict set rules and the construction of an entire 1940s western town. By Jada Yuan. July 19, 2023 at 5:00 a ... In Christopher Nolan's new film, \"Oppenheimer,\" Cillian Murphy stars as J. Robert Oppenheimer, the American physicist who oversaw the Manhattan Project in Los Alamos, N.M. Universal Pictures... Oppenheimer: Directed by Christopher Nolan. With Cillian Murphy, Emily Blunt, Robert Downey Jr., Alden Ehrenreich. The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb. Christopher Nolan goes deep on 'Oppenheimer,' his most 'extreme' film to date. By Kenneth Turan. July 11, 2023 5 AM PT. For Subscribers. Christopher Nolan is photographed in Los Angeles ... Oppenheimer is a 2023 epic biographical thriller film written and directed by Christopher Nolan.It is based on the 2005 biography American Prometheus by Kai Bird and Martin J. Sherwin about J. Robert Oppenheimer, a theoretical physicist who was pivotal in developing the first nuclear weapons as part of the Manhattan Project and thereby ushering in the Atomic Age.\nThought:The director of the 2023 film Oppenheimer is Christopher Nolan. Now I need to find out his age.\nAction: duckduckgo_search\nAction Input: \"Christopher Nolan age\"\nObservation: Christopher Edward Nolan CBE (born 30 July 1970) is a British and American filmmaker. Known for his Hollywood blockbusters with complex storytelling, Nolan is considered a leading filmmaker of the 21st century. His films have grossed $5 billion worldwide. The recipient of many accolades, he has been nominated for five Academy Awards, five BAFTA Awards and six Golden Globe Awards. July 30, 1970 (age 52) London England Notable Works: \"Dunkirk\" \"Tenet\" \"The Prestige\" See all related content → Recent News Jul. 13, 2023, 11:11 AM ET (AP) Cillian Murphy, playing Oppenheimer, finally gets to lead a Christopher Nolan film July 11, 2023 5 AM PT For Subscribers Christopher Nolan is photographed in Los Angeles. (Joe Pugliese / For The Times) This is not the story I was supposed to write. Oppenheimer director Christopher Nolan, Cillian Murphy, Emily Blunt and Matt Damon on the stakes of making a three-hour, CGI-free summer film. Christopher Nolan, the director behind such films as \"Dunkirk,\" \"Inception,\" \"Interstellar,\" and the \"Dark Knight\" trilogy, has spent the last three years living in Oppenheimer's world, writing ...\nThought:Christopher Nolan was born on July 30, 1970, which makes him 52 years old in 2023. Now I need to calculate his age in days.\nAction: Calculator\nAction Input: 52*365\nObservation: Answer: 18980\nThought:",
      "stop": [
        "\nObservation:",
        "\n\tObservation:"
      ]
    }
    [llm/start] [1:RunTypeEnum.chain:AgentExecutor > 14:RunTypeEnum.chain:LLMChain > 15:RunTypeEnum.llm:ChatOpenAI] Entering LLM run with input:
    {
      "prompts": [
        "Human: Answer the following questions as best you can. You have access to the following tools:\n\nduckduckgo_search: A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.\nCalculator: Useful for when you need to answer questions about math.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [duckduckgo_search, Calculator]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?\nThought:I need to find out who directed the 2023 film Oppenheimer and their age. Then, I need to calculate their age in days. I will use DuckDuckGo to find out the director and their age.\nAction: duckduckgo_search\nAction Input: \"Director of the 2023 film Oppenheimer and their age\"\nObservation: Capturing the mad scramble to build the first atomic bomb required rapid-fire filming, strict set rules and the construction of an entire 1940s western town. By Jada Yuan. July 19, 2023 at 5:00 a ... In Christopher Nolan's new film, \"Oppenheimer,\" Cillian Murphy stars as J. Robert Oppenheimer, the American physicist who oversaw the Manhattan Project in Los Alamos, N.M. Universal Pictures... Oppenheimer: Directed by Christopher Nolan. With Cillian Murphy, Emily Blunt, Robert Downey Jr., Alden Ehrenreich. The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb. Christopher Nolan goes deep on 'Oppenheimer,' his most 'extreme' film to date. By Kenneth Turan. July 11, 2023 5 AM PT. For Subscribers. Christopher Nolan is photographed in Los Angeles ... Oppenheimer is a 2023 epic biographical thriller film written and directed by Christopher Nolan.It is based on the 2005 biography American Prometheus by Kai Bird and Martin J. Sherwin about J. Robert Oppenheimer, a theoretical physicist who was pivotal in developing the first nuclear weapons as part of the Manhattan Project and thereby ushering in the Atomic Age.\nThought:The director of the 2023 film Oppenheimer is Christopher Nolan. Now I need to find out his age.\nAction: duckduckgo_search\nAction Input: \"Christopher Nolan age\"\nObservation: Christopher Edward Nolan CBE (born 30 July 1970) is a British and American filmmaker. Known for his Hollywood blockbusters with complex storytelling, Nolan is considered a leading filmmaker of the 21st century. His films have grossed $5 billion worldwide. The recipient of many accolades, he has been nominated for five Academy Awards, five BAFTA Awards and six Golden Globe Awards. July 30, 1970 (age 52) London England Notable Works: \"Dunkirk\" \"Tenet\" \"The Prestige\" See all related content → Recent News Jul. 13, 2023, 11:11 AM ET (AP) Cillian Murphy, playing Oppenheimer, finally gets to lead a Christopher Nolan film July 11, 2023 5 AM PT For Subscribers Christopher Nolan is photographed in Los Angeles. (Joe Pugliese / For The Times) This is not the story I was supposed to write. Oppenheimer director Christopher Nolan, Cillian Murphy, Emily Blunt and Matt Damon on the stakes of making a three-hour, CGI-free summer film. Christopher Nolan, the director behind such films as \"Dunkirk,\" \"Inception,\" \"Interstellar,\" and the \"Dark Knight\" trilogy, has spent the last three years living in Oppenheimer's world, writing ...\nThought:Christopher Nolan was born on July 30, 1970, which makes him 52 years old in 2023. Now I need to calculate his age in days.\nAction: Calculator\nAction Input: 52*365\nObservation: Answer: 18980\nThought:"
      ]
    }
    [llm/end] [1:RunTypeEnum.chain:AgentExecutor > 14:RunTypeEnum.chain:LLMChain > 15:RunTypeEnum.llm:ChatOpenAI] [3.52s] Exiting LLM run with output:
    {
      "generations": [
        [
          {
            "text": "I now know the final answer\nFinal Answer: The director of the 2023 film Oppenheimer is Christopher Nolan and he is 52 years old. His age in days is approximately 18980 days.",
            "generation_info": {
              "finish_reason": "stop"
            },
            "message": {
              "lc": 1,
              "type": "constructor",
              "id": [
                "langchain",
                "schema",
                "messages",
                "AIMessage"
              ],
              "kwargs": {
                "content": "I now know the final answer\nFinal Answer: The director of the 2023 film Oppenheimer is Christopher Nolan and he is 52 years old. His age in days is approximately 18980 days.",
                "additional_kwargs": {}
              }
            }
          }
        ]
      ],
      "llm_output": {
        "token_usage": {
          "prompt_tokens": 926,
          "completion_tokens": 43,
          "total_tokens": 969
        },
        "model_name": "gpt-4"
      },
      "run": null
    }
    [chain/end] [1:RunTypeEnum.chain:AgentExecutor > 14:RunTypeEnum.chain:LLMChain] [3.52s] Exiting Chain run with output:
    {
      "text": "I now know the final answer\nFinal Answer: The director of the 2023 film Oppenheimer is Christopher Nolan and he is 52 years old. His age in days is approximately 18980 days."
    }
    [chain/end] [1:RunTypeEnum.chain:AgentExecutor] [21.96s] Exiting Chain run with output:
    {
      "output": "The director of the 2023 film Oppenheimer is Christopher Nolan and he is 52 years old. His age in days is approximately 18980 days."
    }





    'The director of the 2023 film Oppenheimer is Christopher Nolan and he is 52 years old. His age in days is approximately 18980 days.'
```

</CodeOutputBlock>

</details>

### `set_vebose(True)`

Setting the `verbose` flag will print out inputs and outputs in a slightly more readable format and will skip logging certain raw outputs (like the token usage stats for an LLM call) so that you can focus on application logic.


```python
from langchain.globals import set_verbose

set_verbose(True)

agent.run("Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?")
```

<details> <summary>Console output</summary>

<CodeOutputBlock lang="python">

```
    
    
    > Entering new AgentExecutor chain...
    
    
    > Entering new LLMChain chain...
    Prompt after formatting:
    Answer the following questions as best you can. You have access to the following tools:
    
    duckduckgo_search: A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.
    Calculator: Useful for when you need to answer questions about math.
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [duckduckgo_search, Calculator]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?
    Thought:
    
    > Finished chain.
    First, I need to find out who directed the film Oppenheimer in 2023 and their birth date to calculate their age.
    Action: duckduckgo_search
    Action Input: "Director of the 2023 film Oppenheimer"
    Observation: Oppenheimer: Directed by Christopher Nolan. With Cillian Murphy, Emily Blunt, Robert Downey Jr., Alden Ehrenreich. The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb. In Christopher Nolan's new film, "Oppenheimer," Cillian Murphy stars as J. Robert ... 2023, 12:16 p.m. ET. ... including his role as the director of the Manhattan Engineer District, better ... J Robert Oppenheimer was the director of the secret Los Alamos Laboratory. It was established under US president Franklin D Roosevelt as part of the Manhattan Project to build the first atomic bomb. He oversaw the first atomic bomb detonation in the New Mexico desert in July 1945, code-named "Trinity". In this opening salvo of 2023's Oscar battle, Nolan has enjoined a star-studded cast for a retelling of the brilliant and haunted life of J. Robert Oppenheimer, the American physicist whose... Oppenheimer is a 2023 epic biographical thriller film written and directed by Christopher Nolan.It is based on the 2005 biography American Prometheus by Kai Bird and Martin J. Sherwin about J. Robert Oppenheimer, a theoretical physicist who was pivotal in developing the first nuclear weapons as part of the Manhattan Project and thereby ushering in the Atomic Age.
    Thought:
    
    > Entering new LLMChain chain...
    Prompt after formatting:
    Answer the following questions as best you can. You have access to the following tools:
    
    duckduckgo_search: A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.
    Calculator: Useful for when you need to answer questions about math.
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [duckduckgo_search, Calculator]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?
    Thought:First, I need to find out who directed the film Oppenheimer in 2023 and their birth date to calculate their age.
    Action: duckduckgo_search
    Action Input: "Director of the 2023 film Oppenheimer"
    Observation: Oppenheimer: Directed by Christopher Nolan. With Cillian Murphy, Emily Blunt, Robert Downey Jr., Alden Ehrenreich. The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb. In Christopher Nolan's new film, "Oppenheimer," Cillian Murphy stars as J. Robert ... 2023, 12:16 p.m. ET. ... including his role as the director of the Manhattan Engineer District, better ... J Robert Oppenheimer was the director of the secret Los Alamos Laboratory. It was established under US president Franklin D Roosevelt as part of the Manhattan Project to build the first atomic bomb. He oversaw the first atomic bomb detonation in the New Mexico desert in July 1945, code-named "Trinity". In this opening salvo of 2023's Oscar battle, Nolan has enjoined a star-studded cast for a retelling of the brilliant and haunted life of J. Robert Oppenheimer, the American physicist whose... Oppenheimer is a 2023 epic biographical thriller film written and directed by Christopher Nolan.It is based on the 2005 biography American Prometheus by Kai Bird and Martin J. Sherwin about J. Robert Oppenheimer, a theoretical physicist who was pivotal in developing the first nuclear weapons as part of the Manhattan Project and thereby ushering in the Atomic Age.
    Thought:
    
    > Finished chain.
    The director of the 2023 film Oppenheimer is Christopher Nolan. Now I need to find out his birth date to calculate his age.
    Action: duckduckgo_search
    Action Input: "Christopher Nolan birth date"
    Observation: July 30, 1970 (age 52) London England Notable Works: "Dunkirk" "Tenet" "The Prestige" See all related content → Recent News Jul. 13, 2023, 11:11 AM ET (AP) Cillian Murphy, playing Oppenheimer, finally gets to lead a Christopher Nolan film Christopher Edward Nolan CBE (born 30 July 1970) is a British and American filmmaker. Known for his Hollywood blockbusters with complex storytelling, Nolan is considered a leading filmmaker of the 21st century. His films have grossed $5 billion worldwide. The recipient of many accolades, he has been nominated for five Academy Awards, five BAFTA Awards and six Golden Globe Awards. Christopher Nolan is currently 52 according to his birthdate July 30, 1970 Sun Sign Leo Born Place Westminster, London, England, United Kingdom Residence Los Angeles, California, United States Nationality Education Chris attended Haileybury and Imperial Service College, in Hertford Heath, Hertfordshire. Christopher Nolan's next movie will study the man who developed the atomic bomb, J. Robert Oppenheimer. Here's the release date, plot, trailers & more. July 2023 sees the release of Christopher Nolan's new film, Oppenheimer, his first movie since 2020's Tenet and his split from Warner Bros. Billed as an epic thriller about "the man who ...
    Thought:
    
    > Entering new LLMChain chain...
    Prompt after formatting:
    Answer the following questions as best you can. You have access to the following tools:
    
    duckduckgo_search: A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.
    Calculator: Useful for when you need to answer questions about math.
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [duckduckgo_search, Calculator]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?
    Thought:First, I need to find out who directed the film Oppenheimer in 2023 and their birth date to calculate their age.
    Action: duckduckgo_search
    Action Input: "Director of the 2023 film Oppenheimer"
    Observation: Oppenheimer: Directed by Christopher Nolan. With Cillian Murphy, Emily Blunt, Robert Downey Jr., Alden Ehrenreich. The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb. In Christopher Nolan's new film, "Oppenheimer," Cillian Murphy stars as J. Robert ... 2023, 12:16 p.m. ET. ... including his role as the director of the Manhattan Engineer District, better ... J Robert Oppenheimer was the director of the secret Los Alamos Laboratory. It was established under US president Franklin D Roosevelt as part of the Manhattan Project to build the first atomic bomb. He oversaw the first atomic bomb detonation in the New Mexico desert in July 1945, code-named "Trinity". In this opening salvo of 2023's Oscar battle, Nolan has enjoined a star-studded cast for a retelling of the brilliant and haunted life of J. Robert Oppenheimer, the American physicist whose... Oppenheimer is a 2023 epic biographical thriller film written and directed by Christopher Nolan.It is based on the 2005 biography American Prometheus by Kai Bird and Martin J. Sherwin about J. Robert Oppenheimer, a theoretical physicist who was pivotal in developing the first nuclear weapons as part of the Manhattan Project and thereby ushering in the Atomic Age.
    Thought:The director of the 2023 film Oppenheimer is Christopher Nolan. Now I need to find out his birth date to calculate his age.
    Action: duckduckgo_search
    Action Input: "Christopher Nolan birth date"
    Observation: July 30, 1970 (age 52) London England Notable Works: "Dunkirk" "Tenet" "The Prestige" See all related content → Recent News Jul. 13, 2023, 11:11 AM ET (AP) Cillian Murphy, playing Oppenheimer, finally gets to lead a Christopher Nolan film Christopher Edward Nolan CBE (born 30 July 1970) is a British and American filmmaker. Known for his Hollywood blockbusters with complex storytelling, Nolan is considered a leading filmmaker of the 21st century. His films have grossed $5 billion worldwide. The recipient of many accolades, he has been nominated for five Academy Awards, five BAFTA Awards and six Golden Globe Awards. Christopher Nolan is currently 52 according to his birthdate July 30, 1970 Sun Sign Leo Born Place Westminster, London, England, United Kingdom Residence Los Angeles, California, United States Nationality Education Chris attended Haileybury and Imperial Service College, in Hertford Heath, Hertfordshire. Christopher Nolan's next movie will study the man who developed the atomic bomb, J. Robert Oppenheimer. Here's the release date, plot, trailers & more. July 2023 sees the release of Christopher Nolan's new film, Oppenheimer, his first movie since 2020's Tenet and his split from Warner Bros. Billed as an epic thriller about "the man who ...
    Thought:
    
    > Finished chain.
    Christopher Nolan was born on July 30, 1970. Now I need to calculate his age in 2023 and then convert it into days.
    Action: Calculator
    Action Input: (2023 - 1970) * 365
    
    > Entering new LLMMathChain chain...
    (2023 - 1970) * 365
    
    > Entering new LLMChain chain...
    Prompt after formatting:
    Translate a math problem into a expression that can be executed using Python's numexpr library. Use the output of running this code to answer the question.
    
    Question: ${Question with math problem.}
    ```text
    ${single line mathematical expression that solves the problem}
    ```
    ...numexpr.evaluate(text)...
    ```output
    ${Output of running the code}
    ```
    Answer: ${Answer}
    
    Begin.
    
    Question: What is 37593 * 67?
    ```text
    37593 * 67
    ```
    ...numexpr.evaluate("37593 * 67")...
    ```output
    2518731
    ```
    Answer: 2518731
    
    Question: 37593^(1/5)
    ```text
    37593**(1/5)
    ```
    ...numexpr.evaluate("37593**(1/5)")...
    ```output
    8.222831614237718
    ```
    Answer: 8.222831614237718
    
    Question: (2023 - 1970) * 365
    
    
    > Finished chain.
    ```text
    (2023 - 1970) * 365
    ```
    ...numexpr.evaluate("(2023 - 1970) * 365")...
    
    Answer: 19345
    > Finished chain.
    
    Observation: Answer: 19345
    Thought:
    
    > Entering new LLMChain chain...
    Prompt after formatting:
    Answer the following questions as best you can. You have access to the following tools:
    
    duckduckgo_search: A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.
    Calculator: Useful for when you need to answer questions about math.
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [duckduckgo_search, Calculator]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?
    Thought:First, I need to find out who directed the film Oppenheimer in 2023 and their birth date to calculate their age.
    Action: duckduckgo_search
    Action Input: "Director of the 2023 film Oppenheimer"
    Observation: Oppenheimer: Directed by Christopher Nolan. With Cillian Murphy, Emily Blunt, Robert Downey Jr., Alden Ehrenreich. The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb. In Christopher Nolan's new film, "Oppenheimer," Cillian Murphy stars as J. Robert ... 2023, 12:16 p.m. ET. ... including his role as the director of the Manhattan Engineer District, better ... J Robert Oppenheimer was the director of the secret Los Alamos Laboratory. It was established under US president Franklin D Roosevelt as part of the Manhattan Project to build the first atomic bomb. He oversaw the first atomic bomb detonation in the New Mexico desert in July 1945, code-named "Trinity". In this opening salvo of 2023's Oscar battle, Nolan has enjoined a star-studded cast for a retelling of the brilliant and haunted life of J. Robert Oppenheimer, the American physicist whose... Oppenheimer is a 2023 epic biographical thriller film written and directed by Christopher Nolan.It is based on the 2005 biography American Prometheus by Kai Bird and Martin J. Sherwin about J. Robert Oppenheimer, a theoretical physicist who was pivotal in developing the first nuclear weapons as part of the Manhattan Project and thereby ushering in the Atomic Age.
    Thought:The director of the 2023 film Oppenheimer is Christopher Nolan. Now I need to find out his birth date to calculate his age.
    Action: duckduckgo_search
    Action Input: "Christopher Nolan birth date"
    Observation: July 30, 1970 (age 52) London England Notable Works: "Dunkirk" "Tenet" "The Prestige" See all related content → Recent News Jul. 13, 2023, 11:11 AM ET (AP) Cillian Murphy, playing Oppenheimer, finally gets to lead a Christopher Nolan film Christopher Edward Nolan CBE (born 30 July 1970) is a British and American filmmaker. Known for his Hollywood blockbusters with complex storytelling, Nolan is considered a leading filmmaker of the 21st century. His films have grossed $5 billion worldwide. The recipient of many accolades, he has been nominated for five Academy Awards, five BAFTA Awards and six Golden Globe Awards. Christopher Nolan is currently 52 according to his birthdate July 30, 1970 Sun Sign Leo Born Place Westminster, London, England, United Kingdom Residence Los Angeles, California, United States Nationality Education Chris attended Haileybury and Imperial Service College, in Hertford Heath, Hertfordshire. Christopher Nolan's next movie will study the man who developed the atomic bomb, J. Robert Oppenheimer. Here's the release date, plot, trailers & more. July 2023 sees the release of Christopher Nolan's new film, Oppenheimer, his first movie since 2020's Tenet and his split from Warner Bros. Billed as an epic thriller about "the man who ...
    Thought:Christopher Nolan was born on July 30, 1970. Now I need to calculate his age in 2023 and then convert it into days.
    Action: Calculator
    Action Input: (2023 - 1970) * 365
    Observation: Answer: 19345
    Thought:
    
    > Finished chain.
    I now know the final answer
    Final Answer: The director of the 2023 film Oppenheimer is Christopher Nolan and he is 53 years old in 2023. His age in days is 19345 days.
    
    > Finished chain.


    'The director of the 2023 film Oppenheimer is Christopher Nolan and he is 53 years old in 2023. His age in days is 19345 days.'
```

</CodeOutputBlock>

</details>

### `Chain(..., verbose=True)`

You can also scope verbosity down to a single object, in which case only the inputs and outputs to that object are printed (along with any additional callbacks calls made specifically by that object).


```python
# Passing verbose=True to initialize_agent will pass that along to the AgentExecutor (which is a Chain).
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent.run("Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?")
```

<details> <summary>Console output</summary>

<CodeOutputBlock lang="python">

```
    > Entering new AgentExecutor chain...
    First, I need to find out who directed the film Oppenheimer in 2023 and their birth date. Then, I can calculate their age in years and days.
    Action: duckduckgo_search
    Action Input: "Director of 2023 film Oppenheimer"
    Observation: Oppenheimer: Directed by Christopher Nolan. With Cillian Murphy, Emily Blunt, Robert Downey Jr., Alden Ehrenreich. The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb. In Christopher Nolan's new film, "Oppenheimer," Cillian Murphy stars as J. Robert Oppenheimer, the American physicist who oversaw the Manhattan Project in Los Alamos, N.M. Universal Pictures... J Robert Oppenheimer was the director of the secret Los Alamos Laboratory. It was established under US president Franklin D Roosevelt as part of the Manhattan Project to build the first atomic bomb. He oversaw the first atomic bomb detonation in the New Mexico desert in July 1945, code-named "Trinity". A Review of Christopher Nolan's new film 'Oppenheimer' , the story of the man who fathered the Atomic Bomb. Cillian Murphy leads an all star cast ... Release Date: July 21, 2023. Director ... For his new film, "Oppenheimer," starring Cillian Murphy and Emily Blunt, director Christopher Nolan set out to build an entire 1940s western town.
    Thought:The director of the 2023 film Oppenheimer is Christopher Nolan. Now I need to find out his birth date to calculate his age.
    Action: duckduckgo_search
    Action Input: "Christopher Nolan birth date"
    Observation: July 30, 1970 (age 52) London England Notable Works: "Dunkirk" "Tenet" "The Prestige" See all related content → Recent News Jul. 13, 2023, 11:11 AM ET (AP) Cillian Murphy, playing Oppenheimer, finally gets to lead a Christopher Nolan film Christopher Edward Nolan CBE (born 30 July 1970) is a British and American filmmaker. Known for his Hollywood blockbusters with complex storytelling, Nolan is considered a leading filmmaker of the 21st century. His films have grossed $5 billion worldwide. The recipient of many accolades, he has been nominated for five Academy Awards, five BAFTA Awards and six Golden Globe Awards. Christopher Nolan is currently 52 according to his birthdate July 30, 1970 Sun Sign Leo Born Place Westminster, London, England, United Kingdom Residence Los Angeles, California, United States Nationality Education Chris attended Haileybury and Imperial Service College, in Hertford Heath, Hertfordshire. Christopher Nolan's next movie will study the man who developed the atomic bomb, J. Robert Oppenheimer. Here's the release date, plot, trailers & more. Date of Birth: 30 July 1970 . ... Christopher Nolan is a British-American film director, producer, and screenwriter. His films have grossed more than US$5 billion worldwide, and have garnered 11 Academy Awards from 36 nominations. ...
    Thought:Christopher Nolan was born on July 30, 1970. Now I can calculate his age in years and then in days.
    Action: Calculator
    Action Input: {"operation": "subtract", "operands": [2023, 1970]}
    Observation: Answer: 53
    Thought:Christopher Nolan is 53 years old in 2023. Now I need to calculate his age in days.
    Action: Calculator
    Action Input: {"operation": "multiply", "operands": [53, 365]}
    Observation: Answer: 19345
    Thought:I now know the final answer
    Final Answer: The director of the 2023 film Oppenheimer is Christopher Nolan. He is 53 years old in 2023, which is approximately 19345 days.
    
    > Finished chain.


    'The director of the 2023 film Oppenheimer is Christopher Nolan. He is 53 years old in 2023, which is approximately 19345 days.'
```

</CodeOutputBlock>

</details>

## Other callbacks

`Callbacks` are what we use to execute any functionality within a component outside the primary component logic. All of the above solutions use `Callbacks` under the hood to log intermediate steps of components. There's a number of `Callbacks` relevant for debugging that come with LangChain out of the box, like the [FileCallbackHandler](/docs/modules/callbacks/how_to/filecallbackhandler). You can also implement your own callbacks to execute custom functionality.

See here for more info on [Callbacks](/docs/modules/callbacks/), how to use them, and customize them.

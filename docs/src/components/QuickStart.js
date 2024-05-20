import CodeBlock from "@theme/CodeBlock";
import React from "react";
import TabItem from "@theme/TabItem";
import Tabs from "@theme/Tabs";
import {
  CodeTabs,
  PythonBlock,
  ShellBlock,
  TypeScriptBlock,
} from "./InstructionsWithCode";

export const TypeScriptSDKTracingCode = () =>
  `import { OpenAI } from "openai";
import { traceable } from "langsmith/traceable";
import { wrapOpenAI } from "langsmith/wrappers";\n
// Auto-trace LLM calls in-context
const client = wrapOpenAI(new OpenAI());
// Auto-trace this function
const pipeline = traceable(async (user_input) => {
    const result = await client.chat.completions.create({
        messages: [{ role: "user", content: user_input }],
        model: "gpt-3.5-turbo",
    });
    return result.choices[0].message.content;
});

await pipeline("Hello, world!")
// Out: Hello there! How can I assist you today?`;

export function TypeScriptSDKTracingCodeBlock() {
  return (
    <CodeBlock language="typescript">{TypeScriptSDKTracingCode()}</CodeBlock>
  );
}

export function PythonAPITracingCodeBlock() {
  return (
    <CodeBlock language="python">
      {`import openai
import requests
from datetime import datetime
from uuid import uuid4

def post_run(run_id, name, run_type, inputs, parent_id=None):
    """Function to post a new run to the API."""
    data = {
        "id": run_id.hex,
        "name": name,
        "run_type": run_type,
        "inputs": inputs,
        "start_time": datetime.utcnow().isoformat(),
    }
    if parent_id:
        data["parent_run_id"] = parent_id.hex
    requests.post(
        "https://api.smith.langchain.com/runs",
        json=data,
        headers=headers
    )

def patch_run(run_id, outputs):
    """Function to patch a run with outputs."""
    requests.patch(
        f"https://api.smith.langchain.com/runs/{run_id}",
        json={
            "outputs": outputs,
            "end_time": datetime.utcnow().isoformat(),
        },
        headers=headers,
    )

# Send your API Key in the request headers
headers = {"x-api-key": "<YOUR API KEY>"}

# This can be a user input to your app
question = "Can you summarize this morning's meetings?"

# This can be retrieved in a retrieval step
context = "During this morning's meeting, we solved all world conflict."
messages = [
    {"role": "system", "content": "You are a helpful assistant. Please respond to the user's request only based on the given context."},
    {"role": "user", "content": f"Question: {question}\\nContext: {context}"}
]

# Create parent run
parent_run_id = uuid4()
post_run(parent_run_id, "Chat Pipeline", "chain", {"question": question})

# Create child run
child_run_id = uuid4()
post_run(child_run_id, "OpenAI Call", "llm", {"messages": messages}, parent_run_id)

# Generate a completion
client = openai.Client()
chat_completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)

# End runs
patch_run(child_run_id, chat_completion.dict())
patch_run(parent_run_id, {"answer": chat_completion.choices[0].message.content})`}
    </CodeBlock>
  );
}

export const PythonSDKTracingCode = () =>
  `import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable\n
# Auto-trace LLM calls in-context
client = wrap_openai(openai.Client())\n
@traceable # Auto-trace this function
def pipeline(user_input: str):
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="gpt-3.5-turbo"
    )
    return result.choices[0].message.content\n
pipeline("Hello, world!")
# Out:  Hello there! How can I assist you today?`;

export function PythonSDKTracingCodeBlock() {
  return <CodeBlock language="python">{PythonSDKTracingCode()}</CodeBlock>;
}

export function LangChainInstallationCodeTabs() {
  return (
    <CodeTabs
      groupId="client-language"
      tabs={[
        {
          value: "python",
          label: "pip",
          language: "bash",
          content: `pip install langchain_openai langchain_core`,
        },
        {
          value: "typescript",
          label: "yarn",
          language: "bash",
          content: `yarn add @langchain/openai @langchain/core`,
        },
        {
          value: "npm",
          label: "npm",
          language: "bash",
          content: `npm install @langchain/openai @langchain/core`,
        },
        {
          value: "pnpm",
          label: "pnpm",
          language: "bash",
          content: `pnpm add @langchain/openai @langchain/core`,
        },
      ]}
    />
  );
}

export function ConfigureSDKEnvironmentCodeTabs({}) {
  return (
    <CodeTabs
      tabs={[
        ShellBlock(`export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>

# The below examples use the OpenAI API, though it's not necessary in general
export OPENAI_API_KEY=<your-openai-api-key>`),
      ]}
      groupId="client-language"
    />
  );
}

export function ConfigureEnvironmentCodeTabs({}) {
  return (
    <CodeTabs
      tabs={[
        ShellBlock(`export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>

# The below examples use the OpenAI API, though it's not necessary in general
export OPENAI_API_KEY=<your-openai-api-key>`),
      ]}
      groupId="client-language"
    />
  );
}

export function LangChainQuickStartCodeTabs({}) {
  const simpleTSBlock = `import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant. Please respond to the user's request only based on the given context."],
  ["user", "Question: {question}\\nContext: {context}"],
]);
const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo" });
const outputParser = new StringOutputParser();

const chain = prompt.pipe(model).pipe(outputParser);

const question = "Can you summarize this morning's meetings?"
const context = "During this morning's meeting, we solved all world conflict."
await chain.invoke({ question: question, context: context });`;

  const alternativeTSBlock = `import { Client } from "langsmith";
import { LangChainTracer } from "langchain/callbacks";

const client = new Client({
  apiUrl: "https://api.smith.langchain.com",
  apiKey: "YOUR_API_KEY"
});

const tracer = new LangChainTracer({
  projectName: "YOUR_PROJECT_NAME",
  client
});

const model = new ChatOpenAI({
  openAIApiKey: "YOUR_OPENAI_API_KEY"
});

await model.invoke("Hello, world!", { callbacks: [tracer] })`;

  return (
    <Tabs groupId="client-language" className="code-tabs">
      <TabItem key="python" value="python" label="Python">
        <CodeBlock className="python" language="python">
          {`from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's request only based on the given context."),
    ("user", "Question: {question}\\nContext: {context}")
])
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

question = "Can you summarize this morning's meetings?"
context = "During this morning's meeting, we solved all world conflict."
chain.invoke({"question": question, "context": context})`}
        </CodeBlock>
      </TabItem>
      <TabItem key="typescript" value="typescript" label="TypeScript">
        <CodeBlock className="typescript" language="typescript">
          {simpleTSBlock}
        </CodeBlock>
      </TabItem>
    </Tabs>
  );
}

const TraceableQuickStart = PythonBlock(`from typing import Any, Iterable\n
import openai
from langsmith import traceable
from langsmith.wrappers import wrap_openai\n
# Optional: wrap the openai client to add tracing directly
client = wrap_openai(openai.Client())\n\n
@traceable(run_type="tool")
def my_tool() -> str:
    return "In the meeting, we solved all world conflict."\n\n
@traceable
def my_chat_bot(prompt: str) -> Iterable[str]:
    tool_response = my_tool()
    messages = [
        {
            "role": "system",
            "content": f"You are an AI Assistant.\\n\\nTool response: {tool_response}",
        },
        {"role": "user", "content": prompt},
    ]
    chunks = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, stream=True
    )
    for chunk in chunks:
        yield chunk.choices[0].delta.content\n\n
for tok in my_chat_bot("Summarize this morning's meetings."):
    print(tok, end="")
# See an example run at: https://smith.langchain.com/public/3e853ad8-77ce-404d-ad4c-05726851ad0f/r`);

export function TraceableQuickStartCodeBlock({}) {
  return (
    <CodeBlock
      className={TraceableQuickStart.value}
      language={TraceableQuickStart.language ?? TraceableQuickStart.value}
    >
      {TraceableQuickStart.content}
    </CodeBlock>
  );
}

export function TraceableThreadingCodeBlock({}) {
  return (
    <CodeBlock
      className={TraceableQuickStart.value}
      language={TraceableQuickStart.language ?? TraceableQuickStart.value}
    >
      {`import asyncio
import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List\n
import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable, RunTree\n\n
# Optional: wrap the openai client to add tracing directly
client = wrap_openai(openai.Client())\n
def call_llm(prompt: str, temperature: float = 0.0, **kwargs: Any):
    """Call a completion model."""
    \n\n
@traceable(run_type="chain")
def llm_chain(user_input: str, **kwargs: Any) -> str:
    """Select the text from the openai call."""
    return client.completions.create(
        model="gpt-3.5-turbo-instruct", prompt=user_input, temperature=1.0, **kwargs
    ).choices[0].text\n\n
@traceable(run_type="llm")
def my_chat_model(messages: List[Dict], temperature: float = 0.0, **kwargs: Any):
    """Call a chat model."""
    return client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=temperature, **kwargs
    )\n\n
@traceable(run_type="chain")
def llm_chat_chain(user_input: str, **kwargs: Any) -> str:
    """Prepare prompt & select first choice response."""
    messages = [
        {
            "role": "system",
            "content": "You are an AI Assistant. The time is "
            + str(datetime.datetime.now()),
        },
        {"role": "user", "content": user_input},
    ]
    return my_chat_model(messages=messages, **kwargs).choices[0].message.content\n\n
@traceable(run_type="chain")
# highlight-next-line
async def nested_chain(text: str, run_tree: RunTree, **kwargs: Any) -> str:
    """Example with nesting and thread pools."""
    futures = []
    with ThreadPoolExecutor() as thread_pool:
        for i in range(2):
            futures.append(
                thread_pool.submit(
                    llm_chain,
                    f"Completion gather {i}: {text}",
                    # highlight-next-line
                    langsmith_extra={"run_tree": run_tree},
                    **kwargs,
                )
            )
        for i in range(2):
            futures.append(
                thread_pool.submit(
                    llm_chat_chain,
                    f"Chat gather {i}: {text}",
                    # highlight-next-line
                    langsmith_extra={"run_tree": run_tree},
                    **kwargs,
                )
            )
    return "\\n".join([future.result() for future in futures])\n\n
asyncio.run(nested_chain("Summarize meeting"))`}
    </CodeBlock>
  );
}

export function RunTreeQuickStartCodeTabs({}) {
  return (
    <CodeTabs
      tabs={[
        TraceableQuickStart,
        {
          value: "python-run-tree",
          label: "Python (Run Tree)",
          language: "python",
          content: `from langsmith.run_trees import RunTree\n
parent_run = RunTree(
    name="My Chat Bot",
    run_type="chain",
    inputs={"text": "Summarize this morning's meetings."},
    serialized={}
)\n
child_llm_run = parent_run.create_child(
    name="My Proprietary LLM",
    run_type="llm",
    inputs={
        "prompts": [
            "You are an AI Assistant. Summarize this morning's meetings."
        ]
    },
)\n
child_llm_run.end(outputs={"generations": ["Summary of the meeting..."]})
parent_run.end(outputs={"output": ["The meeting notes are as follows:..."]})\n
res = parent_run.post(exclude_child_runs=False)
res.result()`,
        },
        TypeScriptBlock(`import { RunTree, RunTreeConfig } from "langsmith";\n
const parentRunConfig: RunTreeConfig = {
    name: "My Chat Bot",
    run_type: "chain",
    inputs: {
        text: "Summarize this morning's meetings.",
    },
    serialized: {}
};\n
const parentRun = new RunTree(parentRunConfig);\n
const childLlmRun = await parentRun.createChild({
    name: "My Proprietary LLM",
    run_type: "llm",
    inputs: {
        prompts: [
        "You are an AI Assistant. Summarize this morning's meetings.",
        ],
    },
});\n
await childLlmRun.end({
outputs: {
    generations: [
    "Summary of the meeting...",
    ],
},
});\n
await parentRun.end({
outputs: {
    output: ["The meeting notes are as follows:..."],
},
});\n
// False means post all nested runs as a batch
// (don't exclude child runs)
await parentRun.postRun(false);

  `),
      ]}
      groupId="client-language"
    />
  );
}

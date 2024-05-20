import CodeBlock from "@theme/CodeBlock";
import React from "react";
import TabItem from "@theme/TabItem";
import Tabs from "@theme/Tabs";

export function AccessRunIdBlock({}) {
  const callbackPythonBlock = `from langchain import chat_models, prompts, callbacks
chain = (
    prompts.ChatPromptTemplate.from_template("Say hi to {name}")
    | chat_models.ChatAnthropic()
)
with callbacks.collect_runs() as cb:
  result = chain.invoke({"name": "Clara"})
  run_id = cb.traced_runs[0].id
print(run_id)
`;

  const alternativePythonBlock = `from langchain.chat_models import ChatAnthropic
from langchain.chains import LLMChain\n
chain = LLMChain.from_string(ChatAnthropic(), "Say hi to {name}")
response = chain("Clara", include_run_info=True)
run_id = response["__run"].run_id
print(run_id)`;

  const chatModelPythonBlock = `from langchain.chat_models import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
 
 chat_model = ChatAnthropic()
 
 prompt = ChatPromptTemplate.from_messages(
     [
         ("system", "You are a cat"),
         ("human", "Hi"),
     ]
 )
 res = chat_model.generate(messages=[prompt.format_messages()])
 res.run[0].run_id`;

  const llmModelPythonBlock = `from langchain.llms import OpenAI

openai = OpenAI()
res = openai.generate(["You are a good bot"])
print(res.run[0].run_id)`;

  const callbackTypeScriptBlock = `import { ChatAnthropic } from "langchain/chat_models/anthropic";
import { PromptTemplate } from "langchain/prompts";
import { RunCollectorCallbackHandler } from "langchain/callbacks";\n
const runCollector = new RunCollectorCallbackHandler();
const prompt = PromptTemplate.fromTemplate("Say hi to {name}");
const chain = prompt.pipe(new ChatAnthropic());
const pred = await chain.invoke(
  { name: "Clara" },
  {
    callbacks: [runCollector],
  }
);
const runId = runCollector.tracedRuns[0].id;
console.log(runId);`;
  const oldTypeScriptBlock = `import { ChatAnthropic } from "langchain/chat_models/anthropic";
import { LLMChain } from "langchain/chains";
import { PromptTemplate } from "langchain/prompts";\n
const prompt = PromptTemplate.fromTemplate("Say hi to {name}");
const chain = new LLMChain({
  llm: new ChatAnthropic(),
  prompt: prompt,
});\n
const response = await chain.invoke({ name: "Clara" });
console.log(response.__run);`;
  return (
    <Tabs groupId="client-language">
      <TabItem key="python" value="python" label="LangChain (Python)">
        <CodeBlock className="python" language="python">
          {callbackPythonBlock}
        </CodeBlock>
        <p>
          For older versions of LangChain ({`<`}0.0.276), you can instruct the
          chain to return the run ID by specifying the `include_run_info=True`
          parameter to the call function:
        </p>
        <CodeBlock className="python" language="python">
          {alternativePythonBlock}
        </CodeBlock>
        <p>
          For python LLMs/chat models, the run information is returned
          automatically when calling the `generate()` method. Example:
        </p>
        <CodeBlock className="python" language="python">
          {chatModelPythonBlock}
        </CodeBlock>
        <p>or for LLMs</p>
        <CodeBlock className="python" language="python">
          {llmModelPythonBlock}
        </CodeBlock>
      </TabItem>
      <TabItem key="typescript" value="typescript" label="LangChain (JS)">
        <p>
          For newer versions of Langchain ({`>=`}0.0.139), you can use the
          `RunCollectorCallbackHandler` for any chain or runnable.
        </p>
        <CodeBlock className="typescript" language="typescript">
          {callbackTypeScriptBlock}
        </CodeBlock>
        <p>
          If youre on an older version of LangChain, you can still retrieve the
          run ID directly from chain calls.
        </p>
        <CodeBlock className="typescript" language="typescript">
          {oldTypeScriptBlock}
        </CodeBlock>
      </TabItem>
    </Tabs>
  );
}

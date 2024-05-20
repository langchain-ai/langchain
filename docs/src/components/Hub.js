import React from "react";
import Tabs from "@theme/Tabs";
import TabItem from "@theme/TabItem";
import CodeBlock from "@theme/CodeBlock";

import { CodeTabs } from "./InstructionsWithCode";

export function HubInstallationCodeTabs() {
  return (
    <CodeTabs
      groupId="client-language"
      tabs={[
        {
          value: "python",
          label: "pip",
          language: "bash",
          content: `pip install -U langchain langchainhub langchain-openai`,
        },
        {
          value: "typescript",
          label: "yarn",
          language: "bash",
          content: `yarn add langchain`,
        },
        {
          value: "npm",
          label: "npm",
          language: "bash",
          content: `npm install -S langchain`,
        },
      ]}
    />
  );
}

export function HubPullCodeTabs() {
  const pyBlock = `from langchain import hub

# pull a chat prompt
prompt = hub.pull("efriis/my-first-prompt")

# create a model to use it with
from langchain_openai import ChatOpenAI
model = ChatOpenAI()

# use it in a runnable
runnable = prompt | model
response = runnable.invoke({
	"profession": "biologist",
	"question": "What is special about parrots?",
})

print(response)
`;

  const jsBlock = `// import
import * as hub from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";

// pull a chat prompt
const prompt = await hub.pull<ChatPromptTemplate>("efriis/my-first-prompt");


// create a model to use it with
const model = new ChatOpenAI();

// use it in a runnable
const runnable = prompt.pipe(model);
const result = await runnable.invoke({
  "profession": "biologist",
  "question": "What is special about parrots?",
});

console.log(result);`;

  return (
    <Tabs groupId="client-language">
      <TabItem key="python" value="python" label="Python">
        <CodeBlock className="python" language="python">
          {pyBlock}
        </CodeBlock>
      </TabItem>
      <TabItem key="typescript" value="typescript" label="TypeScript">
        <CodeBlock className="typescript" language="typescript">
          {jsBlock}
        </CodeBlock>
      </TabItem>
    </Tabs>
  );
}

export function HubPushCodeTabs() {
  const pyBlock = `from langchain import hub
from langchain.prompts.chat import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

hub.push("<handle>/topic-joke-generator", prompt, new_repo_is_public=False)`;

  const jsBlock = `import * as hub from "langchain/hub";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
} from '@langchain/core/prompts';

const message = HumanMessagePromptTemplate.fromTemplate(
  'tell me a joke about {topic}'
);
const prompt = ChatPromptTemplate.fromMessages([message]);

await hub.push("<handle>/my-first-prompt", prompt, { newRepoIsPublic: false });`;

  return (
    <Tabs groupId="client-language">
      <TabItem key="python" value="python" label="Python">
        <CodeBlock className="python" language="python">
          {pyBlock}
        </CodeBlock>
      </TabItem>
      <TabItem key="typescript" value="typescript" label="TypeScript">
        <CodeBlock className="typescript" language="typescript">
          {jsBlock}
        </CodeBlock>
      </TabItem>
    </Tabs>
  );
}

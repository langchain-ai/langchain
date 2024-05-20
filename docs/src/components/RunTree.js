import React from "react";
import { CodeTabs, PythonBlock, TypeScriptBlock } from "./InstructionsWithCode";

export function RunTreeExampleCodeTabs() {
  return (
    <CodeTabs
      tabs={[
        PythonBlock(`from langsmith.run_trees import RunTree\n
parent_run = RunTree(
    name="My Chat Bot",
    run_type="chain",
    inputs={"text": "Summarize this morning's meetings."},
    serialized={},  # Serialized representation of this chain
    # project_name= "Defaults to the LANGCHAIN_PROJECT env var"
    # api_url= "Defaults to the LANGCHAIN_ENDPOINT env var"
    # api_key= "Defaults to the LANGCHAIN_API_KEY env var"
)
# .. My Chat Bot calls an LLM
child_llm_run = parent_run.create_child(
    name="My Proprietary LLM",
    run_type="llm",
    inputs={
        "prompts": [
            "You are an AI Assistant. The time is XYZ."
            " Summarize this morning's meetings."
        ]
    },
)
child_llm_run.end(
    outputs={
        "generations": [
            "I should use the transcript_loader tool"
            " to fetch meeting_transcripts from XYZ"
        ]
    }
)
# ..  My Chat Bot takes the LLM output and calls
# a tool / function for fetching transcripts ..
child_tool_run = parent_run.create_child(
    name="transcript_loader",
    run_type="tool",
    inputs={"date": "XYZ", "content_type": "meeting_transcripts"},
)
# The tool returns meeting notes to the chat bot
child_tool_run.end(outputs={"meetings": ["Meeting1 notes.."]})\n
child_chain_run = parent_run.create_child(
    name="Unreliable Component",
    run_type="tool",
    inputs={"input": "Summarize these notes..."},
)\n
try:
    # .... the component does work
    raise ValueError("Something went wrong")
except Exception as e:
    child_chain_run.end(error=f"I errored again {e}")
    pass
# .. The chat agent recovers\n
parent_run.end(outputs={"output": ["The meeting notes are as follows:..."]})\n
# This posts all nested runs as a batch
res = parent_run.post(exclude_child_runs=False)
res.result()
`),
        TypeScriptBlock(`import { RunTree, RunTreeConfig } from "langsmith";\n
const parentRunConfig: RunTreeConfig = {
  name: "My Chat Bot",
  run_type: "chain",
  inputs: {
    text: "Summarize this morning's meetings.",
  },
  serialized: {}, // Serialized representation of this chain
  // session_name: "Defaults to the LANGCHAIN_PROJECT env var"
  // apiUrl: "Defaults to the LANGCHAIN_ENDPOINT env var"
  // apiKey: "Defaults to the LANGCHAIN_API_KEY env var"
};\n
const parentRun = new RunTree(parentRunConfig);\n
const childLlmRun = await parentRun.createChild({
  name: "My Proprietary LLM",
  run_type: "llm",
  inputs: {
    prompts: [
      "You are an AI Assistant. The time is XYZ." +
        " Summarize this morning's meetings.",
    ],
  },
});\n
await childLlmRun.end({
  outputs: {
    generations: [
      "I should use the transcript_loader tool" +
        " to fetch meeting_transcripts from XYZ",
    ],
  },
});\n
const childToolRun = await parentRun.createChild({
  name: "transcript_loader",
  run_type: "tool",
  inputs: {
    date: "XYZ",
    content_type: "meeting_transcripts",
  },
});\n
await childToolRun.end({
  outputs: {
    meetings: ["Meeting1 notes.."],
  },
});\n
const childChainRun = await parentRun.createChild({
  name: "Unreliable Component",
  run_type: "tool",
  inputs: {
    input: "Summarize these notes...",
  },
});\n
try {
  // .... the component does work
  throw new Error("Something went wrong");
} catch (e) {
  await childChainRun.end({
    error: \`I errored again $\{e.message}\`,
  });
}\n
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

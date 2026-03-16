from pathlib import Path

from langchain_core.callbacks import CallbackManager
from langchain_core.runnables import RunnableLambda

from agent_evidence.integrations.langchain import EvidenceCallbackHandler
from agent_evidence.recorder import EvidenceRecorder
from agent_evidence.storage.local import LocalEvidenceStore


output_dir = Path("./evidence_bundle")
store = LocalEvidenceStore(output_dir / "events.jsonl")
callback = EvidenceCallbackHandler(recorder=EvidenceRecorder(store))

manager = CallbackManager([callback])
chain = RunnableLambda(lambda text: text.upper()).with_config(
    {"callbacks": manager.handlers, "run_name": "runtime-evidence-demo"}
)

result = chain.invoke("hello")
print(result)
print(f"Evidence events written to {store.path}")

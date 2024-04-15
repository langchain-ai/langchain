import uuid
from types import SimpleNamespace
from unittest import mock

from langchain_core.outputs import LLMResult

from langchain_community.callbacks.tracers import comet


def test_comet_tracer__trace_chain_with_single_span__happyflow() -> None:
    # Setup mocks
    chain_module_mock = mock.Mock()
    chain_instance_mock = mock.Mock()
    chain_module_mock.Chain.return_value = chain_instance_mock

    span_module_mock = mock.Mock()
    span_instance_mock = mock.MagicMock()
    span_instance_mock.__api__start__ = mock.Mock()
    span_instance_mock.__api__end__ = mock.Mock()

    span_module_mock.Span.return_value = span_instance_mock

    experiment_info_module_mock = mock.Mock()
    experiment_info_module_mock.get.return_value = "the-experiment-info"

    chain_api_module_mock = mock.Mock()

    comet_ml_api_mock = SimpleNamespace(
        chain=chain_module_mock,
        span=span_module_mock,
        experiment_info=experiment_info_module_mock,
        chain_api=chain_api_module_mock,
        flush="not-used-in-this-test",
    )

    # Create tracer
    with mock.patch.object(
        comet, "import_comet_llm_api", return_value=comet_ml_api_mock
    ):
        tracer = comet.CometTracer()

    run_id_1 = uuid.UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a")
    run_id_2 = uuid.UUID("4f31216e-7c26-4027-a5fd-0bbf9ace17dc")

    # Parent run
    tracer.on_chain_start(
        {"name": "chain-input"},
        {"input": "chain-input-prompt"},
        parent_run_id=None,
        run_id=run_id_1,
    )

    # Check that chain was created
    chain_module_mock.Chain.assert_called_once_with(
        inputs={"input": "chain-input-prompt"},
        metadata=None,
        experiment_info="the-experiment-info",
    )

    # Child run
    tracer.on_llm_start(
        {"name": "span-input"},
        ["span-input-prompt"],
        parent_run_id=run_id_1,
        run_id=run_id_2,
    )

    # Check that Span was created and attached to chain
    span_module_mock.Span.assert_called_once_with(
        inputs={"prompts": ["span-input-prompt"]},
        category=mock.ANY,
        metadata=mock.ANY,
        name=mock.ANY,
    )
    span_instance_mock.__api__start__(chain_instance_mock)

    # Child run end
    tracer.on_llm_end(
        LLMResult(generations=[], llm_output={"span-output-key": "span-output-value"}),
        run_id=run_id_2,
    )
    # Check that Span outputs are set and span is ended
    span_instance_mock.set_outputs.assert_called_once()
    actual_span_outputs = span_instance_mock.set_outputs.call_args[1]["outputs"]
    assert {
        "llm_output": {"span-output-key": "span-output-value"},
        "generations": [],
    }.items() <= actual_span_outputs.items()
    span_instance_mock.__api__end__()

    # Parent run end
    tracer.on_chain_end({"chain-output-key": "chain-output-value"}, run_id=run_id_1)

    # Check that chain outputs are set and chain is logged
    chain_instance_mock.set_outputs.assert_called_once()
    actual_chain_outputs = chain_instance_mock.set_outputs.call_args[1]["outputs"]
    assert ("chain-output-key", "chain-output-value") in actual_chain_outputs.items()
    chain_api_module_mock.log_chain.assert_called_once_with(chain_instance_mock)

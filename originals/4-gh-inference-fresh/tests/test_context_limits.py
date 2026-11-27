"""Context and history regressions; execute on GitHub Actions only."""
import types
from unittest.mock import Mock, patch

import pytest

from gh_ai_runner import Agent, DEFAULT_CONTEXT, MODELS
from gh_ai_runner.validation import _validate
from test_large_models import worker_functions
from test_remote_agents import completion, execute_worker, remote_result


@pytest.mark.parametrize("model", list(MODELS))
def test_default_is_8k_or_supported_maximum(model):
    assert DEFAULT_CONTEXT == 8192
    assert MODELS[model]["n_ctx"] == min(DEFAULT_CONTEXT, MODELS[model]["max_n_ctx"])


@pytest.mark.parametrize("model", list(MODELS))
def test_advertised_ceiling_accepted_with_sufficient_hardware(model):
    cfg = MODELS[model]
    _validate(model, 512, .7, cfg["max_n_ctx"], runner_ram_gb=256, runner_disk_gb=100)
    with pytest.raises(ValueError, match="supports n_ctx up to"):
        _validate(model, 512, .7, cfg["max_n_ctx"] + 1, runner_ram_gb=256)


def test_full_qwen_starter_context_and_output_fit_default_budget():
    _validate("qwen0.5b", 8192, .7, 32768)
    with pytest.raises(ValueError, match="supports max_tokens up to"):
        _validate("qwen0.5b", 8193, .7, 32768)


def test_output_without_separate_publisher_cap_uses_remaining_context():
    _validate("llama", 16384, .7, 32768, runner_ram_gb=64)
    with pytest.raises(ValueError, match="must be less than"):
        _validate("llama", 32768, .7, 32768, runner_ram_gb=64)


def test_long_context_still_checks_ram():
    with pytest.raises(ValueError, match="RAM"):
        _validate("qwen7b", 512, .7, 131072)


@pytest.mark.parametrize("context,scaled", [(32768, False), (65536, True), (131072, True)])
def test_qwen_yarn_only_enabled_above_native_window(monkeypatch, tmp_path, context, scaled):
    namespace = worker_functions(monkeypatch, tmp_path)
    namespace["memory_info"] = lambda: {"MemAvailable": int(256e9)}
    model = namespace["MODEL_MAP"]["qwen7b"]
    cache = tmp_path / "model_cache"
    cache.mkdir()
    (cache / model["filename"]).write_bytes(b"cached mock weights")
    constructor = Mock()
    monkeypatch.setitem(__import__('sys').modules, "llama_cpp", types.SimpleNamespace(
        Llama=constructor, LLAMA_ROPE_SCALING_TYPE_YARN=2))
    namespace["get_local_model"]("qwen7b", context)
    options = constructor.call_args.kwargs
    assert options["n_ctx"] == context
    assert ("rope_scaling_type" in options) is scaled
    if scaled:
        assert options["rope_freq_scale"] == .25
        assert options["yarn_orig_ctx"] == 32768
        assert options["yarn_ext_factor"] == 1.0


def test_worker_rejects_context_above_model_ceiling_before_download(monkeypatch, tmp_path):
    namespace = worker_functions(monkeypatch, tmp_path)
    with patch("urllib.request.urlopen") as download, pytest.raises(ValueError, match="supports context"):
        namespace["get_local_model"]("tinyllama", 8192)
    download.assert_not_called()


def test_history_rolls_forward_instead_of_overflowing(tmp_path, monkeypatch):
    # Tokenizer stub counts words. Six thousand old tokens plus the newest
    # pair fit 8K; another 3K pair makes the oldest exchange fall out.
    history = [
        {"role": "user", "content": "old " * 3000},
        {"role": "assistant", "content": "old answer "},
        {"role": "user", "content": "recent " * 3000},
        {"role": "assistant", "content": "recent answer "},
        {"role": "user", "content": "latest " * 3000},
        {"role": "assistant", "content": "latest answer "},
    ]
    payload, _, llm, _ = execute_worker(
        tmp_path, monkeypatch, ["current question"], [completion()],
        history=history, max_history_turns=32)
    messages = llm.create_chat_completion.call_args.kwargs["messages"]
    assert messages[1]["content"].startswith("recent ")
    assert messages[-1]["content"] == "current question"
    assert payload["agent"]["results"][0]["dropped_history_turns"] == 1


def test_untrimmed_history_reaches_runtime_even_past_context(tmp_path, monkeypatch):
    history = [
        {"role": "user", "content": "old " * 4500},
        {"role": "assistant", "content": "answer " * 4500},
    ]
    payload, _, llm, _ = execute_worker(
        tmp_path, monkeypatch, ["current"], [completion()],
        history=history, trim_history=False)
    # The stub accepts overflow; the real runtime decides what to do with it.
    messages = llm.create_chat_completion.call_args.kwargs["messages"]
    assert messages[1:3] == history
    assert payload["agent"]["results"][0]["dropped_history_turns"] == 0
    assert len(payload["agent"]["history"]) == 4


def test_untrimmed_history_ignores_exchange_cap_at_both_ends(tmp_path, monkeypatch):
    history = [item for i in range(8) for item in (
        {"role": "user", "content": f"question {i}"},
        {"role": "assistant", "content": f"answer {i}"},
    )]
    payload, _, llm, _ = execute_worker(
        tmp_path, monkeypatch, ["next", "last"], [completion(), completion()],
        history=history, trim_history=False, max_history_turns=1)
    assert llm.create_chat_completion.call_args_list[0].kwargs["messages"][1:17] == history
    assert len(payload["agent"]["history"]) == 20


def test_default_history_cap_still_applies(tmp_path, monkeypatch):
    history = [item for i in range(8) for item in (
        {"role": "user", "content": f"question {i}"},
        {"role": "assistant", "content": f"answer {i}"},
    )]
    payload, _, llm, _ = execute_worker(
        tmp_path, monkeypatch, ["next"], [completion()], history=history)
    assert llm.create_chat_completion.call_args.kwargs["messages"][1]["content"] == "question 2"
    assert len(payload["agent"]["history"]) == 12


def test_untrimmed_transport_failure_does_not_discard_or_retry_history():
    runner = Mock()
    runner.submit.return_value.result.return_value = remote_result(["first answer"])
    agent = Agent(runner=runner, trim_history=False)
    agent.run("first")
    retained = list(agent._history)
    runner.submit.reset_mock()
    runner.submit.side_effect = ValueError("Combined workflow inputs exceed GitHub's limit")
    with pytest.raises(ValueError, match="Combined workflow inputs"):
        agent.run("next")
    assert agent._history == retained
    runner.submit.assert_called_once()
    assert runner.submit.call_args.kwargs["agent_config"]["trim_history"] is False


def test_runtime_failure_preserves_client_history():
    runner = Mock()
    runner.submit.return_value.result.return_value = remote_result(["first answer"])
    agent = Agent(runner=runner, trim_history=False)
    agent.run("first")
    retained = list(agent._history)
    runner.submit.return_value.result.side_effect = RuntimeError("model runtime rejected overflow")
    with pytest.raises(RuntimeError, match="runtime rejected"):
        agent.run("too much")
    assert agent._history == retained


@pytest.mark.parametrize("value", [None, "false", 0, 1])
def test_trim_history_requires_boolean(value):
    with pytest.raises(TypeError, match="trim_history"):
        Agent(runner=Mock(), trim_history=value)

"""Resource-aware catalog and embedded-worker regression tests."""
import hashlib
import io
import json
import sys
import types
from unittest.mock import Mock, patch

import pytest

from gh_ai_runner import MODELS, GitHubAIRunner, InferenceRequest, model_resources
from gh_ai_runner.runner import INFERENCE_SCRIPT, WORKFLOW_YAML
from gh_ai_runner.validation import _validate


def prepared_runner(**kwargs):
    client = GitHubAIRunner("token", verbose=False, metadata_path=None, **kwargs)
    client._prepared = True
    client._username = "tester"
    client._default_branch = "main"
    return client


def test_catalog_covers_up_to_16b_with_exact_download_metadata():
    assert len(MODELS) == 20
    assert max(model["parameters_b"] for model in MODELS.values()) == 16
    for model in MODELS.values():
        assert model["size_bytes"] > 0
        assert model["size_gb"] == model["size_bytes"] / 1e9
        assert len(bytes.fromhex(model["sha256"])) == 32
        assert model["quantization"] == "Q4_K_M"
        assert model["url"].endswith(model["filename"])


@pytest.mark.parametrize('model', ['glitter-12b', 'gemma-4-12b',
                                  'gemma-4-styletune-12b', 'rocinante-x-12b'])
def test_builtin_writers_resolve_offline_and_pin_the_download(model):
    from gh_inference import resolve_model
    with patch('gh_ai_runner.hub.requests.get') as get:
        assert resolve_model(model) == model
        get.assert_not_called()
    cfg = MODELS[model]
    revision = cfg['url'].split('/resolve/')[1].split('/')[0]
    assert len(bytes.fromhex(revision)) == 20
    _validate(model, 512, .7, 4096, runner_ram_gb=16)


def test_large_model_rejected_before_preparation_on_default_budget():
    client = prepared_runner()
    client.prepare = Mock()
    with pytest.raises(ValueError, match="configured budget is 8 GB"):
        client.submit("write a test", model="deepseek-coder16b")
    client.prepare.assert_not_called()


@pytest.mark.parametrize("model", list(MODELS))
def test_all_catalog_models_fit_16gb_budget_at_default_context(model):
    _validate(model, 128, .7, None, runner_ram_gb=16)


def test_large_model_dispatches_runner_label_and_skips_weight_cache():
    client = prepared_runner(runner_label="my-linux-32gb", runner_ram_gb=32, runner_disk_gb=40)
    response = Mock()
    with patch("gh_ai_runner.client.requests.post", return_value=response) as post:
        job = client.submit("write a test", model="deepseek-coder16b")
    config = json.loads(post.call_args.kwargs["json"]["inputs"]["config"])
    assert job.request["model"] == "deepseek-coder16b"
    assert config["runner_label"] == "my-linux-32gb"
    assert config["cache_weights"] is False
    assert config["model_sha256"] == MODELS["deepseek-coder16b"]["sha256"]
    assert "runs-on: ${{ fromJSON(inputs.config).runner_label }}" in WORKFLOW_YAML


def test_cache_reserves_disk_for_archive_and_large_models_avoid_double_storage():
    cached = model_resources("qwen7b")
    uncached = model_resources("qwen7b", cache=False)
    assert cached["disk_gb"] - uncached["disk_gb"] == pytest.approx(MODELS["qwen7b"]["size_gb"])
    large = model_resources("deepseek-coder16b")
    assert large["cache_weights"] is False
    assert large["disk_gb"] < 14
    with pytest.raises(ValueError, match="disk"):
        _validate("qwen7b", 128, .7, None, runner_ram_gb=16, runner_disk_gb=8)
    _validate("qwen7b", 128, .7, None, runner_ram_gb=16, runner_disk_gb=8, cache=False)


def test_context_increases_ram_but_not_model_storage():
    short = model_resources("qwen14b", n_ctx=2048)
    long = model_resources("qwen14b", n_ctx=8192)
    assert long["ram_gb"] > short["ram_gb"]
    assert long["disk_gb"] == short["disk_gb"]


def test_batch_rejects_large_request_before_dispatching_smaller_one():
    client = prepared_runner()
    with patch("gh_ai_runner.client.requests.post") as post, pytest.raises(ValueError):
        client.submit_batch([InferenceRequest("small"), InferenceRequest("large", model="qwen14b")])
    post.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"runner_ram_gb": 0}, {"runner_ram_gb": float("nan")},
                                   {"runner_disk_gb": -1}, {"runner_label": " "}])
def test_invalid_runner_configuration(kwargs):
    with pytest.raises(ValueError):
        GitHubAIRunner("token", metadata_path=None, **kwargs)


def worker_functions(monkeypatch, tmp_path):
    # Extract functions from the exact generated worker without launching its entrypoint.
    namespace = {}
    source = INFERENCE_SCRIPT.split("\nstarted = time.monotonic()", 1)[0]
    exec(compile(source, "run_inference.py", "exec"), namespace)  # noqa: S102
    namespace["config"] = {"cache_weights": False}
    namespace["memory_info"] = lambda: {"MemAvailable": int(32e9)}
    namespace["shutil"] = types.SimpleNamespace(disk_usage=lambda _: types.SimpleNamespace(free=int(40e9)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=Mock()))
    return namespace


def test_worker_rejects_real_ram_shortage_before_download(monkeypatch, tmp_path):
    namespace = worker_functions(monkeypatch, tmp_path)
    namespace["memory_info"] = lambda: {"MemAvailable": int(8e9)}
    with patch("urllib.request.urlopen") as download, pytest.raises(RuntimeError, match="RAM"):
        namespace["local_inference"]([], 128, .7, "deepseek-coder16b", None)
    download.assert_not_called()


def test_worker_rejects_real_disk_shortage_before_download(monkeypatch, tmp_path):
    namespace = worker_functions(monkeypatch, tmp_path)
    namespace["shutil"] = types.SimpleNamespace(disk_usage=lambda _: types.SimpleNamespace(free=int(3e9)))
    with patch("urllib.request.urlopen") as download, pytest.raises(RuntimeError, match="disk"):
        namespace["local_inference"]([], 128, .7, "deepseek-coder16b", None)
    download.assert_not_called()


@pytest.mark.parametrize("corrupted", [False, True])
def test_worker_download_checks_size_and_checksum(monkeypatch, tmp_path, corrupted):
    namespace = worker_functions(monkeypatch, tmp_path)
    payload = b"GGUF-test-weights"
    cfg = namespace["MODEL_MAP"]["tinyllama"]
    cfg["size_bytes"] = len(payload)
    cfg["sha256"] = hashlib.sha256(payload).hexdigest()
    downloaded = payload[:-1] + b"!" if corrupted else payload
    with patch("urllib.request.urlopen", return_value=io.BytesIO(downloaded)):
        if corrupted:
            with pytest.raises(RuntimeError, match="checksum"):
                namespace["local_inference"]([], 128, .7, "tinyllama", None)
            assert not (tmp_path / "model_cache" / cfg["filename"]).exists()
        else:
            namespace["local_inference"]([], 128, .7, "tinyllama", None)
            assert (tmp_path / "model_cache" / cfg["filename"]).read_bytes() == payload
    assert not list((tmp_path / "model_cache").glob("*.part"))

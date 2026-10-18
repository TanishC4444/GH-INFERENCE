import copy
from unittest.mock import Mock, patch

import pytest

from gh_inference import MODELS, clear_model_cache, list_quantizations, resolve_model
from gh_ai_runner.integrity import request_sha256


@pytest.fixture(autouse=True)
def isolated_catalog():
    original = copy.deepcopy(MODELS)
    clear_model_cache()
    yield
    MODELS.clear()
    MODELS.update(original)
    clear_model_cache()


def info():
    return {"sha": "a" * 40, "gguf": {"context_length": 32768}, "siblings": [
        {"rfilename": name, "size": 12, "lfs": {"sha256": "b" * 64}}
        for name in ["model-Q8_0.gguf", "sub/model-IQ4_XS-00001-of-00002.gguf",
                     "sub/model-IQ4_XS-00002-of-00002.gguf", "model-Q2_K-00001-of-00002.gguf",
                     "mmproj-F16.gguf"]]}


def test_all_variants_and_complete_shards():
    with patch("gh_ai_runner.hub.requests.get", return_value=Mock(json=info)) as get:
        variants = list_quantizations("example/model")
        assert len(variants) == 2
        key = resolve_model("example/model", "IQ4_XS")
        cfg = MODELS[key]
        assert cfg["size_bytes"] == 24 and len(cfg["files"]) == 2
        assert cfg["max_n_ctx"] == 32768
        assert all("/resolve/" + "a" * 40 in p["url"] for p in cfg["files"])
        assert get.call_count == 1
        assert resolve_model(key) == key


def test_missing_checksum_and_ambiguous_selection():
    metadata = info()
    metadata["siblings"][0]["lfs"] = {}
    with patch("gh_ai_runner.hub.requests.get", return_value=Mock(json=lambda: metadata)):
        with pytest.raises(ValueError, match="SHA-256"):
            resolve_model("example/model", "Q8_0")
        with pytest.raises(ValueError, match="Expected one"):
            resolve_model("example/model", "Q4_K_M")


def test_dynamic_manifest_integrity_roundtrip(tmp_path, monkeypatch):
    from test_remote_agents import execute_worker, completion
    with patch("gh_ai_runner.hub.requests.get", return_value=Mock(json=info)):
        key = resolve_model("example/model", "Q8_0")
    cache = tmp_path / "model_cache" / key
    cache.mkdir(parents=True)
    (cache / "model-Q8_0.gguf").write_bytes(b"mock weights")
    payload, constructor, llm, job = execute_worker(tmp_path, monkeypatch, ["Hello"],
        [completion("Hello back")], inference_options={"model": key})
    assert payload["integrity"]["request_sha256"] == request_sha256(job.request)
    assert "model_spec" in job.request
    assert payload["agent"]["results"][0]["output"]

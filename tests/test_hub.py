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


@pytest.mark.parametrize("corrupted", [False, True])
def test_worker_downloads_all_shards_and_rejects_corruption(tmp_path, monkeypatch, corrupted):
    import hashlib
    import io
    from test_large_models import worker_functions
    namespace = worker_functions(monkeypatch, tmp_path)
    parts = [b"first shard", b"second shard"]
    cfg = copy.deepcopy(MODELS["tinyllama"])
    cfg["files"] = [{"filename": f"sub/model-{i + 1:05d}-of-00002.gguf",
                     "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                     "url": f"https://huggingface.co/example/model/resolve/commit/{i}.gguf"}
                    for i, data in enumerate(parts)]
    namespace["config"]["model_spec"] = cfg
    downloaded = [parts[0], b"corrupt" if corrupted else parts[1]]
    with patch("urllib.request.urlopen", side_effect=[io.BytesIO(data) for data in downloaded]) as get:
        if corrupted:
            with pytest.raises(RuntimeError, match="checksum"):
                namespace["get_local_model"]("dynamic", 1024)
        else:
            first = namespace["get_local_model"]("dynamic", 1024)
            assert namespace["get_local_model"]("dynamic", 1024) is first
            assert namespace["_MODEL_STATS"][("dynamic", 1024)]["weights_cached"] is False
        assert get.call_count == 2
    assert not list((tmp_path / "model_cache").rglob("*.part"))


def test_unknown_context_is_reported_without_inventing_ceiling():
    metadata = info()
    metadata.pop("gguf")
    with patch("gh_ai_runner.hub.requests.get", return_value=Mock(json=lambda: metadata)):
        key = resolve_model("example/model", "Q8_0")
    assert MODELS[key]["max_n_ctx"] is None
    assert MODELS[key]["n_ctx"] == 8192


def test_unsafe_filename_is_rejected():
    metadata = info()
    metadata["siblings"][0]["rfilename"] = "../model-Q8_0.gguf"
    with patch("gh_ai_runner.hub.requests.get", return_value=Mock(json=lambda: metadata)):
        with pytest.raises(ValueError, match="Unsafe"):
            resolve_model("example/model", "Q8_0")

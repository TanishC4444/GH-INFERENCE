"""Release evidence must stay on the branch whose code was checked."""
import base64
import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from gh_inference import __version__


@pytest.mark.parametrize("branch", ["main", "codex/reconcile-local-inference"])
def test_verification_records_results_on_triggering_branch(monkeypatch, tmp_path, branch):
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_REF_NAME", branch)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repository")
    monkeypatch.setenv("REPORT_TOKEN", "offline-test-token")
    response = Mock()
    script = Path(__file__).resolve().parents[1] / ".github/verify_release.py"
    monkeypatch.chdir(tmp_path)
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="passed", stderr="")), patch(
        "gh_inference.search_models", return_value=[]
    ), patch("gh_inference.list_quantizations", return_value={}), patch(
        "gh_inference.resolve_model", return_value="qwen0.5b"
    ), patch("urllib.request.urlopen", return_value=response) as upload:
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code == 0
    evidence = json.loads((tmp_path / "verification-results/123.json").read_text())
    if branch == "main":
        request = upload.call_args.args[0]
        payload = json.loads(request.data)
        assert payload["branch"] == branch
        assert json.loads(base64.b64decode(payload["content"])) == evidence
    else:
        upload.assert_not_called()
    assert evidence["source_sha"] == "a" * 40
    assert evidence["passed"] is True
    assert len(evidence["checks"]) == 3


@pytest.mark.parametrize("branch", ["main", "codex/reconcile-local-inference"])
@pytest.mark.parametrize("published", [False, True])
def test_model_smoke_uses_current_version_and_keeps_branch_evidence(monkeypatch, tmp_path, branch, published):
    script = Path(__file__).resolve().parents[1] / ".github/open_model_smoke.py"
    monkeypatch.chdir(tmp_path)
    # Exercise orchestration without subprocesses, network, model weights or inference.
    worker = '''import json
json.dump({"agent": {"results": [
    {"output": "4"},
    {"output": "blue", "model_reused": True, "timings": {"load_seconds": 0}}
]}}, open("result.json", "w"))
'''
    with patch.dict(os.environ, {
        "GITHUB_SHA": "b" * 40, "GITHUB_REF_NAME": branch,
        "GITHUB_RUN_ID": "456", "GITHUB_REPOSITORY": "owner/repository",
        "REPORT_TOKEN": "offline-test-token",
        "GH_INFERENCE_TEST_PYPI_VERSION": __version__ if published else "",
    }), patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="passed", stderr="")) as run, patch(
        "gh_inference.resolve_model", return_value="qwen0.5b"
    ), patch("gh_ai_runner.runner.INFERENCE_SCRIPT", worker), patch(
        "urllib.request.urlopen", return_value=Mock()
    ) as upload, patch("gh_inference.__file__", "/test/site-packages/gh_inference/__init__.py"):
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code == 0
    wheel_check = next(call.args[0] for call in run.call_args_list if "-c" in call.args[0])
    assert f"assert __version__ == {__version__!r}" in wheel_check[-1]
    evidence = json.loads((tmp_path / "verification-results/smoke-456.json").read_text())
    assert evidence["passed"] is True
    assert len(evidence["inference"]) == 3
    assert evidence["distribution_source"] == ("pypi" if published else "source-built-wheel")
    install = next(call.args[0] for call in run.call_args_list if "install" in call.args[0])
    if published:
        assert "gh-inference==" + __version__ in install
        assert "https://pypi.org/simple" in install
    if branch == "main" and not published:
        payload = json.loads(upload.call_args.args[0].data)
        assert payload["branch"] == branch
        assert json.loads(base64.b64decode(payload["content"])) == evidence
    else:
        upload.assert_not_called()

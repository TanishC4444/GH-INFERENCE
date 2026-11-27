"""Release evidence must stay on the branch whose code was checked."""
import base64
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


@pytest.mark.parametrize("branch", ["main", "codex/reconcile-local-inference"])
def test_verification_records_results_on_triggering_branch(monkeypatch, branch):
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_REF_NAME", branch)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repository")
    monkeypatch.setenv("REPORT_TOKEN", "offline-test-token")
    response = Mock()
    script = Path(__file__).resolve().parents[1] / ".github/verify_release.py"
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="passed", stderr="")), patch(
        "gh_inference.search_models", return_value=[]
    ), patch("gh_inference.list_quantizations", return_value={}), patch(
        "gh_inference.resolve_model", return_value="qwen0.5b"
    ), patch("urllib.request.urlopen", return_value=response) as upload:
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code == 0
    request = upload.call_args.args[0]
    payload = json.loads(request.data)
    assert payload["branch"] == branch
    evidence = json.loads(base64.b64decode(payload["content"]))
    assert evidence["source_sha"] == "a" * 40
    assert evidence["passed"] is True
    assert len(evidence["checks"]) == 3

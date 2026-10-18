"""Real models run here only on GitHub Actions."""
import base64
import glob
import json
import os
import subprocess
import sys
import traceback
import urllib.request

from gh_inference import MODELS, resolve_model
from gh_ai_runner.runner import INFERENCE_SCRIPT

report = {"source_sha": os.environ["GITHUB_SHA"], "inference": []}
failed = False
try:
    subprocess.run([sys.executable, "-m", "build"], check=True)
    subprocess.run([sys.executable, "-m", "venv", "/tmp/gh-wheel"], check=True)
    subprocess.run(["/tmp/gh-wheel/bin/pip", "install", *glob.glob("dist/*.whl")], check=True)
    for command in [["/tmp/gh-wheel/bin/python", "-c", "from gh_inference import Agent, search_models; from gh_ai_runner import __version__; assert __version__ == '0.5.0'"],
                    ["/tmp/gh-wheel/bin/gh-inference", "--help"], ["/tmp/gh-wheel/bin/gh-inference", "models"]]:
        result = subprocess.run(command, cwd="/tmp", capture_output=True, text=True, check=True)
    report["installed_wheel"] = True
    selections = [("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "Q4_K_M"),
                  ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "Q8_0"),
                  ("bartowski/SmolLM2-135M-Instruct-GGUF", "Q4_K_M")]
    for repo, quant in selections:
        key = resolve_model(repo, quant)
        cfg = MODELS[key]
        prompt = "What is 2 plus 2? Answer briefly."
        os.environ.update({"JOB_ID": "open-smoke", "ATTEMPT": "1", "BACKEND": "local",
                           "LOCAL_MODEL": key, "PROMPT": prompt, "SYSTEM": "You are a helpful assistant.",
                           "CACHE": "true", "CONFIG_JSON": json.dumps({
                               "model": key, "model_spec": cfg, "max_tokens": 32,
                               "n_ctx": 1024, "temperature": 0, "cache_weights": True,
                               "agent": {"tasks": ["Name one color."], "track_time": True,
                                         "remember": True, "refine_prompt": False}})})
        exec(compile(INFERENCE_SCRIPT, "remote_worker.py", "exec"), {})
        payload = json.load(open("result.json"))
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        results = payload["agent"]["results"]
        assert all(item["output"].strip() for item in results)
        assert results[1]["model_reused"] and results[1]["timings"]["load_seconds"] == 0
        report["inference"].append({"repo": repo, "quantization": quant,
                                    "prompt": prompt, "followup": "Name one color.", "results": results})
except Exception:
    failed = True
    report["error"] = traceback.format_exc()
finally:
    report["passed"] = not failed
    data = json.dumps(report, indent=2).encode()
    payload = {"message": "Record real model smoke verification", "content": base64.b64encode(data).decode()}
    request = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/contents/verification-results/smoke-{os.environ['GITHUB_RUN_ID']}.json",
        data=json.dumps(payload).encode(), method="PUT", headers={
            "Authorization": "Bearer " + os.environ["REPORT_TOKEN"], "Accept": "application/vnd.github+json",
            "Content-Type": "application/json"})
    urllib.request.urlopen(request, timeout=30).close()
    print(data.decode())
sys.exit(int(failed))

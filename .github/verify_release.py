"""GitHub-only verification. Publish a readable report without tokens or prompts."""
import base64
import json
import os
import subprocess
import sys
import traceback
import urllib.request

report = {"source_sha": os.environ["GITHUB_SHA"], "checks": []}
failed = False
try:
    for command in [[sys.executable, "-m", "pytest", "-q"],
                    [sys.executable, "-m", "build"],
                    [sys.executable, "-m", "twine", "check", "dist/*"]]:
        if command[-1] == "dist/*":
            import glob
            command = command[:-1] + glob.glob("dist/*")
        result = subprocess.run(command, capture_output=True, text=True)
        report["checks"].append({"command": command, "exit": result.returncode,
                                 "output": (result.stdout + result.stderr)[-18000:]})
        failed |= result.returncode != 0
    from gh_inference import search_models, list_quantizations, resolve_model, MODELS
    report["families"] = {family: search_models(family, limit=3) for family in
                          ["Qwen3", "Llama", "Gemma", "Phi", "Mistral", "DeepSeek", "SmolLM"]}
    variants = list_quantizations("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    report["qwen_variants"] = variants
    for quant in ["Q4_K_M", "Q8_0"]:
        key = resolve_model("Qwen/Qwen2.5-0.5B-Instruct-GGUF", quant)
        report[quant] = MODELS[key]
except Exception:
    failed = True
    report["error"] = traceback.format_exc()
finally:
    report["passed"] = not failed
    data = json.dumps(report, indent=2).encode()
    path = f"verification-results/{os.environ['GITHUB_RUN_ID']}.json"
    payload = {"message": "Record GitHub release verification", "content": base64.b64encode(data).decode()}
    request = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/contents/{path}",
        data=json.dumps(payload).encode(), method="PUT",
        headers={"Authorization": "Bearer " + os.environ["REPORT_TOKEN"],
                 "Accept": "application/vnd.github+json", "Content-Type": "application/json"})
    urllib.request.urlopen(request, timeout=30).close()
    print(data.decode())
sys.exit(int(failed))

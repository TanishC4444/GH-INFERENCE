import hashlib
import inspect

from .models import DISK_RESERVE_GB, MODELS, RUNTIME_OVERHEAD_GB
from .worker_agent import AGENT_SCRIPT
from .pipeline import normalize_pipeline
from .limits import Limits

RUNTIME_VERSION = "0.3.36"

INFERENCE_SCRIPT = r'''
import hashlib
import json
import os
import importlib.metadata
import platform
import resource
import shutil
import time
import urllib.error
import urllib.request
import warnings

warnings.filterwarnings("ignore")

MODEL_MAP = __MODEL_MAP__
_LOADED_MODELS = {}
_MODEL_STATS = {}
RUNTIME_OVERHEAD_GB = __RUNTIME_OVERHEAD_GB__
DISK_RESERVE_GB = __DISK_RESERVE_GB__
DEFAULT_LIMITS = __DEFAULT_LIMITS__


def memory_info():
    values = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as stream:
            for line in stream:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def resource_snapshot(started):
    memory = memory_info()
    disk = shutil.disk_usage(".")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak *= 1024
    try:
        runtime_version = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        runtime_version = "unavailable"
    return {
        "runtime_version": runtime_version,
        "cpu_count": os.cpu_count() or 1,
        "memory_total_bytes": memory.get("MemTotal", 0),
        "memory_available_bytes": memory.get("MemAvailable", 0),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "peak_memory_bytes": peak,
        "duration_seconds": round(time.monotonic() - started, 3),
        "runner_os": platform.platform(),
        "cpu_seconds": time.process_time() - cpu_started,
    }


def provider_inference(messages, max_tokens, temperature, model, base_url):
    base_url = base_url.rstrip("/")
    api_key = os.environ.get("PROVIDER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GitHub Actions secret 'GH_AI_PROVIDER_API_KEY' is missing"
        )
    payload = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        **config.get("generation_options", {}),
    }).encode()
    request = urllib.request.Request(
        base_url + "/chat/completions", data=payload,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.get("limits", {}).get("provider_timeout_seconds", 300)) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"Provider returned HTTP {error.code}: {detail}") from error


def model_config(local_model):
    return config.get("model_spec") or MODEL_MAP[local_model]


def get_local_model(local_model, requested_n_ctx):
    from llama_cpp import Llama

    cfg = model_config(local_model)
    policy = dict(DEFAULT_LIMITS)
    policy.update(config.get("limits", {}))
    n_ctx = int(requested_n_ctx) if requested_n_ctx else cfg["n_ctx"]
    if n_ctx < 1 or (policy["enforce_model_limits"] and cfg["max_n_ctx"] and n_ctx > cfg["max_n_ctx"]):
        raise ValueError(f"{local_model} supports context up to {cfg['max_n_ctx']:,} tokens")
    key = (local_model, n_ctx)
    if key in _LOADED_MODELS:
        return _LOADED_MODELS[key]
    cache_dir = os.path.join("model_cache", local_model) if "files" in cfg else "model_cache"
    files = cfg.get("files", [cfg])
    paths = [os.path.join(cache_dir, item["filename"]) for item in files]
    model_path = paths[0]
    required_ram = cfg["size_gb"] + policy["runtime_overhead_gb"] + n_ctx / 1024 * (
        cfg["ram_per_1k_ctx_gb"] if policy["ram_per_1k_ctx_gb"] is None else policy["ram_per_1k_ctx_gb"])
    available_ram = memory_info().get("MemAvailable")
    if policy["check_resources"] and available_ram and required_ram * 1e9 > available_ram:
        raise RuntimeError(
            f"Model/context needs approximately {required_ram:.1f} GB RAM, "
            f"but only {available_ram / 1e9:.1f} GB is available. Use a smaller model/context or a larger runner."
        )
    # Cached files already occupy disk; cache upload can need another full copy.
    required_disk = (
        sum(item["size_bytes"] for item, path in zip(files, paths) if not os.path.exists(path)) / 1e9
        + cfg["size_gb"] * int(config.get("cache_weights", False))
        + policy["disk_reserve_gb"]
    )
    if policy["check_resources"] and required_disk * 1e9 > shutil.disk_usage(".").free:
        raise RuntimeError(f"Model/cache needs approximately {required_disk:.1f} GB additional free disk")
    cached_weights = all(os.path.exists(path) for path in paths)
    download_started = time.monotonic()
    for file_cfg, file_path in zip(files, paths):
        if os.path.exists(file_path):
            continue
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        partial_path = file_path + ".part"
        try:
            with urllib.request.urlopen(file_cfg["url"], timeout=policy["download_timeout_seconds"]) as source:
                with open(partial_path, "wb") as target:
                    digest = hashlib.sha256()
                    size = 0
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            if size != file_cfg["size_bytes"] or digest.hexdigest() != file_cfg["sha256"]:
                raise RuntimeError("Downloaded model size/checksum does not match the catalog")
            os.replace(partial_path, file_path)
        finally:
            if os.path.exists(partial_path):
                os.remove(partial_path)
    cpu_count = os.cpu_count() or 1
    threads = cpu_count
    download_seconds = time.monotonic() - download_started
    load_started = time.monotonic()
    context_options = {}
    original_context = cfg.get("yarn_original_context")
    if original_context and n_ctx > original_context:
        from llama_cpp import LLAMA_ROPE_SCALING_TYPE_YARN
        # Qwen's published long-context recipe uses static factor-4 YaRN.
        # Leave short contexts and other models' GGUF scaling metadata alone.
        context_options = {
            "rope_scaling_type": LLAMA_ROPE_SCALING_TYPE_YARN,
            "rope_freq_scale": original_context / cfg["max_n_ctx"],
            "yarn_orig_ctx": original_context,
            "yarn_ext_factor": 1.0,
        }
    options = {"n_threads": threads, "n_gpu_layers": 0, "verbose": False, **context_options,
               **config.get("model_options", {})}
    if isinstance(options.get("rope_scaling_type"), str):
        import llama_cpp
        options["rope_scaling_type"] = getattr(
            llama_cpp, "LLAMA_ROPE_SCALING_TYPE_" + options["rope_scaling_type"].upper())
    template_options = options.pop("chat_template_options", None)
    llm = Llama(model_path=model_path, n_ctx=n_ctx,
                use_mmap=False, use_mlock=True, **options)
    if template_options is not None:
        from functools import partial
        handler = llm._chat_handlers.get("chat_template.default")
        if handler is None:
            raise ValueError("chat_template_options requires an embedded default chat template")
        llm.chat_handler = partial(handler, **template_options)
    _LOADED_MODELS[key] = llm
    _MODEL_STATS[key] = {
        "weights_cached": cached_weights,
        "download_seconds": download_seconds,
        "load_seconds": time.monotonic() - load_started,
    }
    return llm


def local_inference(messages, max_tokens, temperature, local_model, requested_n_ctx):
    llm = get_local_model(local_model, requested_n_ctx)
    if max_tokens is None and config.get("limits", {}).get("enforce_model_limits", True):
        max_tokens = model_config(local_model).get("max_output_tokens")
    return llm.create_chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=temperature,
        **config.get("generation_options", {})
    )


started = time.monotonic()
cpu_started = time.process_time()
job_id = os.environ["JOB_ID"]
attempt = int(os.environ.get("ATTEMPT", "1"))
backend = os.environ.get("BACKEND", "local")
config = json.loads(os.environ.get("CONFIG_JSON", "{}"))
provider_name = config.get("provider_name", "local")
model = config["model"]
messages = [
    {"role": "system", "content": os.environ.get("SYSTEM", "You are a helpful assistant.")},
    {"role": "user", "content": os.environ["PROMPT"]},
]
max_tokens = config.get("max_tokens", 512)
temperature = float(config.get("temperature", 0.7))
request_document = {
    "prompt": os.environ["PROMPT"],
    "system": os.environ.get("SYSTEM", "You are a helpful assistant."),
    "backend": backend,
    "model": model,
    "local_model": os.environ.get("LOCAL_MODEL", "tinyllama"),
    "cache": os.environ.get("CACHE", "true") == "true",
    "max_tokens": max_tokens,
    "temperature": temperature,
    "n_ctx": config.get("n_ctx"),
    "provider_name": provider_name,
    "provider_url": config.get("provider_url", ""),
}
if "agent" in config:
    request_document["agent"] = config["agent"]
for key in ("limits", "model_options", "generation_options", "model_spec"):
    if key in config:
        request_document[key] = config[key]
request_sha256 = hashlib.sha256(
    json.dumps(request_document, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

result = {
    "schema_version": 1, "job_id": job_id, "attempt": attempt,
    "correlation_id": f"{job_id}-{attempt}", "model": model,
    "provider": provider_name if backend == "provider" else "local",
    "output": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    "resources": {},
    "integrity": {
        "algorithm": "sha256", "request_sha256": request_sha256, "output_sha256": "",
    },
}

try:
    if "agent" in config:
        operation = (run_pipeline if "pipeline" in config["agent"] else
                     run_discussion if "discussion" in config["agent"] else run_agent)
        result["agent"] = operation(
            config["agent"], messages, config)
        result["output"] = result["agent"]["results"][-1]["output"]
        result["usage"] = result["agent"]["usage"]
    else:
        response = (
        provider_inference(
            messages, max_tokens, temperature, model, config.get("provider_url", "")
        )
        if backend == "provider"
        else local_inference(
            messages, max_tokens, temperature,
            os.environ.get("LOCAL_MODEL", "tinyllama"), config.get("n_ctx"),
        )
    )
        result["output"] = response["choices"][0]["message"]["content"] or ""
        usage = response.get("usage") or {}
        result["usage"] = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
except Exception as error:
    result["error"] = {"type": type(error).__name__, "message": str(error)}
    raise
finally:
    if "agent" in result:
        result["integrity"]["agent_sha256"] = hashlib.sha256(
            json.dumps(result["agent"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    result["integrity"]["output_sha256"] = hashlib.sha256(result["output"].encode()).hexdigest()
    result["resources"] = resource_snapshot(started)
    with open("result.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
'''.replace("\nstarted = time.monotonic()", inspect.getsource(normalize_pipeline) + "\n" + AGENT_SCRIPT + "\nstarted = time.monotonic()", 1).replace(
    "__MODEL_MAP__", repr(MODELS)
).replace(
    "__RUNTIME_OVERHEAD_GB__", repr(RUNTIME_OVERHEAD_GB)
).replace("__DISK_RESERVE_GB__", repr(DISK_RESERVE_GB)).replace("__DEFAULT_LIMITS__", repr(Limits().to_dict()))


WORKFLOW_YAML = """\
name: AI Inference
run-name: AI inference ${{ inputs.job_id }}-${{ inputs.attempt }}
on:
  workflow_dispatch:
    inputs:
      job_id:          { description: "Correlation ID", required: true }
      attempt:         { description: "Attempt number", required: true, default: "1" }
      prompt:          { description: "User prompt", required: true }
      system:          { description: "System prompt", required: false, default: "You are a helpful assistant." }
      backend:         { description: "local or provider", required: true, default: "local" }
      local_model:     { description: "Local model key", required: false, default: "tinyllama" }
      cache:           { description: "Cache weights", required: false, default: "true" }
      config:          { description: "Non-secret inference settings JSON", required: true }

jobs:
  inference:
    runs-on: ${{ fromJSON(inputs.config).runner_label }}
    timeout-minutes: ${{ fromJSON(inputs.config).job_timeout_minutes || 30 }}
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Cache venv
        id: cache-venv
        if: ${{ inputs.backend == 'local' }}
        uses: actions/cache@v4
        with:
          path: .venv
          key: venv-${{ runner.os }}-${{ runner.arch }}-python311-llama-cpp-__RUNTIME_VERSION__-v5

      - name: Cache model weights
        if: ${{ inputs.backend == 'local' && fromJSON(inputs.config).cache_weights }}
        uses: actions/cache@v4
        with:
          path: model_cache
          key: gguf-${{ inputs.local_model }}-${{ fromJSON(inputs.config).model_sha256 }}

      - name: Install model runtime on GitHub
        if: ${{ inputs.backend == 'local' && steps.cache-venv.outputs.cache-hit != 'true' }}
        run: |
          python -m venv .venv
          .venv/bin/pip install --only-binary=llama-cpp-python \
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
            'llama-cpp-python==__RUNTIME_VERSION__' -q || \
            CMAKE_ARGS="-DGGML_METAL=off" .venv/bin/pip install 'llama-cpp-python==__RUNTIME_VERSION__' -q

      - name: Run inference on GitHub
        env:
          JOB_ID:           ${{ inputs.job_id }}
          ATTEMPT:          ${{ inputs.attempt }}
          PROMPT:           ${{ inputs.prompt }}
          SYSTEM:           ${{ inputs.system }}
          BACKEND:          ${{ inputs.backend }}
          LOCAL_MODEL:      ${{ inputs.local_model }}
          CACHE:            ${{ inputs.cache }}
          CONFIG_JSON:      ${{ inputs.config }}
          PROVIDER_API_KEY: ${{ inputs.backend == 'provider' && secrets.GH_AI_PROVIDER_API_KEY || '' }}
        run: |
          if [ "$BACKEND" = "local" ]; then
            .venv/bin/python run_inference.py
          else
            python run_inference.py
          fi

      - name: Upload structured result
        if: ${{ always() && hashFiles('result.json') != '' }}
        uses: actions/upload-artifact@v4
        with:
          name: ai-output-${{ inputs.job_id }}-${{ inputs.attempt }}
          path: result.json
          retention-days: ${{ fromJSON(inputs.config).artifact_retention_days || 3 }}
"""


WORKFLOW_YAML = WORKFLOW_YAML.replace("__RUNTIME_VERSION__", RUNTIME_VERSION)


def _script_hash():
    return hashlib.sha256(INFERENCE_SCRIPT.encode()).hexdigest()


def _workflow_hash():
    return hashlib.sha256(WORKFLOW_YAML.encode()).hexdigest()

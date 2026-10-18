"""Metadata-only discovery; model weights are downloaded by GitHub workers."""
from functools import lru_cache
import hashlib
import json
import re
from urllib.parse import quote

import requests

from .models import MODELS


def search_models(query="", limit=30):
    """Search public Hugging Face GGUF repositories (no weight downloads)."""
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    response = requests.get("https://huggingface.co/api/models", params={
        "search": query, "filter": "gguf", "limit": limit,
        "sort": "downloads", "direction": -1,
    }, timeout=30)
    response.raise_for_status()
    return [item["id"] for item in response.json()]


def _repository(model):
    if model in MODELS:
        return MODELS[model]["url"].split("huggingface.co/", 1)[1].split("/resolve/", 1)[0]
    if not isinstance(model, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", model):
        raise ValueError("model must be a catalog alias or Hugging Face owner/repository")
    return model


@lru_cache(maxsize=128)
def _info(repo, revision):
    response = requests.get(
        f"https://huggingface.co/api/models/{repo}/revision/{quote(revision, safe='')}",
        params={"blobs": "true"}, timeout=30)
    response.raise_for_status()
    return response.json()


def clear_model_cache():
    """Refresh cached Hub metadata after a publisher updates a branch."""
    _info.cache_clear()


def list_quantizations(model, revision="main"):
    """List every GGUF file, grouping complete split files as a single variant.

    Return exact filenames as keys, so unusual or future quantizations work too.
    An incomplete split is excluded. No architecture or quantization whitelist.
    """
    info = _info(_repository(model), revision)
    grouped = {}
    for item in info.get("siblings", []):
        name = item["rfilename"]
        if not name.lower().endswith(".gguf") or "mmproj" in name.lower():
            continue
        split = re.search(r"-([0-9]{5})-of-([0-9]{5})\.gguf$", name, re.I)
        key = name[:split.start()] + ".gguf" if split else name
        grouped.setdefault(key, []).append(item)
    result = {}
    for name, parts in grouped.items():
        parts.sort(key=lambda item: item["rfilename"])
        split = re.search(r"-([0-9]{5})-of-([0-9]{5})\.gguf$", parts[0]["rfilename"], re.I)
        if split:
            count = int(split[2])
            if [int(re.search(r"-([0-9]{5})-of-", p["rfilename"])[1]) for p in parts] != list(range(1, count + 1)):
                continue
        result[name] = {
            "files": [p["rfilename"] for p in parts],
            "size_bytes": sum(p.get("size") or p.get("lfs", {}).get("size", 0) for p in parts),
        }
    return result


def resolve_model(model, quantization=None, *, filename=None, revision="main"):
    """Return an immutable-content catalog key usable by Agent and all clients.

    Repositories default to Q4_K_M. Use filename for ambiguous multi-model repos.
    Requires public, ungated LFS GGUF weights with published SHA-256 metadata.
    """
    if model in MODELS and quantization is None and filename is None and revision == "main":
        return model
    repo = _repository(model)
    info = _info(repo, revision)
    variants = list_quantizations(model, revision)
    if filename:
        matches = [name for name, value in variants.items() if filename == name or filename in value["files"]]
    else:
        quant = quantization or "Q4_K_M"
        if not re.fullmatch(r"[\w]+", quant):
            raise ValueError("quantization must be a label such as Q4_K_M; use filename for exact selection")
        matches = [name for name in variants if re.search(r"(?:^|[./_-])" + re.escape(quant) + r"(?:[.-]|$)", name, re.I)]
    if len(matches) != 1:
        raise ValueError(f"Expected one GGUF variant, found {len(matches)}. Select filename from list_quantizations({repo!r}).")
    siblings = {p["rfilename"]: p for p in info["siblings"]}
    commit = info.get("sha", "")
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Hub response lacks a pinned commit SHA")
    files = []
    for name in variants[matches[0]]["files"]:
        item = siblings[name]
        digest = item.get("lfs", {}).get("sha256", "")
        size = item.get("size") or item.get("lfs", {}).get("size")
        if not re.fullmatch(r"[a-f0-9]{64}", digest) or not size or size < 1:
            raise ValueError(f"Missing LFS size/SHA-256 metadata for {name}")
        files.append({"filename": name, "size_bytes": size, "sha256": digest,
                      "url": f"https://huggingface.co/{repo}/resolve/{commit}/{quote(name, safe='/')}"})
    original = MODELS.get(model, {})
    context = original.get("max_n_ctx") or info.get("gguf", {}).get("context_length")
    if type(context) is not int or context < 1:
        context = None
    size = sum(p["size_bytes"] for p in files)
    cfg = {"name": repo, "parameters_b": original.get("parameters_b"),
           "quantization": quantization or matches[0], "files": files,
           "filename": files[0]["filename"], "url": files[0]["url"],
           "size_bytes": size, "size_gb": size / 1e9,
           "n_ctx": min(8192, context) if context else 8192, "max_n_ctx": context,
           "max_output_tokens": original.get("max_output_tokens"),
           "yarn_original_context": original.get("yarn_original_context"),
           "context_source": original.get("context_source", f"https://huggingface.co/{repo}"),
           "ram_per_1k_ctx_gb": original.get("ram_per_1k_ctx_gb", .25)}
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    cfg["sha256"] = digest
    key = "hf-" + digest[:24]
    # Content-addressed keys never change; independent submissions can safely reuse them.
    MODELS.setdefault(key, cfg)
    return key

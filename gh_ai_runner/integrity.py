from __future__ import annotations

import hashlib
import json


def request_payload(request: dict) -> dict:
    provider = request.get("provider")
    payload = {
        "prompt": request["prompt"],
        "system": request["system"],
        "backend": "provider" if provider else "local",
        "model": request["model"],
        "local_model": request["local_model"],
        "cache": bool(request["cache"]),
        "max_tokens": int(request["max_tokens"]) if request["max_tokens"] is not None else None,
        "temperature": float(request["temperature"]),
        "n_ctx": request["n_ctx"],
        "provider_name": provider.name if provider else "local",
        "provider_url": provider.base_url if provider else "",
    }
    if "agent" in request:
        payload["agent"] = request["agent"]
    for key in ("limits", "model_options", "generation_options", "model_spec"):
        if key in request:
            payload[key] = request[key]
    return payload


def sha256_json(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def request_sha256(request: dict) -> str:
    return sha256_json(request_payload(request))


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()

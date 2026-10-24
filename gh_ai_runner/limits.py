"""Editable project policy. GitHub and backend constraints remain external."""
from dataclasses import asdict, dataclass, replace
import json
import math
from typing import Optional


@dataclass(frozen=True)
class Limits:
    enforce_model_limits: bool = True
    enforce_output_budget: bool = True
    check_resources: bool = True
    max_prompt_chars: Optional[int] = 60000
    max_workflow_input_chars: Optional[int] = 65000
    max_prompts: Optional[int] = 32
    min_temperature: Optional[float] = 0.0
    max_temperature: Optional[float] = 2.0
    context_reserve_tokens: int = 128
    message_overhead_tokens: int = 32
    max_cached_model_gb: Optional[float] = 6.0
    runtime_overhead_gb: float = 1.5
    disk_reserve_gb: float = 2.0
    ram_per_1k_ctx_gb: Optional[float] = None
    job_timeout_minutes: int = 30
    artifact_retention_days: int = 3
    download_timeout_seconds: float = 30
    provider_timeout_seconds: float = 300
    refiner_min_tokens: int = 128
    refiner_max_tokens: int = 512
    refiner_context_fraction: float = .25

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key in ("enforce_model_limits", "enforce_output_budget", "check_resources"):
                if type(value) is not bool:
                    raise TypeError(f"{key} must be a bool")
                continue
            if value is None:
                if key not in ("max_prompt_chars", "max_workflow_input_chars", "max_prompts",
                               "min_temperature", "max_temperature", "max_cached_model_gb",
                               "ram_per_1k_ctx_gb"):
                    raise ValueError(f"{key} cannot be None")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            if key not in ("min_temperature", "max_temperature") and value < 0:
                raise ValueError(f"{key} cannot be negative")
            if key in ("max_prompt_chars", "max_workflow_input_chars", "max_prompts",
                       "context_reserve_tokens", "message_overhead_tokens", "job_timeout_minutes",
                       "artifact_retention_days", "refiner_min_tokens", "refiner_max_tokens"):
                if type(value) is not int:
                    raise ValueError(f"{key} must be an integer")
            if key in ("job_timeout_minutes", "artifact_retention_days", "download_timeout_seconds",
                       "provider_timeout_seconds", "refiner_max_tokens", "refiner_context_fraction") and value <= 0:
                raise ValueError(f"{key} must be positive")
        if (self.min_temperature is not None and self.max_temperature is not None
                and self.min_temperature > self.max_temperature):
            raise ValueError("min_temperature cannot exceed max_temperature")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def unrestricted(cls, **overrides):
        """Remove project policy ceilings; does not disable external limits."""
        return replace(cls(
            enforce_model_limits=False, enforce_output_budget=False, check_resources=False,
            max_prompt_chars=None, max_workflow_input_chars=None, max_prompts=None,
            min_temperature=None, max_temperature=None, max_cached_model_gb=None,
            context_reserve_tokens=0, message_overhead_tokens=0,
            job_timeout_minutes=360,
        ), **overrides)


def validate_options(model_options=None, generation_options=None):
    for label, options, reserved in (
        ("model_options", model_options, {"model_path", "n_ctx", "n_gpu_layers", "use_mmap", "use_mlock"}),
        ("generation_options", generation_options, {"messages", "max_tokens", "temperature", "stream"}),
    ):
        if options is None:
            continue
        if not isinstance(options, dict) or any(not isinstance(key, str) for key in options):
            raise TypeError(f"{label} must be a dict with string keys")
        template = options.get("chat_template_options") if label == "model_options" else None
        if template is not None and (not isinstance(template, dict) or
                any(not isinstance(key, str) for key in template) or
                set(template) - {"enable_thinking"} or
                any(type(value) is not bool for value in template.values())):
            raise ValueError("chat_template_options accepts only boolean enable_thinking")
        if reserved.intersection(options):
            raise ValueError(f"{label} cannot override {sorted(reserved.intersection(options))}; use the named settings")
        json.dumps(options, allow_nan=False)

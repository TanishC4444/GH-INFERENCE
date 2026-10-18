import math

from .models import DEFAULT_CONTEXT, MODELS, RUNNER_DISK_GB, RUNNER_RAM_GB, model_resources
from .limits import Limits


def _validate(model, max_tokens, temperature, n_ctx, *, runner_ram_gb=RUNNER_RAM_GB,
              runner_disk_gb=RUNNER_DISK_GB, cache=True, limits=None):
    """Validate portable limits before any repository or workflow changes."""
    if model is not None and model not in MODELS:
        raise ValueError(f"model must be one of: {list(MODELS)}")
    limits = limits if limits is not None else Limits()
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
        raise ValueError("max_tokens must be a positive integer or None")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or (limits.min_temperature is not None and temperature < limits.min_temperature)
            or (limits.max_temperature is not None and temperature > limits.max_temperature)):
        raise ValueError("temperature must be finite and within the configured temperature bounds")
    if n_ctx is not None and (type(n_ctx) is not int or n_ctx < 1):
        raise ValueError("n_ctx must be a positive integer")
    cfg = MODELS[model] if model is not None else None
    context = n_ctx if n_ctx is not None else (cfg["n_ctx"] if cfg else DEFAULT_CONTEXT)
    if limits.enforce_model_limits and cfg and cfg["max_n_ctx"] and context > cfg["max_n_ctx"]:
        raise ValueError(f"{model} supports n_ctx up to {cfg['max_n_ctx']:,} tokens; requested {context:,}")
    if (limits.enforce_model_limits and cfg and cfg["max_output_tokens"] is not None
            and max_tokens is not None and max_tokens > cfg["max_output_tokens"]):
        raise ValueError(f"{model} supports max_tokens up to {cfg['max_output_tokens']:,}")
    if limits.enforce_output_budget and max_tokens is not None and max_tokens >= context:
        raise ValueError(f"max_tokens={max_tokens} must be less than n_ctx={context}")
    if cfg:
        resources = model_resources(model, context, cache, limits=limits)
        if limits.check_resources and resources["ram_gb"] > runner_ram_gb:
            raise ValueError(
                f"{model} at n_ctx={context} needs approximately {resources['ram_gb']:.1f} GB RAM; "
                f"configured budget is {runner_ram_gb:g} GB. Use a smaller model/context or "
                "configure runner_ram_gb for a runner that actually has more memory."
            )
        if limits.check_resources and resources["disk_gb"] > runner_disk_gb:
            raise ValueError(
                f"{model} needs approximately {resources['disk_gb']:.1f} GB free disk "
                f"including cache overhead; configured budget is {runner_disk_gb:g} GB. "
                "Use cache=False or a runner with more disk."
            )


def validate_text(prompt, system, cache, limits=None):
    limits = limits if limits is not None else Limits()
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if not isinstance(system, str):
        raise TypeError("system must be a string")
    if type(cache) is not bool:
        raise TypeError("cache must be a bool")
    # Leave space for the other workflow inputs, including provider configuration.
    if limits.max_prompt_chars is not None and len(prompt) + len(system) > limits.max_prompt_chars:
        raise ValueError(f"Combined prompt and system text must not exceed {limits.max_prompt_chars:,} characters")

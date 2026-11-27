from .models import (
    MODELS,
    RUNNER_RAM_GB,
    RAM_PER_1K_CTX_GB,
    MAX_TOKENS_LIMIT,
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
)


def _validate(model, max_tokens, temperature, n_ctx):
    cfg    = MODELS[model]
    errors = []

    if not (MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE):
        errors.append(
            f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, got {temperature}"
        )

    if max_tokens < 1:
        errors.append(f"max_tokens must be at least 1, got {max_tokens}")
    if max_tokens > MAX_TOKENS_LIMIT:
        errors.append(f"max_tokens={max_tokens} exceeds hard limit of {MAX_TOKENS_LIMIT}.")

    effective_n_ctx = n_ctx or cfg["n_ctx"]
    if effective_n_ctx > cfg["max_n_ctx"]:
        errors.append(
            f"n_ctx={effective_n_ctx} exceeds safe limit of {cfg['max_n_ctx']}. Risk of OOM."
        )
    if max_tokens >= effective_n_ctx:
        errors.append(
            f"max_tokens={max_tokens} must be less than n_ctx={effective_n_ctx}."
        )

    base_ram  = cfg["size_gb"]
    extra_ctx = max(0, effective_n_ctx - cfg["n_ctx"])
    total_ram = base_ram + (extra_ctx / 1000) * RAM_PER_1K_CTX_GB + 1.0
    if total_ram > RUNNER_RAM_GB:
        errors.append(
            f"Estimated RAM ({total_ram:.1f} GB) exceeds runner limit ({RUNNER_RAM_GB} GB)."
        )

    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    if max_tokens > 2048:
        print(f"[warn] max_tokens={max_tokens} is large — expect slower inference.")
    if effective_n_ctx > 4096:
        print(f"[warn] n_ctx={effective_n_ctx} is large — estimated RAM: {total_ram:.1f} GB.")
    if temperature > 1.2:
        print(f"[warn] temperature={temperature} is high — output may be incoherent.")
"""Small, explicitly routed plans for one resident inference worker."""


def normalize_pipeline(steps, max_prompts=None):
    """Validate before dispatch; references name earlier outputs, never prompt copies."""
    if not isinstance(steps, (list, tuple)) or not steps:
        raise ValueError("steps must be a nonempty list or tuple")
    if max_prompts is not None and len(steps) > max_prompts:
        raise ValueError("Pipeline exceeds configured max_prompts")
    normalized = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or set(step) - {"agent", "instruction", "inputs"}:
            raise ValueError("Each step accepts only agent, instruction, and inputs")
        for name in ("agent", "instruction"):
            if not isinstance(step.get(name), str) or not step[name].strip():
                raise ValueError(f"Step {index} needs a nonempty {name}")
        sources = step.get("inputs", [index - 1] if index else [])
        if not isinstance(sources, (list, tuple)) or any(
            type(source) is not int or not 0 <= source < index for source in sources
        ):
            raise ValueError(f"Step {index} inputs must reference earlier step indices")
        normalized.append({"agent": step["agent"], "instruction": step["instruction"],
                           "inputs": list(dict.fromkeys(sources))})
    return normalized

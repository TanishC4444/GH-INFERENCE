"""Text-agent implementation embedded in the GitHub worker only."""

AGENT_SCRIPT = r'''
import re
from functools import lru_cache


def run_agent(agent, initial_messages, inference_config):
    started = time.monotonic()
    trim_history = agent.get("trim_history", True)
    if type(trim_history) is not bool:
        raise ValueError("trim_history must be a bool")
    policy = dict(DEFAULT_LIMITS)
    policy.update(inference_config.get("limits", {}))
    local_model = os.environ.get("LOCAL_MODEL", "qwen0.5b")
    context = inference_config.get("n_ctx") or model_config(local_model)["n_ctx"]
    maximum = inference_config.get("max_tokens", 512)
    temperature = float(inference_config.get("temperature", .7))
    key = (local_model, context)
    was_loaded = key in _LOADED_MODELS
    llm = get_local_model(local_model, context)
    stats = _MODEL_STATS[key]
    tasks = [initial_messages[-1]["content"]] + agent.get("tasks", [])
    if (not tasks or (policy["max_prompts"] is not None and len(tasks) > policy["max_prompts"])
            or any(not isinstance(task, str) or not task.strip() for task in tasks)):
        raise ValueError("Agent prompts are empty, invalid, or exceed configured max_prompts")
    max_turns = agent.get("max_history_turns", 6)
    if max_turns is not None and (type(max_turns) is not int or max_turns < 1):
        raise ValueError("max_history_turns must be a positive integer or None")
    history = list(agent.get("history", [])) if agent.get("remember", True) else []
    if len(history) % 2 or any(
        not isinstance(item, dict) or item.get("role") != ("user" if i % 2 == 0 else "assistant")
        or not isinstance(item.get("content"), str) for i, item in enumerate(history)
    ):
        raise ValueError("Invalid conversation history")
    if trim_history and max_turns is not None:
        history = history[-2 * max_turns:]
    system = initial_messages[0]
    @lru_cache(maxsize=256)
    def count(text):
        return len(llm.tokenize(text.encode(), add_bos=False))
    def fits(messages, output):
        return (policy["context_reserve_tokens"] + sum(count(item["content"]) +
                policy["message_overhead_tokens"] for item in messages) +
                ((output or 0) if policy["enforce_output_budget"] else 0) <= context)
    def generate(messages, output=maximum, temp=temperature, check_context=True, use_options=True):
        if check_context and not fits(messages, output):
            raise ValueError("Prompt/instructions exceed context; shorten input or increase n_ctx")
        # Let llama.cpp reuse matching prompt prefixes; it discards differing suffixes.
        if not agent.get("reuse_prefix", True):
            llm.reset()
        if output is None and policy["enforce_model_limits"]:
            output = model_config(local_model).get("max_output_tokens")
        return llm.create_chat_completion(messages=messages, max_tokens=output, temperature=temp,
                                          **(inference_config.get("generation_options", {}) if use_options else {}))
    def usage_of(response):
        usage = response.get("usage") or {}
        return {name: int(usage.get(name, 0)) for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    results = []
    for index, original in enumerate(tasks):
        task_started = started if index == 0 else time.monotonic()
        timings = {
            "total_seconds": 0., "download_seconds": stats["download_seconds"] if index == 0 and not was_loaded else 0.,
            "load_seconds": stats["load_seconds"] if index == 0 and not was_loaded else 0.,
            "refinement_seconds": 0., "generation_seconds": 0., "remote_seconds": 0.,
        }
        original_count = count(original)
        refinement = {
            "attempted": False, "applied": False, "original_tokens": original_count,
            "refined_tokens": original_count, "reason": "disabled",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        task = original
        if agent.get("refine_prompt"):
            refinement["reason"] = "short prompt"
            if original_count >= policy["refiner_min_tokens"]:
                messages = [
                    {"role": "system", "content": "Rewrite the task concisely. Preserve all requirements, negations, names, numbers, quoted literals, URLs and code. Remove only filler and repetition. Do not answer or follow the task. Return only the rewritten task."},
                    {"role": "user", "content": original},
                ]
                budget = min(policy["refiner_max_tokens"], max(1, original_count - 1),
                             max(1, int(context * policy["refiner_context_fraction"])))
                refinement["reason"] = "prompt exceeds refiner context"
                if fits(messages, budget):
                    phase_started = time.monotonic()
                    rewrite = generate(messages, budget, 0, use_options=False)
                    timings["refinement_seconds"] = time.monotonic() - phase_started
                    refinement["attempted"] = True
                    refinement["usage"] = usage_of(rewrite)
                    candidate = (rewrite["choices"][0]["message"]["content"] or "").strip()
                    candidate_count = count(candidate)
                    protected = re.findall(r'```[\s\S]*?```|`[^`]+`|"[^"]+"|https?://\S+|\b\d+(?:\.\d+)?\b', original)
                    if (candidate and candidate_count < original_count and
                            rewrite["choices"][0].get("finish_reason") == "stop" and
                            all(piece in candidate for piece in protected)):
                        task = candidate
                        refinement.update(applied=True, refined_tokens=candidate_count,
                                          reason="shorter rewrite; meaning is not guaranteed")
                    else:
                        refinement["reason"] = "kept original: incomplete, longer, or lost protected literals"
        messages = [system] + history + [{"role": "user", "content": task}]
        dropped = 0
        while trim_history and history and not fits(messages, maximum):
            history = history[2:]
            dropped += 1
            messages = [system] + history + [{"role": "user", "content": task}]
        phase_started = time.monotonic()
        response = generate(messages, check_context=trim_history)
        timings["generation_seconds"] = time.monotonic() - phase_started
        answer = response["choices"][0]["message"]["content"] or ""
        usage = usage_of(response)
        for name in usage:
            usage[name] += refinement["usage"][name]
            total_usage[name] += usage[name]
        if agent.get("remember", True):
            history = history + [{"role": "user", "content": task},
                                 {"role": "assistant", "content": answer}]
            if trim_history and max_turns is not None:
                history = history[-2 * max_turns:]
        timings["total_seconds"] = time.monotonic() - task_started
        results.append({
            "output": answer, "usage": usage, "refinement": refinement,
            "timings": timings if agent.get("track_time") else None,
            "model_reused": was_loaded or index > 0, "weights_cached": stats["weights_cached"],
            "dropped_history_turns": dropped,
            "finish_reason": response["choices"][0].get("finish_reason", "stop"),
        })
    return {"results": results, "usage": total_usage, "history": history}


def run_discussion(agent, initial_messages, inference_config):
    options = agent["discussion"]
    count, rounds, peer_chars = (options.get(name) for name in ("agents", "rounds", "peer_chars"))
    if any(type(value) is not int or value < 1 for value in (count, rounds, peer_chars)):
        raise ValueError("Discussion counts must be positive integers")
    topology = options.get("topology", "all")
    if topology not in ("all", "ring"):
        raise ValueError("topology must be all or ring")
    policy = dict(DEFAULT_LIMITS)
    policy.update(inference_config.get("limits", {}))
    if policy["max_prompts"] is not None and count * rounds > policy["max_prompts"]:
        raise ValueError("Discussion exceeds configured max_prompts")
    task = initial_messages[-1]["content"]
    turn_config = dict(agent, tasks=[], remember=False, history=[])
    turn_config.pop("discussion")
    results, previous = [], []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for round_index in range(rounds):
        current = []
        for index in range(count):
            peers = ([previous[(index - 1) % count]] if topology == "ring" and previous else previous)
            shared = "\n".join("Agent %d: %s" % (peer, reply[:peer_chars]) for peer, reply in peers)
            prompt = ("Task: %s\nPrevious round replies:\n%s\n"
                      "You are Agent %d, round %d. Propose a distinct improvement and test. "
                      "Address another agent by number when replies are provided."
                      % (task, shared, index, round_index + 1))
            if policy["max_prompt_chars"] is not None and len(prompt) + len(initial_messages[0]["content"]) > policy["max_prompt_chars"]:
                raise ValueError("Discussion prompt exceeds configured max_prompt_chars")
            messages = [initial_messages[0], {"role": "user", "content": prompt}]
            response = run_agent(turn_config, messages, inference_config)
            item = response["results"][0]
            results.append(item)
            current.append((index, item["output"]))
            for name in usage:
                usage[name] += response["usage"][name]
        previous = current
    return {"results": results, "usage": usage, "history": []}


def run_pipeline(agent, initial_messages, inference_config):
    policy = dict(DEFAULT_LIMITS)
    policy.update(inference_config.get("limits", {}))
    steps = normalize_pipeline(agent["pipeline"], policy["max_prompts"])
    # Replies are stored once in results. Recipients reference them by step index;
    # rendered source blocks are cached, not regenerated for every recipient.
    results, rendered = [], {}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    turn_config = dict(agent, tasks=[], remember=False, history=[])
    turn_config.pop("pipeline")
    shared_system = initial_messages[0]["content"] + "\nShared task:\n" + initial_messages[-1]["content"]
    for index, step in enumerate(steps):
        routing_started = time.monotonic()
        for source in step["inputs"]:
            if source not in rendered:
                rendered[source] = ("[Step %d / %s]\n%s" %
                                    (source, steps[source]["agent"], results[source]["output"]))
        handoff = "\n\n".join(rendered[source] for source in step["inputs"])
        # Keep common source text ahead of varying role/instruction text so
        # broadcast recipients can reuse the same computed prompt prefix.
        system = shared_system
        prompt = (("Provided outputs:\n" + handoff + "\n\n") if handoff else "")
        prompt += "Your role: " + step["agent"] + "\n" + step["instruction"]
        if policy["max_prompt_chars"] is not None and len(system) + len(prompt) > policy["max_prompt_chars"]:
            raise ValueError("Pipeline prompt exceeds configured max_prompt_chars")
        handoff_chars, handoff_bytes = len(handoff), len(handoff.encode("utf-8"))
        routing_seconds = time.monotonic() - routing_started
        response = run_agent(turn_config, [{"role": "system", "content": system},
                                          {"role": "user", "content": prompt}], inference_config)
        item = response["results"][0]
        item.update(agent_id=step["agent"], step_index=index, input_steps=step["inputs"],
                    handoff_chars=handoff_chars, handoff_bytes=handoff_bytes)
        if item["timings"] is not None:
            item["timings"]["routing_seconds"] = routing_seconds
            item["timings"]["total_seconds"] += routing_seconds
        results.append(item)
        for name in usage:
            usage[name] += response["usage"][name]
    return {"results": results, "usage": usage, "history": []}
'''

"""Verified Q4_K_M GGUF downloads, with conservative CPU inference estimates."""

RUNNER_RAM_GB = 8
RUNNER_DISK_GB = 14
RAM_PER_1K_CTX_GB = 0.25
RUNTIME_OVERHEAD_GB = 1.5
DISK_RESERVE_GB = 2.0
MAX_CACHED_MODEL_GB = 6.0
DEFAULT_CONTEXT = 8192
MAX_TEMPERATURE = 2.0
MIN_TEMPERATURE = 0.0


def _gguf(name, parameters_b, repo, filename, size_bytes, sha256, n_ctx=DEFAULT_CONTEXT,
          ram_per_1k_ctx_gb=0.25, *, max_n_ctx, context_source,
          max_output_tokens=None, yarn_original_context=None, revision="main"):
    return {
        "name": name, "parameters_b": parameters_b, "quantization": "Q4_K_M",
        "size_bytes": size_bytes, "size_gb": size_bytes / 1e9,
        "sha256": sha256, "n_ctx": min(n_ctx, max_n_ctx), "max_n_ctx": max_n_ctx,
        "context_source": context_source,
        # None means the publisher specifies no separate generation ceiling.
        "max_output_tokens": max_output_tokens,
        "yarn_original_context": yarn_original_context,
        "ram_per_1k_ctx_gb": ram_per_1k_ctx_gb,
        "url": f"https://huggingface.co/{repo}/resolve/{revision}/{filename}",
        "filename": filename,
    }


MODELS = {
    'qwen0.5b': _gguf(
        'Qwen 2.5 0.5B Instruct', 0.5,
        'Qwen/Qwen2.5-0.5B-Instruct-GGUF',
        'qwen2.5-0.5b-instruct-q4_k_m.gguf',
        491400032, '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db',
        ram_per_1k_ctx_gb=0.125, max_n_ctx=32768, max_output_tokens=8192,
        context_source='https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct',
    ),
    'tinyllama': _gguf(
        'TinyLlama 1.1B Chat', 1.1,
        'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
        'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
        668788096, '9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0',
        max_n_ctx=2048,
        context_source='https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/main/config.json',
    ),
    'llama': _gguf(
        'Llama 3.2 1B Instruct', 1,
        'bartowski/Llama-3.2-1B-Instruct-GGUF',
        'Llama-3.2-1B-Instruct-Q4_K_M.gguf',
        807694464, '6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83',
        max_n_ctx=131072,
        context_source='https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct',
    ),
    'phi3': _gguf(
        'Phi-3.5 Mini Instruct', 3.8,
        'bartowski/Phi-3.5-mini-instruct-GGUF',
        'Phi-3.5-mini-instruct-Q4_K_M.gguf',
        2393232672, 'e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5',
        ram_per_1k_ctx_gb=0.4, max_n_ctx=131072,
        context_source='https://huggingface.co/microsoft/Phi-3.5-mini-instruct',
    ),
    'qwen': _gguf(
        'Qwen 2.5 1.5B Instruct', 1.5,
        'Qwen/Qwen2.5-1.5B-Instruct-GGUF',
        'qwen2.5-1.5b-instruct-q4_k_m.gguf',
        1117320736, '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e',
        max_n_ctx=32768, max_output_tokens=8192,
        context_source='https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct',
    ),
    'gemma2': _gguf(
        'Gemma 2 2B Instruct', 2,
        'bartowski/gemma-2-2b-it-GGUF',
        'gemma-2-2b-it-Q4_K_M.gguf',
        1708582752, 'e0aee85060f168f0f2d8473d7ea41ce2f3230c1bc1374847505ea599288a7787',
        max_n_ctx=8192,
        context_source='https://ai.google.dev/gemma/docs/core/model_card_2',
    ),
    'deepseek': _gguf(
        'DeepSeek R1 Distill Qwen 1.5B', 1.5,
        'bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF',
        'DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf',
        1117320800, '1741e5b2d062b07acf048bf0d2c514dadf2a48f94e2b4aa0cfe069af3838ee2f',
        max_n_ctx=131072,
        context_source='https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/blob/main/config.json',
    ),
    'qwen3b': _gguf(
        'Qwen 2.5 3B Instruct', 3,
        'bartowski/Qwen2.5-3B-Instruct-GGUF',
        'Qwen2.5-3B-Instruct-Q4_K_M.gguf',
        1929903264, '9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94',
        max_n_ctx=32768, max_output_tokens=8192,
        context_source='https://huggingface.co/Qwen/Qwen2.5-3B-Instruct',
    ),
    'qwen7b': _gguf(
        'Qwen 2.5 7B Instruct', 7,
        'bartowski/Qwen2.5-7B-Instruct-GGUF',
        'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
        4683074240, '65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423',
        max_n_ctx=131072, max_output_tokens=8192, yarn_original_context=32768,
        context_source='https://huggingface.co/Qwen/Qwen2.5-7B-Instruct',
    ),
    'qwen-coder7b': _gguf(
        'Qwen 2.5 Coder 7B Instruct', 7,
        'bartowski/Qwen2.5-Coder-7B-Instruct-GGUF',
        'Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf',
        4683074336, '1664fccab734674a50763490a8c6931b70e3f2f8ec10031b54806d30e5f956b6',
        max_n_ctx=131072, yarn_original_context=32768,
        context_source='https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct',
    ),
    'llama8b': _gguf(
        'Llama 3.1 8B Instruct', 8,
        'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF',
        'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
        4920739232, '7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c',
        max_n_ctx=131072,
        context_source='https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct',
    ),
    'gemma9b': _gguf(
        'Gemma 2 9B Instruct', 9,
        'bartowski/gemma-2-9b-it-GGUF',
        'gemma-2-9b-it-Q4_K_M.gguf',
        5761057728, '13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe',
        max_n_ctx=8192,
        context_source='https://ai.google.dev/gemma/docs/core/model_card_2',
    ),
    'mistral12b': _gguf(
        'Mistral Nemo 12B Instruct', 12,
        'bartowski/Mistral-Nemo-Instruct-2407-GGUF',
        'Mistral-Nemo-Instruct-2407-Q4_K_M.gguf',
        7477208192, '7c1a10d202d8788dbe5628dc962254d10654c853cae6aaeca0618f05490d4a46',
        max_n_ctx=131072,
        context_source='https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407',
    ),
    'qwen14b': _gguf(
        'Qwen 2.5 14B Instruct', 14,
        'bartowski/Qwen2.5-14B-Instruct-GGUF',
        'Qwen2.5-14B-Instruct-Q4_K_M.gguf',
        8988110976, 'e47ad95dad6ff848b431053b375adb5d39321290ea2c638682577dafca87c008',
        max_n_ctx=131072, max_output_tokens=8192, yarn_original_context=32768,
        context_source='https://huggingface.co/Qwen/Qwen2.5-14B-Instruct',
    ),
    'qwen-coder14b': _gguf(
        'Qwen 2.5 Coder 14B Instruct', 14,
        'bartowski/Qwen2.5-Coder-14B-Instruct-GGUF',
        'Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf',
        8988111072, '2946d28c9e1bb2bcae6d42e8678863a31775df6f740315c7d7e6d6b6411f5937',
        max_n_ctx=131072, yarn_original_context=32768,
        context_source='https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct',
    ),
    'deepseek-coder16b': _gguf(
        'DeepSeek Coder V2 Lite Instruct', 16,
        'bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF',
        'DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf',
        10364416768, '603bd3f8a0281d16571da7c08bd661ee17ff0d1be6fcbd1b42242da257ef0bb8',
        ram_per_1k_ctx_gb=0.5, max_n_ctx=131072,
        context_source='https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
    ),
    'gemma-4-styletune-12b': _gguf(
        'Gemma 4 12B StyleTune', 12.9,
        'mradermacher/Gemma-4-12B-StyleTune-GGUF',
        'Gemma-4-12B-StyleTune.Q4_K_M.gguf',
        7947613824, 'd3e487063f62cf1ed37f7317c93a8e709e6e072a877779ff86b7a73fe904c4a3',
        max_n_ctx=262144, revision='fb2e8be6efb2905308e57ce248f209c6ddf99983',
        context_source='https://huggingface.co/mradermacher/Gemma-4-12B-StyleTune-GGUF',
    ),
    'rocinante-x-12b': _gguf(
        'Rocinante-X 12B v1', 12,
        'bartowski/TheDrummer_Rocinante-X-12B-v1-GGUF',
        'TheDrummer_Rocinante-X-12B-v1-Q4_K_M.gguf',
        7477207712, '1ca25853a19d6b15f9a94f7e5bc86f4cc88bd0c9dea12a00ea55452b4208ade6',
        max_n_ctx=131072, revision='4302599a9fecd0e250c3790d77a38f3bd791fdd4',
        context_source='https://huggingface.co/bartowski/TheDrummer_Rocinante-X-12B-v1-GGUF',
    ),
    'glitter-12b': _gguf(
        'Gemma 3 Glitter 12B', 12,
        'duyntnet/Gemma-3-Glitter-12B-imatrix-GGUF',
        'Gemma-3-Glitter-12B-Q4_K_M.gguf',
        7300779616, 'fa414e89caf6cf22d1b71fa9a5e13c934d431b99607606139915bf8608f39719',
        max_n_ctx=131072, revision='417cddf45b8a6f8977920b251258ec5f58855124',
        context_source='https://huggingface.co/duyntnet/Gemma-3-Glitter-12B-imatrix-GGUF',
    ),
    'gemma-4-12b': _gguf(
        'Gemma 4 12B Instruct', 12,
        'unsloth/gemma-4-12b-it-GGUF',
        'gemma-4-12b-it-Q4_K_M.gguf',
        7121861440, '0a270ec9fe6b34f4a0d33992b6135117b484ebc4766ab76b51d4ae8c457e4c42',
        max_n_ctx=262144, revision='fc034cfff751157913579611efad8462ac1be606',
        context_source='https://huggingface.co/unsloth/gemma-4-12b-it-GGUF',
    ),
}

# Backwards-compatible export: largest catalog output ceiling, not a limit
# shared by every model. Validation uses the selected model's metadata.
MAX_TOKENS_LIMIT = max(cfg["max_output_tokens"] or cfg["max_n_ctx"] - 1 for cfg in MODELS.values())


def model_resources(model, n_ctx=None, cache=True, *, limits=None):
    """Return conservative decimal-GB budgets, not measured peak usage."""
    cfg = MODELS[model]
    from .limits import Limits
    limits = limits if limits is not None else Limits()
    context = cfg["n_ctx"] if n_ctx is None else n_ctx
    if type(context) is not int or context < 1:
        raise ValueError("n_ctx must be a positive integer")
    if limits.enforce_model_limits and cfg["max_n_ctx"] and context > cfg["max_n_ctx"]:
        raise ValueError("n_ctx exceeds the model's advertised context; set enforce_model_limits=False to experiment")
    if type(cache) is not bool:
        raise TypeError("cache must be a bool")
    cache_weights = cache and (limits.max_cached_model_gb is None or cfg["size_gb"] < limits.max_cached_model_gb)
    return {
        "ram_gb": cfg["size_gb"] + limits.runtime_overhead_gb + context / 1024 * (
            cfg["ram_per_1k_ctx_gb"] if limits.ram_per_1k_ctx_gb is None else limits.ram_per_1k_ctx_gb),
        "disk_gb": cfg["size_gb"] * (2 if cache_weights else 1) + limits.disk_reserve_gb,
        "cache_weights": cache_weights,
    }
